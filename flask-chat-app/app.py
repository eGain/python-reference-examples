import os
import json
import base64
import logging
import secrets

import markdown
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)
from authlib.integrations.flask_client import OAuth

from egain_api_python import Egain
from egain_api_python.errors import EgainError

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Optional: import SDK error to log response body when validation fails
try:
    from egain_api_python.errors.responsevalidationerror import ResponseValidationError as EgainResponseValidationError
except ImportError:
    EgainResponseValidationError = None


def _clear_session_and_require_login():
    """Clear session and return JSON response asking the user to log in again."""
    session.pop("token", None)
    session.pop("user", None)
    return jsonify({"error": "Please log in again.", "login_required": True}), 401


app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY", secrets.token_hex(32))

# OAuth (PKCE) 
oauth = OAuth(app)
egain_auth = oauth.register(
    name="egain",
    client_id=os.getenv("EGAIN_CLIENT_ID"),
    client_secret=None,
    access_token_url=os.getenv("EGAIN_TOKEN_URL"),
    authorize_url=os.getenv("EGAIN_AUTH_URL"),
    client_kwargs={
        "scope": "knowledge.portalmgr.read core.aiservices.read",
        "code_challenge_method": "S256",
        "token_endpoint_auth_method": "none",
    },
)

# Config
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.3"))
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
TOP_K = 5
CHANNEL_NAME = "Eight Bank Website"
MAX_CHAT_HISTORY_MESSAGES = 20  # last N messages (user + assistant) for multi-turn
MAX_ACCUMULATED_CHUNKS = 25  # max chunks to keep across turns for fallback when threshold not met


def _chunk_snippet(c):
    """Get snippet from chunk (dict or object)."""
    if isinstance(c, dict):
        return c.get("snippet") or ""
    return getattr(c, "snippet", None) or ""


def _chunk_summary(c):
    """Get contextual summary from chunk (dict or object)."""
    if isinstance(c, dict):
        return c.get("contextualSummary") or c.get("contextual_summary") or ""
    return getattr(c, "contextual_summary", None) or getattr(c, "contextualSummary", None) or ""


def _chunk_article_id(c):
    """Get article id from chunk (dict or object)."""
    if isinstance(c, dict):
        return c.get("article_id") or c.get("articleId") or c.get("id")
    return getattr(c, "article_id", None) or getattr(c, "articleId", None) or getattr(c, "id", None)


def _chunk_name(c):
    """Get display name from chunk (dict or object)."""
    if isinstance(c, dict):
        return c.get("name") or c.get("article_name") or c.get("articleName")
    return getattr(c, "name", None) or getattr(c, "article_name", None) or getattr(c, "articleName", None)


def chunk_to_storable(chunk):
    """Convert API chunk to a dict we can store in session (accumulated_chunks)."""
    aid = _chunk_article_id(chunk)
    return {
        "snippet": _chunk_snippet(chunk),
        "contextualSummary": _chunk_summary(chunk),
        "article_id": str(aid) if aid is not None else None,
        "name": _chunk_name(chunk) or (f"Article {aid}" if aid else "Unknown"),
    }


def get_egain_client():
    token = session.get("token")
    if not token:
        return None
    return Egain(
        access_token=token.get("access_token"),
        server_url=os.getenv("EGAIN_SERVER_URL"), 
    )


def decode_jwt_payload(token_string):
    try:
        parts = token_string.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return None


def get_user_info_from_token(token):
    user_info = {}
    if "id_token" in token:
        payload = decode_jwt_payload(token["id_token"])
        if payload:
            user_info["name"] = (
                payload.get("name")
                or f"{payload.get('given_name', '')} {payload.get('family_name', '')}".strip()
                or payload.get("preferred_username")
                or payload.get("email")
                or payload.get("sub", "User")
            )
            user_info["email"] = payload.get("email")
            user_info["username"] = payload.get("preferred_username") or payload.get("sub")
            if user_info.get("name"):
                return user_info
    if "access_token" in token:
        payload = decode_jwt_payload(token["access_token"])
        if payload:
            user_info["name"] = (
                payload.get("name")
                or f"{payload.get('given_name', '')} {payload.get('family_name', '')}".strip()
                or payload.get("preferred_username")
                or payload.get("email")
                or payload.get("username")
                or payload.get("sub", "User")
            )
            user_info["email"] = payload.get("email")
            user_info["username"] = payload.get("preferred_username") or payload.get("username") or payload.get("sub")
            if user_info.get("name"):
                return user_info

    return None


def get_relevance_score(chunk):
    """Get relevance score from chunk (dict or object)."""
    if isinstance(chunk, dict):
        return chunk.get("relevanceScore") or chunk.get("relevance_score") or chunk.get("score")
    return (
        getattr(chunk, "relevanceScore", None) or getattr(chunk, "relevance_score", None) or getattr(chunk, "score", None)
    )


def build_context_from_results(picked):
    """Build context string from snippet and contextualSummary for each picked result (dict or object)."""
    parts = []
    for i, r in enumerate(picked, 1):
        snippet = _chunk_snippet(r)
        summary = _chunk_summary(r)
        if isinstance(snippet, str) and snippet.strip():
            parts.append(f"[Source {i}]\n{snippet.strip()}")
        if isinstance(summary, str) and summary.strip():
            parts.append(f"[Source {i} summary]\n{summary.strip()}")
    return "\n\n".join(parts) if parts else "No relevant passages found."


def sources_from_chunks(chunks):
    """Build unique sources list from chunks (dict or object)."""
    seen_ids = set()
    sources = []
    for r in chunks:
        aid = _chunk_article_id(r)
        if aid is not None:
            aid = str(aid)
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        name = _chunk_name(r) or (f"Article {aid}" if aid else "Unknown")
        sources.append({"id": aid or "", "name": name})
    return sources


def _serialize_retrieve_response(response):
    """Convert retrieve API response to a JSON-serializable dict for debug display."""
    if response is None:
        return None
    try:
        if hasattr(response, "model_dump"):
            return response.model_dump(by_alias=True, exclude_none=False)
        if hasattr(response, "dict"):
            return response.dict(by_alias=True, exclude_none=False)
    except Exception:
        pass
    # Fallback: manual extract
    out = {}
    search_results = getattr(response, "search_results", None) or getattr(response, "searchResults", None)
    if search_results is not None:
        out["searchResults"] = [_serialize_obj(c) for c in search_results]
    for attr in ("session_id", "sessionId", "answer", "channel", "event_id", "eventId", "client_session_id", "clientSessionId"):
        val = getattr(response, attr, None)
        if val is not None:
            out[attr] = _serialize_obj(val)
    return out if out else {"_raw": str(response)}


def _serialize_obj(obj):
    """Recursively serialize for JSON (Pydantic models, list, dict)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_serialize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(by_alias=True, exclude_none=False)
        if hasattr(obj, "dict"):
            return obj.dict(by_alias=True, exclude_none=False)
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        return {k: _serialize_obj(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def call_llm(user_query: str, context: str, history: list | None = None) -> str:
    """Call configured LLM with optional conversation history for multi-turn."""
    system_prompt = """You are a helpful assistant that answers questions using only the provided context from a knowledge base.
Use the given context passages to answer the user's question. If the context does not contain enough information, say so clearly.
Do not invent information. Prefer quoting or paraphrasing from the context. Keep answers concise but complete.
When the user asks a follow-up question, you may use the latest context and the previous conversation to give a coherent answer."""

    current_turn = f"""Context from the knowledge base:

{context}

---

User question: {user_query}

Please answer based only on the context above."""

    history = history or []
    history = history[-MAX_CHAT_HISTORY_MESSAGES:] if len(history) > MAX_CHAT_HISTORY_MESSAGES else history

    if LLM_PROVIDER == "openai":
        return _call_openai(system_prompt, current_turn, history)
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, current_turn, history)
    if LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, current_turn, history)
    return f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Set LLM_PROVIDER to openai, anthropic, or gemini."


def _call_openai(system_prompt: str, current_turn: str, history: list) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": current_turn})
        r = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logging.exception("OpenAI call failed")
        return f"Error calling OpenAI: {e}"


def _call_anthropic(system_prompt: str, current_turn: str, history: list) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": current_turn})
        r = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        if r.content and len(r.content) > 0:
            return (r.content[0].text or "").strip()
        return ""
    except Exception as e:
        logging.exception("Anthropic call failed")
        return f"Error calling Anthropic: {e}"


def _call_gemini(system_prompt: str, current_turn: str, history: list) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            system_instruction=system_prompt,
        )
        if history:
            parts = ["Previous conversation:\n"]
            for m in history:
                parts.append(f"{m['role'].capitalize()}: {m['content']}\n")
            parts.append("\n---\n\nCurrent question and context:\n\n")
            parts.append(current_turn)
            user_message = "".join(parts)
        else:
            user_message = current_turn
        r = model.generate_content(user_message)
        if r and r.text:
            return r.text.strip()
        return "No response from Gemini."
    except Exception as e:
        logging.exception("Gemini call failed")
        return f"Error calling Gemini: {e}"


def _openai_stream(system_prompt: str, current_turn: str, history: list):
    """Yield text chunks from OpenAI stream."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": current_turn})
    stream = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield delta.content


def _anthropic_stream(system_prompt: str, current_turn: str, history: list):
    """Yield text chunks from Anthropic stream."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": current_turn})
    with client.messages.stream(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _gemini_stream(system_prompt: str, current_turn: str, history: list):
    """Yield text chunks from Gemini stream."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        system_instruction=system_prompt,
    )
    if history:
        parts = ["Previous conversation:\n"]
        for m in history:
            parts.append(f"{m['role'].capitalize()}: {m['content']}\n")
        parts.append("\n---\n\nCurrent question and context:\n\n")
        parts.append(current_turn)
        user_message = "".join(parts)
    else:
        user_message = current_turn
    for chunk in model.generate_content(user_message, stream=True):
        if chunk.text:
            yield chunk.text


def call_llm_stream(user_query: str, context: str, history: list | None = None):
    """Yield text chunks from the configured LLM (streaming)."""
    system_prompt = """You are a helpful assistant that answers questions using only the provided context from a knowledge base.
Use the given context passages to answer the user's question. If the context does not contain enough information, say so clearly.
Do not invent information. Prefer quoting or paraphrasing from the context. Keep answers concise but complete.
When the user asks a follow-up question, you may use the latest context and the previous conversation to give a coherent answer."""

    current_turn = f"""Context from the knowledge base:

{context}

---

User question: {user_query}

Please answer based only on the context above."""

    history = history or []
    history = history[-MAX_CHAT_HISTORY_MESSAGES:] if len(history) > MAX_CHAT_HISTORY_MESSAGES else history

    if LLM_PROVIDER == "openai":
        yield from _openai_stream(system_prompt, current_turn, history)
    elif LLM_PROVIDER == "anthropic":
        yield from _anthropic_stream(system_prompt, current_turn, history)
    elif LLM_PROVIDER == "gemini":
        yield from _gemini_stream(system_prompt, current_turn, history)
    else:
        yield f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Set LLM_PROVIDER to openai, anthropic, or gemini."


# --- Routes ---

def _chat_history_for_display():
    """Build chat history for template: assistant content as HTML, with sources."""
    out = []
    for m in session.get("chat_history", []):
        content = m.get("content") or ""
        if m.get("role") == "assistant":
            content = markdown.markdown(content) if content else ""
        out.append({
            "role": m.get("role", "user"),
            "content": content,
            "sources": m.get("sources") or [],
        })
    return out


@app.route("/")
def index():
    if session.get("token"):
        return render_template(
            "chat.html",
            user=session.get("user"),
            chat_history=_chat_history_for_display(),
        )
    return render_template("index.html")


@app.route("/login")
def login():
    return egain_auth.authorize_redirect(url_for("callback", _external=True))


@app.route("/callback")
def callback():
    token = egain_auth.authorize_access_token()
    session["token"] = token
    user_info = get_user_info_from_token(token)
    session["user"] = user_info or {"name": "Authenticated User"}
    # With Flask's default client-side (cookie) session, the session is sent with the redirect.
    # If you switch to a server-side store that requires an explicit save, save first and redirect in the callback.
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    """Clear session and redirect to index."""
    session.pop("token", None)
    session.pop("user", None)
    return redirect(url_for("index"))


@app.route("/chat/clear", methods=["POST"])
def chat_clear():
    """Clear conversation history and accumulated chunks; used by New chat button."""
    if not session.get("token"):
        return jsonify({"error": "Unauthorized"}), 401
    session.pop("chat_history", None)
    session.pop("accumulated_chunks", None)
    session.modified = True
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    if not session.get("token"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        client = get_egain_client()
        if not client:
            return jsonify({"error": "Not authenticated"}), 401

        with client as egain:
            portal_id = os.getenv("EGAIN_PORTAL_ID", "PZ-9999")
            language = os.getenv("EGAIN_LANGUAGE", "en-US")
            response = egain.aiservices.retrieve.retrieve_chunks(
                q=query,
                portal_id=portal_id,
                language=language,
                channel={"name": CHANNEL_NAME},
            )

        if not response or not getattr(response, "search_results", None):
            return jsonify({
                "answer": "No search results found for your query.",
                "sources": [],
            })

        # Filter by relevance and take top 5
        scored = []
        for chunk in response.search_results:
            score = get_relevance_score(chunk)
            if score is not None:
                try:
                    s = float(score)
                except (TypeError, ValueError):
                    s = 0.0
            else:
                s = 1.0  # treat missing score as acceptable
            if s >= MIN_RELEVANCE_SCORE:
                scored.append((s, chunk))
        scored.sort(key=lambda x: -x[0])
        picked = [chunk for _, chunk in scored[:TOP_K]]

        if not picked:
            accumulated = session.get("accumulated_chunks", [])
            if accumulated:
                context = build_context_from_results(accumulated)
                history = session.get("chat_history", [])
                history_for_llm = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in history]
                answer_text = call_llm(query, context, history=history_for_llm)
                answer = markdown.markdown(answer_text) if answer_text else ""
                sources = sources_from_chunks(accumulated)
                if "chat_history" not in session:
                    session["chat_history"] = []
                session["chat_history"].append({"role": "user", "content": query})
                session["chat_history"].append({"role": "assistant", "content": answer_text, "sources": sources})
                if len(session["chat_history"]) > MAX_CHAT_HISTORY_MESSAGES:
                    session["chat_history"] = session["chat_history"][-MAX_CHAT_HISTORY_MESSAGES:]
                session.modified = True
                return jsonify({"answer": answer, "sources": sources})
            return jsonify({
                "answer": "No results met the relevance threshold. Try rephrasing your question.",
                "sources": [],
            })

        context = build_context_from_results(picked)
        history = session.get("chat_history", [])
        history_for_llm = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in history]

        answer_text = call_llm(query, context, history=history_for_llm)
        answer = markdown.markdown(answer_text) if answer_text else ""

        sources = sources_from_chunks(picked)

        # Accumulate only the top 5 chunks that met the threshold (picked); never store below-threshold chunks.
        accumulated = session.get("accumulated_chunks", [])
        for ch in picked:
            accumulated.append(chunk_to_storable(ch))
        session["accumulated_chunks"] = accumulated[-MAX_ACCUMULATED_CHUNKS:]

        # Persist multi-turn history (raw text for assistant so we can resend to LLM)
        if "chat_history" not in session:
            session["chat_history"] = []
        session["chat_history"].append({"role": "user", "content": query})
        session["chat_history"].append({"role": "assistant", "content": answer_text, "sources": sources})
        if len(session["chat_history"]) > MAX_CHAT_HISTORY_MESSAGES:
            session["chat_history"] = session["chat_history"][-MAX_CHAT_HISTORY_MESSAGES:]
        session.modified = True

        return jsonify({"answer": answer, "sources": sources})

    except EgainError as e:
        if getattr(e, "status_code", None) == 401:
            return _clear_session_and_require_login()
        logging.exception("eGain API error")
        return jsonify({"error": str(e)}), getattr(e, "status_code", 500)
    except Exception as e:
        logging.exception("Chat failed")
        return jsonify({"error": str(e)}), 500


def _chat_stream_generator(query: str, context: str, picked: list):
    """Yield newline-delimited JSON: {t: chunk} for tokens, then {done: true, answer, sources}."""
    history = session.get("chat_history", [])
    history_for_llm = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in history]

    sources = sources_from_chunks(picked)

    accumulated = []
    try:
        for chunk in call_llm_stream(query, context, history=history_for_llm):
            if chunk:
                accumulated.append(chunk)
                yield (json.dumps({"t": chunk}) + "\n").encode("utf-8")
    except Exception as e:
        logging.exception("Streaming LLM failed")
        yield (json.dumps({"error": str(e)}) + "\n").encode("utf-8")
        return

    answer_text = "".join(accumulated).strip()
    answer_html = markdown.markdown(answer_text) if answer_text else ""

    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"role": "user", "content": query})
    session["chat_history"].append({"role": "assistant", "content": answer_text, "sources": sources})
    if len(session["chat_history"]) > MAX_CHAT_HISTORY_MESSAGES:
        session["chat_history"] = session["chat_history"][-MAX_CHAT_HISTORY_MESSAGES:]
    session.modified = True

    yield (json.dumps({"done": True, "answer": answer_html, "sources": sources}) + "\n").encode("utf-8")


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    if not session.get("token"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    debug = data.get("debug", False)
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        client = get_egain_client()
        if not client:
            return jsonify({"error": "Not authenticated"}), 401

        with client as egain:
            portal_id = os.getenv("EGAIN_PORTAL_ID", "PZ-9999")
            language = os.getenv("EGAIN_LANGUAGE", "en-US")
            response = egain.aiservices.retrieve.retrieve_chunks(
                q=query,
                portal_id=portal_id,
                language=language,
                channel={"name": CHANNEL_NAME},
            )

        debug_payload = _serialize_retrieve_response(response) if debug else None

        if not response or not getattr(response, "search_results", None):
            return jsonify({"answer": "No search results found.", "sources": []})

        scored = []
        for chunk in response.search_results:
            score = get_relevance_score(chunk)
            if score is not None:
                try:
                    s = float(score)
                except (TypeError, ValueError):
                    s = 0.0
            else:
                s = 1.0
            if s >= MIN_RELEVANCE_SCORE:
                scored.append((s, chunk))
        scored.sort(key=lambda x: -x[0])
        picked = [chunk for _, chunk in scored[:TOP_K]]

        def _prepend_debug(debug_payload, gen_func, *args, **kwargs):
            if debug_payload is not None:
                yield (json.dumps({"debug_retrieve": debug_payload}) + "\n").encode("utf-8")
            for chunk in gen_func(*args, **kwargs):
                yield chunk

        if not picked:
            accumulated = session.get("accumulated_chunks", [])
            if accumulated:
                context = build_context_from_results(accumulated)
                return Response(
                    stream_with_context(_prepend_debug(debug_payload, _chat_stream_generator, query, context, accumulated)),
                    content_type="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            # No picked and no accumulated: show fallback message.
            msg = "No results met the relevance threshold. Try rephrasing your question."
            payload = json.dumps({"done": True, "answer": f"<p>{msg}</p>", "sources": []}) + "\n"

            if "chat_history" not in session:
                session["chat_history"] = []
            session["chat_history"].append({"role": "user", "content": query})
            session["chat_history"].append({"role": "assistant", "content": msg, "sources": []})
            if len(session["chat_history"]) > MAX_CHAT_HISTORY_MESSAGES:
                session["chat_history"] = session["chat_history"][-MAX_CHAT_HISTORY_MESSAGES:]
            session.modified = True

            def _empty_stream():
                yield payload.encode("utf-8")

            return Response(
                stream_with_context(_prepend_debug(debug_payload, _empty_stream)),
                content_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        context = build_context_from_results(picked)

        # Accumulate only the top 5 chunks that met the threshold (picked); never store below-threshold chunks.
        accumulated = session.get("accumulated_chunks", [])
        for ch in picked:
            accumulated.append(chunk_to_storable(ch))
        session["accumulated_chunks"] = accumulated[-MAX_ACCUMULATED_CHUNKS:]

        return Response(
            stream_with_context(_prepend_debug(debug_payload, _chat_stream_generator, query, context, picked)),
            content_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except EgainError as e:
        if getattr(e, "status_code", None) == 401:
            return _clear_session_and_require_login()
        logging.exception("eGain API error")
        return jsonify({"error": str(e)}), getattr(e, "status_code", 500)
    except Exception as e:
        logging.exception("Chat stream failed")
        return jsonify({"error": str(e)}), 500


def _article_content_to_html(content):
    """Convert article content to HTML; API may return string, list, or object."""
    if content is None:
        return ""
    if isinstance(content, str):
        return markdown.markdown(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(markdown.markdown(item))
            elif hasattr(item, "content") or hasattr(item, "text"):
                parts.append(_article_content_to_html(getattr(item, "content", None) or getattr(item, "text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(content, "content"):
        return _article_content_to_html(getattr(content, "content"))
    return str(content)


def _log_article_api_response(article_id, portal_id, exception):
    """Log request context and any API response info from the exception."""
    logging.error(
        "Article API call: article_id=%r portal_id=%r",
        article_id,
        portal_id,
    )
    logging.error("Exception type: %s", type(exception).__name__)
    logging.error("Exception message: %s", str(exception))
    logging.error("Exception args: %s", getattr(exception, "args", ()))
    # Pydantic validation errors often expose the invalid body
    errs = None
    if hasattr(exception, "errors") and callable(exception.errors):
        try:
            errs = exception.errors()
        except Exception:
            pass
    if errs:
        for i, err in enumerate(errs):
            logging.error("Validation error [%s]: %s", i, err)
    # Some SDKs attach the raw response
    resp = getattr(exception, "response", None)
    if resp is not None:
        logging.error(
            "API response: status_code=%s headers=%s body=%s",
            getattr(resp, "status_code", None),
            dict(getattr(resp, "headers", {})) if hasattr(resp, "headers") else None,
            getattr(resp, "text", None) or getattr(resp, "content", None),
        )
    # Log full repr in case response is nested elsewhere
    logging.error("Exception __dict__: %s", getattr(exception, "__dict__", {}))


@app.route("/api/article/<article_id>")
def api_article(article_id):
    """Return article content for modal when user clicks a source.
    """
    if not session.get("token"):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        client = get_egain_client()
        if not client:
            return jsonify({"error": "Not authenticated"}), 401

        with client as egain:
            portal_id = os.getenv("EGAIN_PORTAL_ID", "PZ-9999")
            language = os.getenv("EGAIN_LANGUAGE", "en-US")
            article = egain.portal.article.get_article_by_id(
                article_id=article_id,
                portal_id=portal_id,
                accept_language="en-US",
                language=language,
                article_additional_attributes=["content"]
            )

        if article is None:
            return jsonify({"error": "Article not found"}), 404

        name = getattr(article, "name", None) or f"Article {article_id}"
        raw_content = getattr(article, "content", None)
        content = _article_content_to_html(raw_content)

        return jsonify({"id": article_id, "name": name, "content": content})

    except EgainError as e:
        if getattr(e, "status_code", None) == 401:
            return _clear_session_and_require_login()
        _log_article_api_response(article_id, os.getenv("EGAIN_PORTAL_ID", "PZ-9999"), e)
        logging.exception("eGain API error")
        return jsonify({"error": str(e)}), getattr(e, "status_code", 500)
    except Exception as e:
        _log_article_api_response(article_id, os.getenv("EGAIN_PORTAL_ID", "PZ-9999"), e)
        logging.exception("Article fetch failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)

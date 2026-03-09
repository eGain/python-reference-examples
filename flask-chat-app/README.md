# Knowledge Chat (Flask)

A Flask app that answers questions using your eGain knowledge base. Users sign in with **PKCE OAuth** (same as the search flask-app), type a question, and get an AI-generated answer plus clickable sources. Clicking a source fetches and shows the full article in a modal.

## Features

- **PKCE authentication** - OAuth 2.0 with PKCE flow
- **Chat UI** – Single input, conversation-style answers with sources
- **Retrieve Chunks API** – Uses `retrieve_chunks` API
- **Relevance filtering** – Only uses top 5 results above `MIN_RELEVANCE_SCORE`
- **LLM answer** – Builds context from `snippet` and `contextualSummary` of those results, plus the user query, and sends a prompt to a configurable LLM to generate the answer
- **Sources** – Each source shows the article ID and name; clicking opens the article content in a modal (fetched via `/api/article/<id>`)
- **Configurable LLMs** – OpenAI, Anthropic, or Gemini via env vars

## 🔧 Configuration

### Step 1: Create a Client Application in eGain

1. Create a new client application following the guide: [To Create a Client Application](https://apidev.egain.com/developer-portal/get-started/authentication_guide/#to-create-a-client-application)
2. **Required Scopes**: Assign the following scopes to your client application:
   - `knowledge.portalmgr.read`
   - `core.aiservices.read`
3. **Redirect URIs**: Configure the following URIs:
   - Redirect URI: `http://<hostname>:5001/callback`
   - Post Logout Redirect URI: `http://<hostname>:5001/`
   - Platform: **Web Application**
4. Save the **Client ID** for the next step
5. Follow https://apidev.egain.com/developer-portal/get-started/authentication_guide/#step-2-find-your-api-endpoints-metadata to capture your endpoints. 

### Step 2: Configure the Application

1. **Copy env and set values**

   ```bash
   cp .env.example .env
   ```

   Fill in (reuse from flask-app where applicable):

   - **Auth:** `EGAIN_CLIENT_ID`, `EGAIN_AUTH_URL`, `EGAIN_TOKEN_URL`, `EGAIN_PORTAL_ID`, `EGAIN_LANGUAGE`, `APP_SECRET_KEY`. 
   - **Chat:** `MIN_RELEVANCE_SCORE` (e.g. `0.3`) – only results with relevance ≥ this are used; top 5 of those are sent to the LLM
   - **LLM:** `LLM_PROVIDER` = `openai` | `anthropic` | `gemini`, and the corresponding API key:
     - OpenAI: `OPENAI_API_KEY` (optional: `OPENAI_MODEL`, default `gpt-4o-mini`)
     - Anthropic: `ANTHROPIC_API_KEY` (optional: `ANTHROPIC_MODEL`, default `claude-3-5-haiku-20241022`)
     - Gemini: `GOOGLE_API_KEY` (optional: `GEMINI_MODEL`, default `gemini-1.5-flash`)

2. **Install and run**

   ```bash
   pip install -r requirements.txt
   python app.py
   ```

   By default the app runs on **port 5001**.

3. **Use the app**

   - Open `http://<hostname>:5001`
   - Log in with eGain (PKCE)
   - Type a question and send; the app calls retrieve chunks, picks top 5 results above the relevance threshold, builds context from snippet + contextualSummary, and asks the LLM for an answer
   - Read the answer and use the **Sources** links; clicking one loads the article via `/api/article/<id>` and shows it in a modal

## Config reference

| Variable | Description |
|----------|-------------|
| `EGAIN_*` | (client id, auth/token/server URLs, portal, language) |
| `APP_SECRET_KEY` | Flask session secret; set this for production so sessions persist across restarts (if unset, each run gets a new key and sessions are invalidated) |
| `MIN_RELEVANCE_SCORE` | Min relevance (0.0–1.0) for a result to be used; then top 5 are taken (default `0.3`) |
| `LLM_PROVIDER` | `openai`, `anthropic`, or `gemini` |
| `OPENAI_API_KEY` | Required if `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | Required if `LLM_PROVIDER=anthropic` |
| `GOOGLE_API_KEY` | Required if `LLM_PROVIDER=gemini` |
| `OPENAI_MODEL` | Optional; default `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Optional; default `claude-3-5-haiku-20241022` |
| `GEMINI_MODEL` | Optional; default `gemini-1.5-flash` |

## Flow

1. User submits a question.
2. App calls **retrieve_chunks** (same as flask-app) with that query.
3. Results are filtered by relevance ≥ `MIN_RELEVANCE_SCORE`; up to **5** are kept.
4. For each kept result, **snippet** and **contextualSummary** are concatenated into a context string; the user query is added, and a system + user prompt is sent to the configured **LLM**.
5. The LLM’s answer is shown in the chat; **sources** are the article **id** and **name** from those 5 results.
6. When the user **clicks a source**, the frontend calls **GET /api/article/<article_id>**; the backend fetches the article (with `content`) and returns JSON; the UI shows it in a **modal**.

## Notes

- Relevance score is read from the chunk as `relevance_score`, `relevanceScore`, or `score` (whichever exists).
- Article content is requested with `article_additional_attributes=["content"]` and rendered as Markdown in the modal.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Redirect URI Mismatch | URI not registered | Verify redirect URI matches exactly in client app settings (include `/callback` path) |
| Invalid Client ID | Wrong client ID | Double-check `EGAIN_CLIENT_ID` in `.env` file |
| 401 Unauthorized | Missing/expired token | Check scopes are assigned to client app; try logging out and back in |
| Port 5001 in use | Another service on port 5000 | Change port in `app.py` or use `flask run --port=<other_port>` |
| Cannot resolve hostname | Hostname not configured | Add hostname to `/etc/hosts` or use DNS |
| Module not found | Dependencies not installed | Run `pip install -r requirements.txt` |

## Development Tips

- **Debug Mode**: Flask's debug mode with auto-reload is enabled by default in development
- **Environment Variables**: Use `.env` file for local development; never commit it to version control
- **Virtual Environment**: Consider using `venv` or `virtualenv` for isolated Python environments:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **Production Deployment**: Use a production WSGI server like Gunicorn or uWSGI instead of Flask's development server

## Additional Resources

- [eGain Developer Portal](https://apidev.egain.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OAuth 2.0 PKCE Flow](https://oauth.net/2/pkce/)
- [eGain API Python SDK](https://github.com/eGain/egain-api-python/)

## License

This is a reference example for demonstration purposes.
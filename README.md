# NexaHR RAG Chatbot — Gemini Edition

This version uses the existing approved HR policy CSV knowledge base and hybrid TF-IDF retrieval, with Google Gemini as the optional grounded response-writing layer.

## Setup (Windows)

1. Open a terminal in this folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.
4. Put your Gemini API key in `.env`:

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

5. Start the app:

```bash
streamlit run hr_rag_app.py
```

## How it works

The application first filters the approved policy chunks by the selected employee role and region, then performs hybrid lexical retrieval. When Gemini is enabled and `GEMINI_API_KEY` is available, only the retrieved authorized evidence is sent to Gemini for response writing. If Gemini is unavailable, the app falls back to the best approved local FAQ/policy answer.

## Security

Do not commit `.env` or paste an API key into `hr_rag_app.py`. `.gitignore` already excludes `.env`. If an API key has been exposed, revoke/rotate it before using the application.

## Model

The default model is `gemini-2.5-flash`, which Google documents as having a free tier. You can override `GEMINI_MODEL` if your account has access to another supported model.


### Gemini troubleshooting
The app calls the Gemini REST API directly and defaults to `gemini-3.5-flash`. It can fall back to `gemini-2.5-flash-lite` and `gemini-2.5-flash` if a model is unavailable. Set `GEMINI_API_KEY` in `.env`.

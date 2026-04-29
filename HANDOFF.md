# HANDOFF - FreeIAGateway

## État : Stack complète opérationnelle ✅

## Ce qui est fait
- `freeai_gateway.py` — QuotaManager SQLite + routing fallback + CLI REPL + `ask_with_messages()`
- `api.py` — FastAPI wrapper `/v1/chat/completions` + `/v1/models` (streaming SSE inclus)
- `tests/` — 29 tests, tous verts
- **AnythingLLM** connecté sur `http://localhost:8000/v1`, modèle `freeai-gateway`
- **MemPalace** installé (v3.3.3), palace minée, MCP configuré dans AnythingLLM + Claude Code

## Architecture
```
FreeIAGateway (freeai_gateway.py)
  ├── QuotaManager (SQLite)
  └── ask_with_messages()  ← utilisé par l'API (stateless)

FastAPI (api.py) — port 8000
  ├── GET  /v1/models
  └── POST /v1/chat/completions (stream=true/false)

Providers (ordre fallback) :
  Gemini → Groq → Mistral → HuggingFace

MemPalace (~/.mempalace/palace)
  ├── MCP server : mempalace-mcp.exe
  ├── AnythingLLM : anythingllm_mcp_servers.json ✅
  └── Claude Code : ~/.claude/.mcp.json ✅
```

## Pour lancer l'API
```bash
cd "/c/Users/Mini pc/Desktop/agregateur"
python -m uvicorn api:app --port 8000
```

## Prochaines étapes possibles
- Miner les sessions Claude Code dans MemPalace : `mempalace mine ~/.claude/projects/ --mode convos`
- Tester le RAG dans AnythingLLM (uploader des documents)
- Ajouter d'autres providers gratuits (Cohere, Together AI free tier...)

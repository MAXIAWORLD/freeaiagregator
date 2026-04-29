import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from freeai_gateway import FreeIAGateway

app = FastAPI(title="FreeIA Gateway")
gateway = FreeIAGateway(session_id="api")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "freeai-gateway", "object": "model", "created": 0, "owned_by": "freeai"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        answer = gateway.ask_with_messages(messages)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "freeai-gateway",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

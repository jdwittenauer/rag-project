from dotenv import load_dotenv

load_dotenv()

import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.llms.base import ChatMessage
from llama_index.llms.types import MessageRole

from src.message import ChatData
from src.engine import get_chat_engine


chat_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_engine
    chat_engine = get_chat_engine()
    yield


app = FastAPI(lifespan=lifespan)

environment = os.getenv("ENVIRONMENT")

if environment == "dev":
    logger = logging.getLogger("uvicorn")
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAG project app!"
    }


@app.post("/api/chat")
async def chat(request: Request, data: ChatData):
    global chat_engine

    # check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    lastMessage = data.messages.pop()
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )
    # convert messages coming from the request to type ChatMessage
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]

    # query chat engine
    response = chat_engine.stream_chat(lastMessage.content, messages)

    # stream response
    async def event_generator():
        for token in response.response_gen:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            yield token

    return StreamingResponse(event_generator(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)

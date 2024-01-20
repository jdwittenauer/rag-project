import os
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.message import ChatData, get_last_message, get_chat_history
from src.engine import get_chat_engine


chat_engine = None

load_dotenv()
environment = os.getenv("ENVIRONMENT")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_engine
    chat_engine = get_chat_engine()
    yield


app = FastAPI(lifespan=lifespan)

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
    last_message = get_last_message(data)

    # convert messages coming from the request to type ChatMessage
    chat_history = get_chat_history(data)

    # query chat engine
    response = chat_engine.stream_chat(last_message.content, chat_history)

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

from typing import List
from llama_index.llms.types import MessageRole
from llama_index.llms.base import ChatMessage
from pydantic import BaseModel
from fastapi import HTTPException, status


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatData(BaseModel):
    messages: List[Message]


def get_last_message(data):
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    last_message = data.messages.pop()
    if last_message.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )
    return last_message


def get_chat_history(data):
    chat_history = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]
    return chat_history
from typing import List
from llama_index.llms.types import MessageRole
from pydantic import BaseModel


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatData(BaseModel):
    messages: List[Message]

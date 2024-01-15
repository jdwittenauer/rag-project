from llama_index.chat_engine import SimpleChatEngine


def get_chat_engine():
    return SimpleChatEngine.from_defaults()

from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.config.llm import (
    DASHSCOPE_API_KEY,
    QWEN_BASE_URL,
    CHAT_MODEL,
    TEMPERATURE,
    validate_llm_config,
)


def get_llm(
    model: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """
    Khởi tạo chat model.

    Tương thích cả 2 kiểu gọi:
    - get_llm(model="qwen-plus")
    - get_llm(model_name="qwen-plus")
    """
    validate_llm_config()

    selected_model = model_name or model or CHAT_MODEL
    selected_temperature = TEMPERATURE if temperature is None else temperature

    return ChatOpenAI(
        model=selected_model,
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=selected_temperature,
    )
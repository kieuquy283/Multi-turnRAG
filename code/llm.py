from langchain_openai import ChatOpenAI

from code.config import OPENAI_API_KEY, CHAT_MODEL


def get_llm(model_name: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """
    Khởi tạo chat model.

    Args:
        model_name: tên model, nếu None thì dùng CHAT_MODEL
        temperature: nhiệt độ sinh text

    Returns:
        ChatOpenAI
    """
    return ChatOpenAI(
        model=model_name or CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature
    )


def generate_answer(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời cuối.

    Args:
        prompt: prompt hoàn chỉnh

    Returns:
        str: nội dung câu trả lời
    """
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip()
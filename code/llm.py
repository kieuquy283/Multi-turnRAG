from code.formatter import format_context
from openai import OpenAI
import os
from dotenv import load_dotenv


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def ask_llm(question, docs):
    context = format_context(docs)

    prompt = f"""
Bạn là chuyên gia pháp luật Việt Nam.

YÊU CẦU:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không tự bịa hoặc suy diễn ngoài tài liệu.
- Nếu không đủ dữ liệu, phải nói rõ: "Không đủ dữ liệu trong tài liệu được truy xuất."
- Khi trả lời, ưu tiên trích rõ Điều nào trong tài liệu.
- Trả lời ngắn gọn, rõ ràng, đúng trọng tâm.

CONTEXT:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý pháp luật."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
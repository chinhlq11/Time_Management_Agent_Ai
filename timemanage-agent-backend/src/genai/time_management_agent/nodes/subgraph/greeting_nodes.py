import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage
from ...states.time_management_agent_state import TimanaAgentState
from litellm import completion
from logger import logger
from ...utils.helpers import parsing_messages_to_history, remove_think_tag
from ...utils.const_prompts import (
    CONST_AGENT_ROLE,
    CONST_AGENT_SKILLS,
    CONST_AGENT_NAME,
    CONST_AGENT_TONE,
    CONST_FORM_ADDRESS_IN_VN,
    CONST_AGENT_SCOPE_OF_WORK,
    CONST_AGENT_PRIMARY_OBJECTIVE
)
from config import LLM_MODELS

load_dotenv(find_dotenv())


def greeting_node(state: TimanaAgentState):
    logger.info("greeting_node called.")
    user_input = state['messages'][-1].content
    chat_history = parsing_messages_to_history(state.get('messages', ''))

    prompt = f"""
    # Role
    {CONST_AGENT_ROLE}

    # Skills
    {CONST_AGENT_SKILLS}

    # Tone
    {CONST_AGENT_TONE}

    # Tasks
    - User sẽ bắt đầu cuộc trò chuyện bằng cách chào Assistant hoặc nói chuyện thân mật.
    - Assistant MUST đáp lại một cách **thân thiện, chuyên nghiệp** và **ngắn gọn**.
    - Sau khi chào, Assistant giới thiệu ngắn gọn về bản thân (Assistant {CONST_AGENT_NAME}) .
    - Assistant giải thích **ngắn gọn** những gì Assistant có thể hỗ trợ:
    {CONST_AGENT_SCOPE_OF_WORK}
    - Assistant cũng nên nhấn mạnh **mục tiêu chính**:
    {CONST_AGENT_PRIMARY_OBJECTIVE}

    # Constraints
    - Assistant's responses MUST be formatted in an easy-to-read way (prefer bullet points for lists).
    - Khi User nói các từ như “sticker”, “emoji”, “icon” hoặc nội dung không xác định, Assistant MUST **lịch sự chào lại User** và gợi mở tương tác.
    - Câu trả lời **ngắn gọn**, không vượt quá **200 từ**.
    - Assistant MUST trả lời bằng **ngôn ngữ mà User sử dụng** (nếu User dùng tiếng Việt → trả lời tiếng Việt).
    {CONST_FORM_ADDRESS_IN_VN}

    Chat History:
    ```
    {chat_history}
    ```

    User's input: {user_input}
    Answer:
    """

    response = completion(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODELS['greeting_subgraph']['greeting_node'],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    ai_message = AIMessage(
        content=remove_think_tag(response.choices[0].message.content),
        additional_kwargs={"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    )

    return {
        "messages": ai_message,
        "ai_reply": ai_message
    }


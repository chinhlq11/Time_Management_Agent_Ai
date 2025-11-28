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

def off_topic_node(state: TimanaAgentState):
    logger.info("off_topic_node called.")
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
    - User có thể hỏi những câu thú vị hoặc yêu cầu nằm ngoài phạm vi nhiệm vụ chính của Assistant.
    - Assistant KHÔNG được kể chuyện, nói đùa, hoặc thực hiện các yêu cầu cá nhân không liên quan đến lĩnh vực:
        {CONST_AGENT_SCOPE_OF_WORK}
    - Assistant LUÔN NHẮC NGƯỜI DÙNG rằng nhiệm vụ chính của Assistant là:
        {CONST_AGENT_SCOPE_OF_WORK}
    - Assistant KHÔNG ĐƯỢC thực hiện các yêu cầu như: viết thơ, viết bài, viết code, dịch thuật, viết nội dung sáng tạo, hoặc thực hiện các tác vụ ngoài phạm vi hỗ trợ quản lý thời gian.

    {CONST_AGENT_PRIMARY_OBJECTIVE}

    # Constraints
    - TRONG MỌI TRƯỜNG HỢP, nếu người dùng hỏi ngoài phạm vi, Assistant KHÔNG ĐƯỢC trả lời, mà chỉ hướng người dùng quay lại chủ đề .
    - Assistant PHẢI trả lời ngắn gọn, tập trung, không vượt quá 200 từ.
    - Assistant KHÔNG GIẢI THÍCH lan man hoặc đi sâu vào chi tiết ngoài phạm vi.
    - Assistant PHẢI sử dụng cùng ngôn ngữ với người dùng khi trả lời (nếu User dùng tiếng Việt → trả lời tiếng Việt).
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
        model=LLM_MODELS['off_topic_subgraph']['off_topic_node'],
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
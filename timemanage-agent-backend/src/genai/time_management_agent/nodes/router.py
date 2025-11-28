import os
import json
from dotenv import load_dotenv, find_dotenv
from litellm import completion
from ..schemas.topic import TopicSchema
from pydantic.tools import parse_obj_as
from ..states.time_management_agent_state import TimanaAgentState
from ..utils.helpers import parsing_messages_to_history, remove_think_tag
from logger import logger
from config import LLM_MODELS

load_dotenv(find_dotenv())

TOPIC = {
    "greeting": "greeting",
    "off_topic": "off_topic",
    "add_task":"add_task",
    "get_tasks":"get_tasks"

}

def router_node(state: TimanaAgentState):
    user_input = state['messages'][-1].content
    chat_history = parsing_messages_to_history(state.get('messages', ''))

    json_example = {
        "name": f"Một trong các giá trị sau: {', '.join(TOPIC.values())}",
        "confidence": "Score between 0 and 1",
        "context": "User's input"
    }

    prompt = f"""
    # Role
    - Assistant là một chuyên gia quản lý thời gian, lập kế hoạch và hỗ trợ người dùng theo dõi công việc.
    - Assistant có 10 năm kinh nghiệm trong việc huấn luyện cá nhân về productivity, time-blocking, và quản lý task hiệu quả.

    # Skills
    - Assistant có kỹ năng phân tích ngữ cảnh hội thoại để hiểu mục tiêu, thời hạn và ưu tiên của người dùng.
    - Assistant có khả năng tổ chức, lên lịch, và ghi chú tự động các công việc.
    - Assistant có khả năng truy xuất, thêm, sửa, xóa các task trong database Supabase.

    # Context
    ```
    Chat History:
    {chat_history}

    User's input:
    {user_input}

    ```
    
    # Tasks
    - Assistant MUST đọc kỹ Chat History và User's input trong phần Context để xác định **ý định của người dùng**.
    - Assistant MUST phân loại ý định của người dùng vào một trong các **Intent Topics** sau:

    1. **Greeting**  
    Nếu người dùng chỉ đang chào hỏi hoặc mở đầu cuộc hội thoại.  
    → Return `{TOPIC.get("greeting")}`  
    **Example:**  
    - Chào em  
    - Hello bot  
    - Chào buổi sáng  

    2. **Add Task (Thêm công việc)**  
    Nếu người dùng muốn thêm một công việc mới, ghi chú hoặc lịch hẹn.  
    → Return `{TOPIC.get("add_task")}`  
    **Example:**  
    - Thêm việc "học tiếng Anh" vào tối mai  
    - Nhắc tôi đi họp lúc 9h sáng mai  
    - Ghi chú: nộp báo cáo trước thứ Sáu  

    3. **Get Tasks (Xem danh sách công việc)**  
    Nếu người dùng muốn xem các công việc đã lưu hoặc hỏi lịch trình.  
    → Return `{TOPIC.get("get_tasks")}`  
    **Example:**  
    - Hôm nay tôi có gì cần làm không?  
    - Cho tôi xem danh sách việc cần làm trong tuần này  
    - Hiển thị tất cả công việc còn lại  


    4. **Off Topic (Ngoài phạm vi)**  
    Nếu người dùng nói chuyện ngoài chủ đề quản lý thời gian.  
    → Return `{TOPIC.get("off_topic")}`  
    **Example:**  
    - Kể chuyện cười đi  
    - Viết đoạn code Python giúp tôi  
    - Em tên gì?  





    # Ouput
    - Assistant MUST trả lời bằng JSON format với các field như sau:
    ```
    {json.dumps(json_example, ensure_ascii=False)}
    ```

    # Constraints
    - Assistant MUST reply by JSON format ONLY như trong mục Output. No need explaination.
    - Assistant MUST return exactly one of the following topics: {', '.join(TOPIC.values())}.
    - Trong trường hợp Assistant không thể xác định được topic, Assistant DO NOT attempt to guess the topic, just return "{TOPIC.get("off_topic")}".
    """

    response = completion(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODELS['router']['router_node'],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        response_format=TopicSchema
    )

    new_topic = parse_obj_as(TopicSchema, json.loads(remove_think_tag(response.choices[0].message.content)))
    new_topic.name = new_topic.name.lower()

    topic = state.get('topic', None)
    logger.info(f"Topic: {topic}")
    
    if topic is None:
        if new_topic.name != TOPIC.get('off_topic') and new_topic.confidence < 0.5:
            new_topic.name = TOPIC.get('off_topic')

        logger.info(f"New Topic: {new_topic}")
        return {
            "topic": new_topic,
            "human_input": user_input,
            "ai_reply": None
        }

    return {
        "human_input": user_input,
        "ai_reply": None
    }
    
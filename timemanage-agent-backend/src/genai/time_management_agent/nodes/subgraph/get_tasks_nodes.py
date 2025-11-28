import os
from datetime import datetime, date
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage
from litellm import completion
from logger import logger
from supabase import create_client
from ...states.time_management_agent_state import TimanaAgentState
from ...utils.helpers import parsing_messages_to_history, remove_think_tag
from ...utils.const_prompts import (
    CONST_AGENT_NAME,
    CONST_AGENT_TONE,
    CONST_FORM_ADDRESS_IN_VN
)
from config import LLM_MODELS

load_dotenv(find_dotenv())

# --- SUPABASE INIT ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_tasks_node(state: TimanaAgentState):
    """
    Node này dùng để:
      - Lấy danh sách công việc từ Supabase.
      - Hiểu ngữ cảnh người dùng (VD: 'hôm nay', 'ngày mai', 'tuần này', 'tất cả').
      - Trả về danh sách công việc bằng giọng thân thiện, dễ hiểu.
    """
    logger.info("🧩 get_tasks_node called.")
    user_input = state["messages"][-1].content
    chat_history = parsing_messages_to_history(state.get("messages", ""))

    # --- Step 1: Phân tích người dùng hỏi ngày nào ---
    # Dùng LLM để xác định phạm vi thời gian (today, tomorrow, week, all)
    detect_time_prompt = f"""
    Bạn là trợ lý thông minh tên {CONST_AGENT_NAME}.
    Hãy đọc câu của người dùng và xác định họ muốn xem công việc trong khoảng thời gian nào.

    Output JSON dạng:
    {{
      "scope": "today" | "tomorrow" | "week" | "all"
    }}

    Ví dụ:
    - "Công việc hôm nay" => "today"
    - "Ngày mai phải làm gì" => "tomorrow"
    - "Tuần này có gì không?" => "week"
    - "Liệt kê tất cả công việc" => "all"

    Câu người dùng: "{user_input}"
    """

    response = completion(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODELS["task_subgraph"]["add_task_node"],
        messages=[{"role": "user", "content": detect_time_prompt}],
        temperature=0.2,
    )

    import json
    try:
        parsed = json.loads(remove_think_tag(response.choices[0].message.content))
        scope = parsed.get("scope", "today")
    except Exception as e:
        logger.error(f"⚠️ JSON parsing error in get_tasks_node: {e}")
        scope = "today"

    logger.info(f"🔎 Detected time scope: {scope}")

    # --- Step 2: Lọc task theo phạm vi ---
    today = date.today()
    if scope == "today":
        query = supabase.table("tasks").select("*").gte("due_date", str(today)).lt("due_date", str(today.replace(day=today.day + 1)))
    elif scope == "tomorrow":
        query = supabase.table("tasks").select("*").gte("due_date", str(today.replace(day=today.day + 1))).lt("due_date", str(today.replace(day=today.day + 2)))
    elif scope == "week":
        query = supabase.table("tasks").select("*").gte("due_date", str(today)).lt("due_date", str(today.replace(day=today.day + 7)))
    else:
        query = supabase.table("tasks").select("*")

    try:
        result = query.order("due_date", desc=False).execute()
        tasks = result.data or []
    except Exception as e:
        logger.error(f"❌ Supabase fetch error: {e}")
        tasks = []

    # --- Step 3: Chuẩn bị câu trả lời ---
    if not tasks:
        ai_reply_text = f"""
        Dạ, hiện tại em chưa thấy có công việc nào trong danh sách {scope if scope != 'all' else 'tất cả'} ạ.
        Anh/Chị có muốn em giúp ghi thêm công việc mới không?
        """
    else:
        task_lines = "\n".join(
            [
                f"- {t['title']} (hạn: {t.get('due_date', 'unknown')})"
                for t in tasks
            ]
        )
        ai_reply_text = f"""
            Dạ, đây là các công việc {scope if scope != 'all' else ''} của Anh/Chị nè:
            {task_lines}

            Anh/Chị có muốn em đặt nhắc nhở cho các công việc này không ạ?
            """

    ai_message = AIMessage(
        content=ai_reply_text.strip(),
        additional_kwargs={
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
    )

    return {
        "messages": [ai_message],
        "ai_reply": ai_message,
        "task_scope": scope,
        "tasks": tasks,
    }

import os
import json
import uuid
from datetime import datetime,date
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage
from litellm import completion
from logger import logger
from ...states.time_management_agent_state import TimanaAgentState
from ...utils.helpers import parsing_messages_to_history, remove_think_tag
from ...utils.const_prompts import CONST_AGENT_NAME
from config import LLM_MODELS

from supabase import create_client

from pydantic import BaseModel, validator, ValidationError
import dateparser
from dateutil import parser as du_parser
from zoneinfo import ZoneInfo

load_dotenv(find_dotenv())

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


# ------------------ MODEL ------------------

class TaskModel(BaseModel):
    title: str
    start_at: str = "unknown"
    due_at: str = "unknown"
    description: str = ""

    @validator("title")
    def title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()

    @validator("start_at", "due_at", pre=True, always=True)
    def ensure_string_and_normalize_datetime(cls, v):
        # Nếu không có giá trị, luôn dùng "unknown"
        if not v:
            return "unknown"
        if isinstance(v, str) and v.lower() == "unknown":
            return "unknown"

        # Nếu có giá trị, cố gắng parse sang datetime có timezone
        try:
            dt = du_parser.isoparse(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=LOCAL_TZ)
            return dt.astimezone(LOCAL_TZ)
        except Exception:
            dt = dateparser.parse(
                v,
                settings={
                    "TIMEZONE": "Asia/Ho_Chi_Minh",
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "PREFER_DATES_FROM": "future",
                    "RELATIVE_BASE": datetime.now(LOCAL_TZ),
                },
            )
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=LOCAL_TZ)
                return dt
        return "unknown"


# ------------------ MAIN NODE ------------------

def add_task_node(state: TimanaAgentState):
    """
    Node này sẽ:
    1. Dùng LLM để trích xuất các task từ câu người dùng.
    2. Chuẩn hóa dữ liệu (start_at, due_at, task_date, description).
    3. Lưu vào bảng `tasks`.
    """
    logger.info("add_task_node called.")

    user_input = state["messages"][-1].content
    chat_history = parsing_messages_to_history(state.get("messages", ""))

    # --- Prompt LLM ---
    extract_prompt = f"""
    You are a Vietnamese AI time assistant named {CONST_AGENT_NAME}.
    Your task is to extract all time-related tasks from the user's message and return ONLY valid JSON — no explanations, no text.

    Today is {date.today().isoformat()} (timezone: +07:00).

    Each extracted task must follow this schema:
    [
    {{
        "title": "string (required)",
        "start_at": "ISO-8601 datetime with timezone (e.g. 2025-11-04T09:00:00+07:00) or 'unknown'",
        "due_at": "ISO-8601 datetime with timezone (e.g. 2025-11-04T17:00:00+07:00) or 'unknown'",
        "description": "string (optional)"
    }}
    ]

    Guidelines:
    - If multiple tasks are mentioned, return all of them in an array.
    - If the user gives a **date but no time**, use `T00:00:00+07:00`.
    - If the user gives **relative time expressions** (e.g. "hôm nay", "chiều mai", "thứ 2 tuần sau"):
    - Convert them into the correct ISO-8601 date based on "today".
    - For "sáng" use ~09:00, for "trưa" use ~12:00, for "chiều" use ~15:00, for "tối" use ~19:00, for "đêm" use ~22:00.
    - If no specific time is mentioned at all, use `"unknown"`.
    - If only one time is given (e.g. "họp lúc 9h sáng"), treat it as `"start_at"` and set `"due_at"` to `"unknown"`.
    - If a time range is given (e.g. "từ 9h đến 11h"), use them as `start_at` and `due_at`.
    - Include short description if the user provides details like purpose, place, or participants.

    Return only valid JSON output.

    User message:""" + json.dumps(user_input, ensure_ascii=False)

    response = completion(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODELS["task_subgraph"]["add_task_node"],
        messages=[{"role": "user", "content": extract_prompt}],
        temperature=0.0,
    )

    raw_content = response.choices[0].message.content
    logger.info(f"LLM raw response: {raw_content}")

    # --- Parse JSON ---
    try:
        cleaned = remove_think_tag(raw_content).strip()
        if not cleaned.startswith("["):
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start != -1 and end != -1:
                cleaned = cleaned[start:end]
        tasks_raw = json.loads(cleaned)
        if not isinstance(tasks_raw, list):
            tasks_raw = [tasks_raw]
    except Exception as e:
        logger.error(f"❌ JSON parsing error: {e}")
        tasks_raw = [{"title": user_input, "start_at": "unknown", "due_at": "unknown", "description": ""}]

    # --- Validate & normalize ---
    valid_tasks = []
    for t in tasks_raw:
        try:
            # Nếu không có trường nào, thêm mặc định
            if "start_at" not in t or t["start_at"] is None:
                t["start_at"] = "unknown"
            if "due_at" not in t or t["due_at"] is None:
                t["due_at"] = "unknown"

            task = TaskModel(**t)
            valid_tasks.append(task)
        except ValidationError as ve:
            logger.warning(f"Validation failed for {t}: {ve}")
            valid_tasks.append(TaskModel(title=t.get("title", "Untitled")))

    # --- Insert vào Supabase ---
    inserted_tasks = []
    for task in valid_tasks:
        start_at = task.start_at if isinstance(task.start_at, datetime) else None
        due_at = task.due_at if isinstance(task.due_at, datetime) else None
        task_date = None

        if start_at:
            task_date = start_at.date().isoformat()
        elif due_at:
            task_date = due_at.date().isoformat()

        data = {
            "id": str(uuid.uuid4()),
            "title": task.title,
            "description": task.description or "",
            "start_at": start_at.isoformat() if start_at else None,
            "due_at": due_at.isoformat() if due_at else None,
            "task_date": task_date,
            "created_at": datetime.now(LOCAL_TZ).isoformat(),
            "completed": False,
        }

        try:
            supabase.table("tasks").insert(data).execute()
            logger.info(f"✅ Saved task: {data['title']}")
        except Exception as e:
            logger.error(f"⚠️ Supabase insert error: {e}")

        inserted_tasks.append(data)

    # --- Tạo phản hồi ---
    def format_time(t):
        return t if t else "chưa xác định"

    task_list = "\n".join([
        f"- {t['title']} (bắt đầu: {format_time(t['start_at'])}, hạn: {format_time(t['due_at'])})"
        for t in inserted_tasks
    ])

    ai_reply_text = f"""
Dạ, {CONST_AGENT_NAME} đã ghi lại công việc của Anh/Chị như sau:
{task_list}
    """.strip()

    ai_message = AIMessage(
        content=ai_reply_text,
        additional_kwargs={"current_time": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")},
    )

    return {
        "messages": [ai_message],
        "ai_reply": ai_message,
        "parsed_tasks": inserted_tasks,
    }

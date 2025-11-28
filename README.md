# AI Time Management Assistant 

## 📋 Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
3. [Cách Hoạt Động](#cách-hoạt-động)
4. [Cài Đặt & Sử Dụng](#cài-đặt--sử-dụng)
5. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
6. [Hướng Dẫn Chi Tiết từng Thành Phần](#hướng-dẫn-chi-tiết-từng-thành-phần)
7. [Ví Dụ Sử Dụng](#ví-dụ-sử-dụng)
8. [Khắc Phục Sự Cố](#khắc-phục-sự-cố)

---

## 🎯 Giới Thiệu

**AI Time Management Assistant** (hay **AI Calendar**) là một trợ lý thông minh được xây dựng với **LangGraph** và **LLM (Groq API)** để giúp người dùng:

✅ **Quản lý công việc (Task Management)**
- Thêm, xem, cập nhật, xóa công việc
- Tự động trích xuất thông tin từ câu văn tự nhiên (NLP)

✅ **Sắp xếp thời gian (Time Blocking)**
- Phân loại công việc theo ngày, tuần
- Quản lý deadline và reminder

✅ **Tối ưu hóa năng suất (Productivity)**
- Gợi ý lịch trình dựa trên mức độ ưu tiên
- Học từ thói quen người dùng

✅ **Giao tiếp thân thiện (Conversational AI)**
- Hỗ trợ tiếng Việt với các từ xưng hô lịch sự
- Phản hồi nhanh và chính xác

---

## 🏗️ Kiến Trúc Hệ Thống

### Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────┐
│                   User Input                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   ROUTER NODE          │
        │  (Phân loại Intent)    │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ FLOW CONTROLLER NODE   │
        │ (Chọn Subgraph)        │
        └────────────┬───────────┘
                     │
        ┌────────────┴────────────┬─────────────┬───────────────┐
        │                         │             │               │
        ▼                         ▼             ▼               ▼
   ┌─────────┐          ┌─────────────┐  ┌──────────┐  ┌──────────────┐
   │GREETING │          │  ADD_TASK   │  │GET_TASKS │  │  OFF_TOPIC   │
   │SUBGRAPH │          │ SUBGRAPH    │  │SUBGRAPH  │  │  SUBGRAPH    │
   └────┬────┘          └──────┬──────┘  └────┬─────┘  └──────┬───────┘
        │                      │               │               │
        ▼                      ▼               ▼               ▼
   ┌─────────┐          ┌─────────────┐  ┌──────────┐  ┌──────────────┐
   │ Supabase│          │ Supabase DB │  │ Supabase │  │   LLM API    │
   │  (Sync) │          │  (Insert)   │  │(Fetch)   │  │   (Reply)    │
   └─────────┘          └─────────────┘  └──────────┘  └──────────────┘
```

### Stack Công Nghệ

| Thành Phần | Công Nghệ | Mục Đích |
|-----------|-----------|---------|
| **LLM** | Groq API (Llama 70B, Gemma2 9B) | Xử lý ngôn ngữ tự nhiên |
| **Orchestration** | LangGraph | Quản lý luồng xử lý |
| **Database** | Supabase (PostgreSQL) | Lưu trữ công việc |
| **Framework** | LangChain | Intergration LLM |
| **Runtime** | Python 3.x | Chạy agent |

---

## 🔄 Cách Hoạt Động

### 1️⃣ Quy Trình Xử Lý Yêu Cầu (Request Flow)

```
┌─ Input từ User
│
├─ Router Node: Phân loại ý định (Intent)
│  ├─ Greeting (Chào hỏi)
│  ├─ Add Task (Thêm công việc)
│  ├─ Get Tasks (Xem công việc)
│  └─ Off Topic (Ngoài chủ đề)
│
├─ Flow Controller: Chọn subgraph tương ứng
│
├─ Execute Subgraph:
│  │
│  ├─ Nếu GREETING → Giới thiệu bản thân
│  │
│  ├─ Nếu ADD_TASK:
│  │  ├─ Trích xuất task từ câu tự nhiên
│  │  ├─ Chuẩn hóa dữ liệu (ngày giờ)
│  │  └─ Lưu vào Supabase
│  │
│  ├─ Nếu GET_TASKS:
│  │  ├─ Phân tích thời gian (hôm nay/tuần/tất cả)
│  │  ├─ Query từ Supabase
│  │  └─ Format trả về
│  │
│  └─ Nếu OFF_TOPIC → Trả lời ngắn gọn
│
├─ Reset Topic → Chuẩn bị cho câu hỏi tiếp theo
│
└─ Output → Trả về cho User
```

### 2️⃣ State Management

```python
# TimanaAgentState
{
    "messages": [...],           # Lịch sử tin nhắn
    "human_input": "...",        # Đầu vào từ user
    "topic": TopicSchema,        # {"name": "add_task", "confidence": 0.95, "context": "..."}
    "selected_flow": "...",      # "greeting" | "add_task" | "get_tasks" | "off_topic"
    "ai_reply": AIMessage        # Phản hồi từ AI
}
```

### 3️⃣ Router Node - Phân Loại Intent

**File**: `nodes/router.py`

Dùng **Groq LLM** để phân loại ý định người dùng:

```python
def router_node(state: TimanaAgentState):
    # Input: Tin nhắn từ user + lịch sử
    # Process: Gửi prompt tới LLM để phân loại
    # Output: TopicSchema với intent + confidence score
    
    return {
        "topic": TopicSchema(
            name="add_task",          # greeting | off_topic | add_task | get_tasks
            confidence=0.95,           # 0.0 - 1.0
            context="Thêm việc học..."
        )
    }
```

**Logic Phân Loại**:

| Intent | Ví Dụ | Confidence Threshold |
|--------|-------|---------------------|
| **Greeting** | "Chào em", "Hello bot" | < 0.5 → off_topic |
| **Add Task** | "Thêm việc học tiếng Anh", "Nhắc tôi họp 9h sáng" | ≥ 0.5 |
| **Get Tasks** | "Hôm nay tôi có gì", "Liệt kê công việc tuần này" | ≥ 0.5 |
| **Off Topic** | "Kể chuyện cười", "Viết code Python" | < 0.5 |

---

## 🛠️ Cài Đặt & Sử Dụng

### Yêu Cầu

- Python 3.8+
- Git
- Groq API Key
- Supabase Account (URL + Key)

### Bước 1: Cài Đặt Môi Trường

```bash
# 1. Clone repository
git clone <repository-url>
cd Time_Management_Agent_Ai

# 2. Tạo virtual environment
python -m venv tm_venv
.\tm_venv\Scripts\activate  # Windows
# hoặc
source tm_venv/bin/activate  # macOS/Linux

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Cấu Hình Biến Môi Trường

Tạo file `.env` trong thư mục `timemanage-agent-backend/src/`:

```env
# Groq Configuration
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
GROQ_LLM_MODEL_GEMMA2_9B=gemma2-9b-it
GROQ_LLM_MODEL_LLAMA_70B=llama-70b-versatile

# Supabase Configuration
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Application Configuration
APP_ROOT_PATH=/path/to/app
```

### Bước 3: Thiết Lập Supabase Database

Tạo bảng `tasks` trong Supabase:

```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    start_at TIMESTAMP WITH TIME ZONE,
    due_at TIMESTAMP WITH TIME ZONE,
    task_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_task_date ON tasks(task_date);
CREATE INDEX idx_due_at ON tasks(due_at);
```

### Bước 4: Chạy Agent

```bash
# Chạy script chính
python main.py

# Hoặc chạy demo Supabase
python supabase_demo.py
```

---

## 📁 Cấu Trúc Dự Án

```
Time_Management_Agent_Ai/
│
├── README.md                           # File README gốc
├── README_DETAILED.md                  # File này - Hướng dẫn chi tiết
│
├── timemanage-agent-backend/
│   ├── logs/                          # Thư mục chứa log files
│   │
│   └── src/
│       ├── config.py                  # Cấu hình LLM models
│       ├── console.py                 # Giao diện console
│       ├── logger.py                  # Logger configuration
│       │
│       └── genai/
│           └── time_management_agent/
│               │
│               ├── __init__.py
│               │
│               ├── agent.py           # Main agent graph builder
│               │
│               ├── states/
│               │   └── time_management_agent_state.py  # State definition
│               │
│               ├── nodes/             # Core nodes
│               │   ├── router.py              # Intent classification
│               │   ├── flow_controller.py    # Flow routing
│               │   ├── do_nothing.py         # Placeholder node
│               │   ├── reset_topic.py        # Topic reset
│               │   │
│               │   └── subgraph/
│               │       ├── greeting_nodes.py        # Greeting logic
│               │       ├── add_task_nodes.py        # Task insertion
│               │       ├── get_tasks_nodes.py       # Task retrieval
│               │       └── off_topic_nodes.py       # Off-topic handling
│               │
│               ├── subgraphs/         # Subgraph implementations
│               │   ├── greeting_subgraph.py
│               │   ├── add_task_subgraph.py
│               │   ├── get_tasks_subgraph.py
│               │   └── off_topic_subgraph.py
│               │
│               ├── schemas/           # Pydantic schemas
│               │   └── topic.py       # Topic/Intent schema
│               │
│               └── utils/
│                   ├── const_prompts.py  # Prompt templates
│                   └── helpers.py        # Utility functions
```

---

## 🔍 Hướng Dẫn Chi Tiết từng Thành Phần

### 1. Router Node (`nodes/router.py`)

**Chức Năng**: Phân loại ý định (Intent) của user

**Input**: 
```python
state = {
    "messages": [...],      # Lịch sử tin nhắn
    "topic": None           # Lần đầu gọi
}
```

**Process**:
1. Lấy tin nhắn cuối cùng từ user: `state['messages'][-1].content`
2. Parse lịch sử chat: `parsing_messages_to_history()`
3. Gửi prompt tới Groq LLM để phân loại
4. Parse kết quả JSON thành `TopicSchema`
5. Kiểm tra confidence score (nếu < 0.5 → off_topic)

**Output**:
```python
{
    "topic": TopicSchema(
        name="add_task",
        confidence=0.92,
        context="User input"
    ),
    "human_input": "Thêm việc học tiếng Anh",
    "ai_reply": None
}
```

**Ví Dụ Prompt**:
```
Role: Bạn là chuyên gia quản lý thời gian với 10 năm kinh nghiệm

Context:
Chat History: ...
User Input: "Thêm việc học tiếng Anh vào tối mai"

Task: Phân loại input vào một trong 4 topics: greeting, off_topic, add_task, get_tasks

Output: JSON với fields {name, confidence, context}
```

---

### 2. Flow Controller Node (`nodes/flow_controller.py`)

**Chức Năng**: Chọn subgraph tương ứng dựa trên topic

**Mapping**:
```python
TOPIC_FLOW_MAPPING = {
    "greeting": "greeting",           # → greeting_subgraph
    "off_topic": "off_topic",         # → off_topic_subgraph
    "add_task": "add_task",           # → add_task_subgraph
    "get_tasks": "get_tasks"          # → get_tasks_subgraph
}
```

**Output**:
```python
{"selected_flow": "add_task"}
```

---

### 3. Add Task Subgraph (`subgraphs/add_task_subgraph.py`)

#### 📝 Luồng Xử Lý Thêm Công Việc

```
Input: "Thêm việc học tiếng Anh vào tối mai"
    │
    ├─ Add Task Node:
    │  ├─ Gửi prompt tới LLM để trích xuất task
    │  ├─ LLM trả về JSON: [{title, start_at, due_at, description}]
    │  ├─ Validate & normalize dữ liệu
    │  └─ Insert vào Supabase
    │
    ├─ Reset Topic Node:
    │  └─ Clear topic để chuẩn bị cho request tiếp theo
    │
    └─ Output: "Dạ, em đã ghi lại: Học tiếng Anh (hạn: 2025-11-28 19:00:00)"
```

#### 🔧 Chi Tiết Add Task Node (`nodes/subgraph/add_task_nodes.py`)

**TaskModel Schema**:
```python
class TaskModel(BaseModel):
    title: str                  # Tiêu đề công việc (bắt buộc)
    start_at: str | datetime   # Thời gian bắt đầu (ISO-8601 hoặc "unknown")
    due_at: str | datetime     # Hạn chót (ISO-8601 hoặc "unknown")
    description: str           # Mô tả (tùy chọn)
```

**Xử Lý Thời Gian**:
- **Input**: "Tối mai" → **Output**: `2025-11-29T19:00:00+07:00`
- **Input**: "9h sáng" → **Output**: `T09:00:00+07:00` (ngày hiện tại)
- **Input**: "từ 9h đến 11h sáng mai" → **Output**: 
  - `start_at`: `2025-11-29T09:00:00+07:00`
  - `due_at`: `2025-11-29T11:00:00+07:00`

**Quy Tắc Tự Động**:
- "sáng" (7-12h) → ~09:00
- "trưa" (12-14h) → ~12:00
- "chiều" (14-18h) → ~15:00
- "tối" (18-22h) → ~19:00
- "đêm" (22-7h) → ~22:00

**Supabase Insert**:
```python
data = {
    "id": str(uuid.uuid4()),
    "title": "Học tiếng Anh",
    "description": "",
    "start_at": "2025-11-28T19:00:00+07:00",
    "due_at": "2025-11-28T20:30:00+07:00",
    "task_date": "2025-11-28",
    "created_at": datetime.now(LOCAL_TZ).isoformat(),
    "completed": False
}
supabase.table("tasks").insert(data).execute()
```

---

### 4. Get Tasks Subgraph (`subgraphs/get_tasks_subgraph.py`)

#### 📋 Luồng Xử Lý Xem Công Việc

```
Input: "Công việc hôm nay là gì?"
    │
    ├─ Get Tasks Node:
    │  ├─ Phân tích scope thời gian (today/tomorrow/week/all)
    │  ├─ Query Supabase dựa trên task_date
    │  ├─ Format kết quả
    │  └─ Tạo câu trả lời thân thiện
    │
    ├─ Reset Topic Node
    │
    └─ Output: "Dạ, hôm nay Anh/Chị có:
                - Học tiếng Anh (hạn: 2025-11-28)
                - Họp dự án (hạn: 2025-11-28)
                Anh/Chị có muốn em đặt nhắc nhở không ạ?"
```

**Time Scope Detection**:
```python
detect_time_prompt = """
Input: "Công việc hôm nay là gì?"
Output: {"scope": "today"}

Input: "Tuần này có gì không?"
Output: {"scope": "week"}

Input: "Liệt kê tất cả việc"
Output: {"scope": "all"}
"""
```

**Supabase Query**:
```python
if scope == "today":
    query = supabase.table("tasks").select("*") \
        .gte("due_date", str(today)) \
        .lt("due_date", str(today.replace(day=today.day + 1)))
elif scope == "week":
    query = supabase.table("tasks").select("*") \
        .gte("due_date", str(today)) \
        .lt("due_date", str(today.replace(day=today.day + 7)))
```

---

### 5. Greeting Subgraph (`subgraphs/greeting_subgraph.py`)

**Chức Năng**: Xử lý lời chào và giới thiệu bản thân

**Input**: "Chào em", "Hello", "Hôm nay thế nào?"

**Process**:
1. Tạo prompt với role, skills, tone từ `const_prompts.py`
2. Gửi tới LLM (Llama 70B)
3. LLM tạo phản hồi thân thiện, ngắn gọn < 200 từ

**Output**:
```
Dạ, chào Anh/Chị! Mình là AI Calendar - trợ lý AI chuyên về quản lý thời gian.

Mình có thể giúp Anh/Chị:
- Thêm, xem, cập nhật công việc
- Sắp xếp thời gian hợp lý
- Nhắc nhở deadline
- Gợi ý lịch trình tối ưu

Anh/Chị cần em hỗ trợ gì hôm nay ạ?
```

---

### 6. Prompts & Constants (`utils/const_prompts.py`)

**Các Thành Phần Prompt Chính**:

```python
CONST_AGENT_NAME = "AI Calendar"
CONST_AGENT_ROLE = "Trợ lý AI 10 năm kinh nghiệm quản lý thời gian"
CONST_AGENT_SKILLS = [
    "Phân tích NLP để hiểu yêu cầu",
    "Lập lịch & tạo nhắc nhở",
    "Tương tác database",
    "Gợi ý lịch trình cá nhân hóa"
]
CONST_AGENT_TONE = "Thân thiện, tập trung, khuyến khích kỷ luật"
CONST_FORM_ADDRESS_IN_VN = {
    "User xưng Anh → Assistant xưng Em",
    "User xưng Chị → Assistant xưng Em",
    "Không rõ → Assistant xưng Em"
}
```

---

## 💡 Ví Dụ Sử Dụng

### Ví Dụ 1: Thêm Công Việc

```
User: "Thêm việc dự án deadline vào thứ 6 tuần sau và gọi là 'Hoàn thành báo cáo'"

Flow:
1. Router Node → Detect: topic="add_task", confidence=0.98
2. Flow Controller → Select: "add_task_subgraph"
3. Add Task Node:
   - Parse: title="Hoàn thành báo cáo", due_at="2025-12-05T23:59:00+07:00"
   - Insert to Supabase
4. Reset Topic
5. AI Response:
   "Dạ, em đã ghi lại công việc:
    - Hoàn thành báo cáo (hạn: 2025-12-05T23:59:00+07:00)"
```

### Ví Dụ 2: Xem Công Việc Hôm Nay

```
User: "Hôm nay tôi có gì cần làm?"

Flow:
1. Router Node → Detect: topic="get_tasks", confidence=0.95
2. Flow Controller → Select: "get_tasks_subgraph"
3. Get Tasks Node:
   - Detect scope: "today"
   - Query: SELECT * FROM tasks WHERE task_date = '2025-11-28'
   - Format result
4. AI Response:
   "Dạ, hôm nay Anh/Chị có 3 việc cần làm:
    - Học tiếng Anh (hạn: 2025-11-28)
    - Họp dự án (hạn: 2025-11-28 14:00)
    - Viết email (hạn: 2025-11-28 17:00)
    
    Anh/Chị có muốn em đặt nhắc nhở không ạ?"
```

### Ví Dụ 3: Off Topic

```
User: "Kể một câu chuyện cười cho tôi"

Flow:
1. Router Node → Detect: topic="off_topic", confidence=0.88
2. Flow Controller → Select: "off_topic_subgraph"
3. Off Topic Node → Respond politely & redirect
4. AI Response:
   "Dạ, em xin lỗi! Em chuyên giúp Anh/Chị quản lý thời gian thôi ạ.
    
    Anh/Chị có muốn em hỗ trợ gì về công việc không?"
```

---

## 🐛 Khắc Phục Sự Cố

### Sự Cố 1: Import Errors

**Lỗi**:
```
ModuleNotFoundError: No module named 'litellm'
```

**Giải Pháp**:
```bash
pip install -r requirements.txt
# Hoặc cài lại toàn bộ
pip install --upgrade --force-reinstall -r requirements.txt
```

---

### Sự Cố 2: Supabase Connection Error

**Lỗi**:
```
Error connecting to Supabase: 401 Unauthorized
```

**Giải Pháp**:
1. Kiểm tra `.env` có biến `SUPABASE_URL` và `SUPABASE_KEY` không
2. Kiểm tra API Key có hợp lệ (copy lại từ Supabase Dashboard)
3. Kiểm tra Network/Firewall

---

### Sự Cố 3: LLM API Error

**Lỗi**:
```
GROQ_API_KEY is not set
```

**Giải Pháp**:
1. Lấy API Key từ [Groq Console](https://console.groq.com)
2. Thêm vào `.env`: `GROQ_API_KEY=gsk_xxxxx`
3. Reload environment: `python -m dotenv run python main.py`

---

### Sự Cố 4: JSON Parsing Error

**Lỗi**:
```
JSONDecodeError: Expecting value in LLM response
```

**Giải Pháp**:
- Kiểm tra `remove_think_tag()` function
- Thử tăng `temperature` trong completion call
- Check Groq API status

---

## 🔗 References

- **LangGraph**: https://github.com/langchain-ai/langgraph
- **Groq API**: https://console.groq.com
- **Supabase**: https://supabase.com
- **LangChain**: https://langchain.com

---

## 📞 Support

Nếu gặp vấn đề, hãy:
1. Kiểm tra logs trong `timemanage-agent-backend/logs/`
2. Enable debug mode bằng cách set `log_level="DEBUG"` trong `logger.py`
3. Liên hệ team hoặc mở issue trên GitHub

---

**Tài liệu này được cập nhật lần cuối**: 28/11/2025


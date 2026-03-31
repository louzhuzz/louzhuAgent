import uuid
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent import LearningAgent
from app_runtime import handle_user_input
from config import load_settings
from langsmith_observer import LangSmithObserver
from runtime_logger import RuntimeLogger


class ChatRequest(BaseModel):
    """描述网页端单次聊天请求的输入结构。"""

    message: str


app = FastAPI(title="个人学习助理 Agent Web")
settings = load_settings()
logger = RuntimeLogger(settings.log_directory)
observer = LangSmithObserver.from_settings(settings)
web_root = Path(__file__).resolve().parent.parent / "web"
agents_by_session: dict[str, LearningAgent] = {}


def get_or_create_agent(session_id: str) -> LearningAgent:
    """根据会话 ID 复用或创建独立的 Agent，避免不同浏览器共享记忆。"""
    agent = agents_by_session.get(session_id)
    if agent is None:
        agent = LearningAgent(settings)
        agents_by_session[session_id] = agent
    return agent


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """返回最小可用的网页聊天界面。"""
    html = (web_root / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request, response: Response) -> JSONResponse:
    """接收网页端消息，并复用主项目运行层返回统一结果。"""
    session_id = request.cookies.get("agent_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    agent = get_or_create_agent(session_id)
    result = handle_user_input(
        agent,
        payload.message,
        logger=logger,
        observer=observer,
        session_id=session_id,
    )

    if result["status"] == "exit":
        agents_by_session.pop(session_id, None)

    json_response = JSONResponse(result)
    json_response.set_cookie("agent_session_id", session_id, httponly=True)
    return json_response

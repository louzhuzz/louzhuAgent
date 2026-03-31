import json
from contextlib import nullcontext
from typing import Any

from agent import LearningAgent
from knowledge_qa import KnowledgeQARequest
from langsmith_observer import LangSmithObserver
from runtime_logger import RuntimeLogger
from study_plan import parse_study_plan_request
from task_breakdown import parse_task_breakdown_request


def _log(
    logger: RuntimeLogger | None,
    session_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """在有 logger 时记录结构化事件。"""
    if logger is None:
        return
    logger.log_event(event_type=event_type, payload=payload, session_id=session_id)


def _build_unexpected_error_result(prefix: str, exc: Exception) -> dict[str, Any]:
    """把未预期异常统一包装成用户可读结果。"""
    return {
        "status": "error",
        "kind": "message",
        "message": f"{prefix}：{type(exc).__name__}: {exc}",
    }


def handle_user_input(
    agent: LearningAgent,
    user_input: str,
    logger: RuntimeLogger | None = None,
    observer: LangSmithObserver | None = None,
    session_id: str = "cli",
) -> dict[str, Any]:
    """带可选日志与 LangSmith tracing 的统一请求入口。"""
    cleaned = user_input.strip()
    context = (
        observer.request_context(session_id=session_id, user_input=cleaned)
        if observer is not None
        else nullcontext()
    )
    with context:
        return _handle_user_input_core(
            agent=agent,
            user_input=user_input,
            logger=logger,
            session_id=session_id,
        )


def _handle_user_input_core(
    agent: LearningAgent,
    user_input: str,
    logger: RuntimeLogger | None = None,
    session_id: str = "cli",
) -> dict[str, Any]:
    """统一处理一条用户输入，供 CLI 和 Web 端共同复用。"""
    cleaned = user_input.strip()
    _log(
        logger,
        session_id,
        "user_input_received",
        {"raw_input": user_input, "cleaned_input": cleaned},
    )

    if not cleaned:
        result = {
            "status": "error",
            "kind": "message",
            "message": "请输入问题。",
        }
        _log(logger, session_id, "input_error", result)
        return result

    if cleaned.lower() in {"exit", "quit"}:
        result = {
            "status": "exit",
            "kind": "message",
            "message": "已退出。",
        }
        _log(logger, session_id, "session_exit", result)
        return result

    if cleaned == "/clear":
        agent.clear_history()
        result = {
            "status": "ok",
            "kind": "message",
            "message": "已清空当前对话记忆。",
        }
        _log(logger, session_id, "memory_cleared", result)
        return result

    if cleaned.startswith("/plan "):
        raw_request = cleaned[6:].strip()
        if not raw_request:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /plan 后面补充内容，例如 /plan LangChain | 零基础 | 5 | 做出一个可运行 Demo",
            }
            _log(logger, session_id, "study_plan_error", result)
            return result

        try:
            request = parse_study_plan_request(raw_request)
            plan = agent.create_study_plan(request)
        except ValueError as exc:
            result = {
                "status": "error",
                "kind": "message",
                "message": f"学习计划请求不合法：{exc}",
            }
            _log(logger, session_id, "study_plan_error", result)
            return result
        except Exception as exc:
            result = _build_unexpected_error_result("学习计划执行失败", exc)
            _log(logger, session_id, "study_plan_error", result)
            return result

        result = {
            "status": "ok",
            "kind": "json",
            "title": "Agent(JSON)",
            "data": plan,
        }
        _log(
            logger,
            session_id,
            "study_plan_created",
            {
                "topic": request.topic,
                "days": request.days,
                "current_level": request.current_level,
            },
        )
        return result

    if cleaned.startswith("/breakdown "):
        raw_request = cleaned[11:].strip()
        if not raw_request:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /breakdown 后面补充内容，例如 /breakdown 做一个个人学习助理 | 零基础 | 14 | 可执行步骤",
            }
            _log(logger, session_id, "task_breakdown_error", result)
            return result

        try:
            request = parse_task_breakdown_request(raw_request)
            result = agent.create_task_breakdown(request)
        except ValueError as exc:
            error_result = {
                "status": "error",
                "kind": "message",
                "message": f"任务拆解请求不合法：{exc}",
            }
            _log(logger, session_id, "task_breakdown_error", error_result)
            return error_result
        except Exception as exc:
            error_result = _build_unexpected_error_result("任务拆解执行失败", exc)
            _log(logger, session_id, "task_breakdown_error", error_result)
            return error_result

        response = {
            "status": "ok",
            "kind": "json",
            "title": "Agent(JSON)",
            "data": result,
        }
        _log(
            logger,
            session_id,
            "task_breakdown_created",
            {
                "goal": request.goal,
                "available_days": request.available_days,
                "output_style": request.output_style,
            },
        )
        return response

    if cleaned.startswith("/qa "):
        question = cleaned[4:].strip()
        if not question:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /qa 后面补充问题。",
            }
            _log(logger, session_id, "knowledge_qa_error", result)
            return result

        try:
            result = agent.answer_knowledge_question(KnowledgeQARequest(question=question))
        except (FileNotFoundError, ValueError) as exc:
            error_result = {
                "status": "error",
                "kind": "message",
                "message": f"知识点问答执行失败：{exc}",
            }
            _log(logger, session_id, "knowledge_qa_error", error_result)
            return error_result
        except Exception as exc:
            error_result = _build_unexpected_error_result("知识点问答执行失败", exc)
            _log(logger, session_id, "knowledge_qa_error", error_result)
            return error_result

        response = {
            "status": "ok",
            "kind": "knowledge_qa",
            "title": "Agent(QA)",
            "data": result,
        }
        _log(
            logger,
            session_id,
            "knowledge_qa_completed",
            {
                "question": question,
                "selected_notes": result["selected_notes"],
                "retrieved_chunk_count": len(result["retrieved_chunks"]),
                "index_statuses": result["index_statuses"],
            },
        )
        return response

    if cleaned.startswith("/v1 "):
        question = cleaned[4:].strip()
        if not question:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /v1 后面补充问题。",
            }
            _log(logger, session_id, "agent_v1_error", result)
            return result

        try:
            result = agent.run_agent_v1(question)
        except (FileNotFoundError, ValueError) as exc:
            error_result = {
                "status": "error",
                "kind": "message",
                "message": f"Agent v1 执行失败：{exc}",
            }
            _log(logger, session_id, "agent_v1_error", error_result)
            return error_result
        except Exception as exc:
            error_result = _build_unexpected_error_result("Agent v1 执行失败", exc)
            _log(logger, session_id, "agent_v1_error", error_result)
            return error_result

        response = {
            "status": "ok",
            "kind": "agent_v1",
            "title": "Agent v1",
            "data": result,
        }
        _log(
            logger,
            session_id,
            "agent_v1_completed",
            {
                "input": question,
                "intent": result["intent"],
                "result_type": result["result_type"],
            },
        )
        return response

    if cleaned == "/notes":
        result = {
            "status": "ok",
            "kind": "notes",
            "title": "知识点文件",
            "data": agent.list_notes_tool(),
        }
        _log(logger, session_id, "notes_listed", {"note_count": len(result["data"])})
        return result

    if cleaned == "/tools":
        result = {
            "status": "ok",
            "kind": "json",
            "title": "工具 Schema",
            "data": agent.get_tool_schemas(),
        }
        _log(logger, session_id, "tool_schemas_viewed", {"tool_count": len(result["data"])})
        return result

    if cleaned.startswith("/read "):
        file_name = cleaned[6:].strip()
        if not file_name:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /read 后面补充文件名，例如 /read 13-输出解析器.md",
            }
            _log(logger, session_id, "read_note_error", result)
            return result

        try:
            content = agent.read_note_tool(file_name)
        except (FileNotFoundError, ValueError) as exc:
            result = {
                "status": "error",
                "kind": "message",
                "message": f"读取失败：{exc}",
            }
            _log(logger, session_id, "read_note_error", result)
            return result

        result = {
            "status": "ok",
            "kind": "content",
            "title": f"文件内容：{file_name}",
            "data": content,
        }
        _log(logger, session_id, "note_read", {"file_name": file_name, "content_length": len(content)})
        return result

    if cleaned.startswith("/tool "):
        payload = cleaned[6:].strip()
        if not payload:
            result = {
                "status": "error",
                "kind": "message",
                "message": '请使用 /tool 工具名 JSON参数，例如 /tool read_note {"file_name":"13-输出解析器.md"}',
            }
            _log(logger, session_id, "tool_call_error", result)
            return result

        tool_name, separator, raw_arguments = payload.partition(" ")
        if not separator:
            result = {
                "status": "error",
                "kind": "message",
                "message": '请补充 JSON 参数，例如 /tool read_note {"file_name":"13-输出解析器.md"}',
            }
            _log(logger, session_id, "tool_call_error", result)
            return result

        try:
            arguments = json.loads(raw_arguments)
            result = agent.execute_tool(tool_name, arguments)
        except json.JSONDecodeError as exc:
            error_result = {
                "status": "error",
                "kind": "message",
                "message": f"工具参数不是合法 JSON：{exc}",
            }
            _log(logger, session_id, "tool_call_error", error_result)
            return error_result
        except (FileNotFoundError, ValueError) as exc:
            error_result = {
                "status": "error",
                "kind": "message",
                "message": f"工具执行失败：{exc}",
            }
            _log(logger, session_id, "tool_call_error", error_result)
            return error_result
        except Exception as exc:
            error_result = _build_unexpected_error_result("工具执行失败", exc)
            _log(logger, session_id, "tool_call_error", error_result)
            return error_result

        response = {
            "status": "ok",
            "kind": "tool_result",
            "title": "工具执行结果",
            "data": result,
        }
        _log(
            logger,
            session_id,
            "tool_call_completed",
            {"tool_name": tool_name, "arguments": arguments},
        )
        return response

    if cleaned.startswith("/study "):
        question = cleaned[7:].strip()
        if not question:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /study 后面补充问题。",
            }
            _log(logger, session_id, "study_mode_error", result)
            return result

        try:
            file_name, answer = agent.answer_with_note_tool(question)
        except (FileNotFoundError, ValueError) as exc:
            result = {
                "status": "error",
                "kind": "message",
                "message": f"工具型问答执行失败：{exc}",
            }
            _log(logger, session_id, "study_mode_error", result)
            return result
        except Exception as exc:
            result = _build_unexpected_error_result("工具型问答执行失败", exc)
            _log(logger, session_id, "study_mode_error", result)
            return result

        response = {
            "status": "ok",
            "kind": "study",
            "title": "Agent(Tool)",
            "data": {
                "file_name": file_name,
                "answer": answer,
            },
        }
        _log(
            logger,
            session_id,
            "study_mode_completed",
            {"question": question, "file_name": file_name},
        )
        return response

    if cleaned.startswith("/agent "):
        question = cleaned[7:].strip()
        if not question:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /agent 后面补充问题。",
            }
            _log(logger, session_id, "tool_agent_error", result)
            return result

        try:
            result = agent.run_tool_calling_agent(question)
        except (FileNotFoundError, ValueError) as exc:
            error_result = {
                "status": "error",
                "kind": "message",
                "message": f"Tool Calling Agent 执行失败：{exc}",
            }
            _log(logger, session_id, "tool_agent_error", error_result)
            return error_result
        except Exception as exc:
            error_result = _build_unexpected_error_result("Tool Calling Agent 执行失败", exc)
            _log(logger, session_id, "tool_agent_error", error_result)
            return error_result

        response = {
            "status": "ok",
            "kind": "tool_agent",
            "title": "Agent(Tool Calling)",
            "data": result,
        }
        _log(
            logger,
            session_id,
            "tool_agent_completed",
            {
                "question": question,
                "step_count": len(result["steps"]),
            },
        )
        return response

    if cleaned.startswith("/react "):
        question = cleaned[7:].strip()
        if not question:
            result = {
                "status": "error",
                "kind": "message",
                "message": "请在 /react 后面补充问题。",
            }
            _log(logger, session_id, "react_error", result)
            return result

        try:
            result = agent.run_react_agent(question)
        except (FileNotFoundError, ValueError) as exc:
            error_result = {
                "status": "error",
                "kind": "message",
                "message": f"ReAct Agent 执行失败：{exc}",
            }
            _log(logger, session_id, "react_error", error_result)
            return error_result
        except Exception as exc:
            error_result = _build_unexpected_error_result("ReAct Agent 执行失败", exc)
            _log(logger, session_id, "react_error", error_result)
            return error_result

        response = {
            "status": "ok",
            "kind": "react",
            "title": "Agent(ReAct)",
            "data": result,
        }
        _log(
            logger,
            session_id,
            "react_completed",
            {
                "question": question,
                "step_count": len(result["steps"]),
            },
        )
        return response

    try:
        answer = agent.reply(cleaned)
    except Exception as exc:
        result = _build_unexpected_error_result("普通对话执行失败", exc)
        _log(logger, session_id, "chat_error", result)
        return result
    result = {
        "status": "ok",
        "kind": "text",
        "title": "Agent",
        "data": answer,
    }
    _log(
        logger,
        session_id,
        "chat_completed",
        {"input": cleaned, "answer_length": len(answer)},
    )
    return result


def render_cli_result(result: dict[str, Any]) -> str:
    """把统一结果结构渲染成命令行可直接打印的文本。"""
    kind = result["kind"]

    if kind == "message":
        return result["message"]

    if kind == "json":
        return f"\n{result['title']}:\n{json.dumps(result['data'], ensure_ascii=False, indent=2)}"

    if kind == "notes":
        lines = [f"\n{result['title']}:"]
        lines.extend(f"- {note}" for note in result["data"])
        return "\n".join(lines)

    if kind == "content":
        return f"\n{result['title']}\n\n{result['data']}"

    if kind == "tool_result":
        data = result["data"]
        if isinstance(data, str):
            return f"\n{result['title']}:\n{data}"
        return f"\n{result['title']}:\n{json.dumps(data, ensure_ascii=False, indent=2)}"

    if kind == "study":
        data = result["data"]
        return (
            f"\n本次自动选择的知识点文件: {data['file_name']}\n"
            f"\n{result['title']}: {data['answer']}"
        )

    if kind == "tool_agent":
        lines = ["\nTool Calling 步骤:"]
        for step in result["data"]["steps"]:
            lines.append(f"- 第 {step['step']} 步: {step['tool_name']} {step['arguments']}")
            lines.append(f"  决策原因: {step['reason']}")
        lines.append(f"\n{result['title']}: {result['data']['answer']}")
        return "\n".join(lines)

    if kind == "react":
        lines = ["\nReAct 步骤:"]
        for step in result["data"]["steps"]:
            lines.append(f"- 第 {step['step']} 步")
            lines.append(f"  Thought: {step['thought']}")
            lines.append(f"  Action: {step['tool_name']} {step['arguments']}")
        lines.append(f"\n{result['title']}: {result['data']['answer']}")
        return "\n".join(lines)

    if kind == "knowledge_qa":
        data = result["data"]
        lines = ["\n本次选中的知识点文件:"]
        lines.extend(f"- {file_name}" for file_name in data["selected_notes"])
        lines.append("\n索引状态:")
        for status in data["index_statuses"]:
            lines.append(
                f"- {status['file_name']}: {status['status']} (chunk 数: {status['chunk_count']})"
            )
        lines.append("\n命中的资料片段:")
        for chunk in data["retrieved_chunks"]:
            lines.append(
                f"- {chunk['file_name']} [chunk_index={chunk['chunk_index']}] "
                f"({chunk['chunk_start']}, {chunk['chunk_end']}) "
                f"distance={chunk['distance']:.4f}"
            )
        lines.append(f"\n{result['title']}: {data['answer']}")
        return "\n".join(lines)

    if kind == "agent_v1":
        data = result["data"]
        lines = [
            f"\nAgent v1 路由意图: {data['intent']}",
            f"路由原因: {data['reason']}",
        ]
        if data["result_type"] == "json":
            lines.append(f"\nAgent v1(JSON):\n{json.dumps(data['result'], ensure_ascii=False, indent=2)}")
            return "\n".join(lines)
        if data["result_type"] == "knowledge_qa":
            qa_result = data["result"]
            lines.append("\n本次选中的知识点文件:")
            lines.extend(f"- {file_name}" for file_name in qa_result["selected_notes"])
            lines.append("\n索引状态:")
            for status in qa_result["index_statuses"]:
                lines.append(
                    f"- {status['file_name']}: {status['status']} (chunk 数: {status['chunk_count']})"
                )
            lines.append("\n命中的资料片段:")
            for chunk in qa_result["retrieved_chunks"]:
                lines.append(
                    f"- {chunk['file_name']} [chunk_index={chunk['chunk_index']}] "
                    f"({chunk['chunk_start']}, {chunk['chunk_end']}) "
                    f"distance={chunk['distance']:.4f}"
                )
            lines.append(f"\nAgent v1(QA): {qa_result['answer']}")
            return "\n".join(lines)
        if data["result_type"] == "tool_agent":
            lines.append("\nTool Calling 步骤:")
            for step in data["result"]["steps"]:
                lines.append(f"- 第 {step['step']} 步: {step['tool_name']} {step['arguments']}")
                lines.append(f"  决策原因: {step['reason']}")
            lines.append(f"\nAgent v1(Tool Agent): {data['result']['answer']}")
            return "\n".join(lines)
        lines.append(f"\nAgent v1: {data['result']}")
        return "\n".join(lines)

    if kind == "text":
        return f"\n{result['title']}: {result['data']}"

    return json.dumps(result, ensure_ascii=False, indent=2)

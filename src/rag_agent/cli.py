"""Command-line interface: ingest / ask / chat / serve / eval."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .app import RAGApp
from .config import load_config, load_env_file
from .logging_utils import setup_logging


def _use_mock() -> None:
    """Force fully-offline providers (no API key, no network)."""
    os.environ["RAG__LLM__PROVIDER"] = "mock"
    os.environ["RAG__EMBEDDING__PROVIDER"] = "mock"
    os.environ["RAG__STORE__NAME"] = "mock"  # keep the offline index separate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-agent", description="知识库问答 Agent")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--mock", action="store_true", help="使用离线 mock provider（无需 API key）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="把文档摄入知识库")
    p_ingest.add_argument("path", help="文件或目录路径")

    p_ask = sub.add_parser("ask", help="问一个单轮问题")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=["rag", "agent"], default="rag")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--session", default=None, help="会话 id（启用多轮记忆）")

    sub.add_parser("chat", help="进入交互式多轮对话")

    sub.add_parser("serve", help="启动 HTTP 服务")

    p_eval = sub.add_parser("eval", help="在评测集上评估检索与回答质量")
    p_eval.add_argument("--dataset", default="data/eval/qa.jsonl")
    p_eval.add_argument("--mode", choices=["rag", "agent"], default="rag")

    return parser


def _print_payload(payload: dict, original_question: str | None = None) -> None:
    rewritten = payload.get("query")
    if rewritten and rewritten != original_question:
        print(f"（改写后查询：{rewritten}）")
    print(payload["answer"])
    for step in payload.get("steps", []):
        print(f"  · 工具 {step['tool']}({step['arguments']}) -> {step['observation'][:80]}...")
    citations = payload.get("citations", [])
    if citations:
        print("\n来源：")
        for c in citations:
            name = Path(c["source"]).name if c["source"] else "?"
            print(f"  [{c['id']}] {name} #{c['index']}  score={c['score']}")


def _chat(app: RAGApp, session: str) -> None:
    print("进入交互模式：直接输入问题；/reset 清空记忆，/exit 退出。")
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/reset":
            app.sessions.reset(session)
            print("记忆已清空。")
            continue
        _print_payload(app.answer(line, session_id=session, mode="rag"), line)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    if args.mock:
        _use_mock()

    config = load_config(args.config)
    setup_logging(config.logging.level)
    app = RAGApp(config)

    if args.command == "ingest":
        added = app.ingest(args.path)
        print(f"新增 {added} 个片段，知识库现有 {len(app.store)} 个片段。")
    elif args.command == "ask":
        payload = app.answer(args.question, session_id=args.session, mode=args.mode, top_k=args.top_k)
        _print_payload(payload, args.question)
    elif args.command == "chat":
        _chat(app, session="cli")
    elif args.command == "serve":
        import uvicorn

        from .api.server import create_app

        print(f"服务启动于 http://{config.server.host}:{config.server.port}")
        uvicorn.run(create_app(app), host=config.server.host, port=config.server.port)
    elif args.command == "eval":
        from .eval.runner import run_eval

        run_eval(app, args.dataset, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

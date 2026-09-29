"""极简入口 —— 少打字。

用法：
    python run.py "什么是混合检索"      # 直接提问（默认 RAG 模式）
    python run.py -a "Agentic RAG 是什么"  # Agent 模式
    python run.py chat                    # 交互式多轮
    python run.py ingest data/docs        # 摄入文档
    python run.py serve                   # 起 HTTP 服务
    python run.py eval                    # 评测

不带参数直接回车 = 进入 chat 交互模式。
"""
import sys
from pathlib import Path

# 让 Windows 控制台也能正确打印中文
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_agent.cli import main  # noqa: E402

_COMMANDS = {"ingest", "ask", "chat", "serve", "eval", "-h", "--help", "--mock", "--config"}


def _dispatch(argv: list[str]) -> int:
    if not argv:
        return main(["chat"])  # 直接进交互模式
    if argv[0] in _COMMANDS:
        return main(argv)
    if argv[0] == "-a":  # 快捷 Agent 模式
        return main(["ask", " ".join(argv[1:]), "--mode", "agent"])
    return main(["ask", " ".join(argv)])  # 其余一律当作问题


if __name__ == "__main__":
    raise SystemExit(_dispatch(sys.argv[1:]))

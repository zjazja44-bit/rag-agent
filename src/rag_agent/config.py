"""Central configuration: dataclass defaults + YAML file + environment overrides.

Precedence (low -> high):
    dataclass defaults  <  config.yaml  <  environment variables

Environment override naming is ``RAG__<SECTION>__<FIELD>`` (case-insensitive),
for example ``RAG__LLM__CHAT_MODEL=gpt-4o-mini`` or
``RAG__EMBEDDING__PROVIDER=mock``. This lets tests and CI run with zero
network access without touching any file.
"""
from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

ENV_PREFIX = "RAG__"


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    provider: str = "openai_compat"          # openai_compat | mock
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    chat_model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 1024
    request_timeout: float = 60.0

    @property
    def api_key(self) -> str | None:
        """Resolve the key from the environment *now* (not at load time)."""
        return os.environ.get(self.api_key_env)


@dataclass
class EmbeddingConfig:
    provider: str = "openai_compat"          # openai_compat | local | mock
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "text-embedding-3-small"
    dim: int = 1536                          # used only by local/mock providers
    batch_size: int = 32

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass
class IngestConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 64
    splitter: str = "recursive"              # recursive | fixed


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    rerank: bool = True
    rerank_top_n: int = 5


@dataclass
class StoreConfig:
    dir: str = "./artifacts/store"
    name: str = "default"

    @property
    def path(self) -> Path:
        return Path(self.dir) / self.name


@dataclass
class AgentConfig:
    max_steps: int = 6
    tools: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = ["knowledge_search", "calculator"]


@dataclass
class MemoryConfig:
    max_turns: int = 10


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AppConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    ingest: IngestConfig
    retrieval: RetrievalConfig
    store: StoreConfig
    agent: AgentConfig
    memory: MemoryConfig
    server: ServerConfig
    logging: LoggingConfig


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def _coerce(value: str, template: Any) -> Any:
    """Convert an env-var string into the type of the existing value."""
    if isinstance(template, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def _section(cls: type[T], data: dict[str, Any], section: str) -> T:
    """Build one config section from YAML data plus env overrides."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        default = None if f.default is MISSING else f.default
        if f.name in data:
            kwargs[f.name] = data[f.name]
        env_name = f"{ENV_PREFIX}{section}__{f.name}".upper()
        if env_name in os.environ:
            kwargs[f.name] = _coerce(os.environ[env_name], kwargs.get(f.name, default))
    return cls(**kwargs)  # type: ignore[call-arg]


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple ``KEY=VALUE`` lines from a .env file (no third-party dep).

    Existing environment variables take precedence, so an explicit ``export``
    always wins over the file.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file (missing file => all defaults)."""
    raw: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    return AppConfig(
        llm=_section(LLMConfig, raw.get("llm", {}), "LLM"),
        embedding=_section(EmbeddingConfig, raw.get("embedding", {}), "EMBEDDING"),
        ingest=_section(IngestConfig, raw.get("ingest", {}), "INGEST"),
        retrieval=_section(RetrievalConfig, raw.get("retrieval", {}), "RETRIEVAL"),
        store=_section(StoreConfig, raw.get("store", {}), "STORE"),
        agent=_section(AgentConfig, raw.get("agent", {}), "AGENT"),
        memory=_section(MemoryConfig, raw.get("memory", {}), "MEMORY"),
        server=_section(ServerConfig, raw.get("server", {}), "SERVER"),
        logging=_section(LoggingConfig, raw.get("logging", {}), "LOGGING"),
    )

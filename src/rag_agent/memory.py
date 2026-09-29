"""Conversation memory and per-session storage."""
from __future__ import annotations

from .llm.base import Message


class ConversationMemory:
    """A bounded rolling window of user/assistant turns."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max(1, max_turns)
        self._messages: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._trim()

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def history(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        limit = self.max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]


class SessionStore:
    """Lazily-created conversation memories keyed by session id."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, ConversationMemory] = {}

    def get(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(self.max_turns)
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

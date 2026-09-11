from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from threading import Lock
from typing import Any, Iterable, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph


class ChatModel(Protocol):
    def invoke(self, payload: Any) -> Any: ...


class EmbeddingModel(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class RubberDuckyAgentGraph:
    CONTROLLER_PROMPT = (
        "You are a routing controller. Read the user message and reply with exactly "
        "one label: 'brainstorm' when the user wants idea generation, exploration, "
        "or creative expansion; otherwise reply with 'problem_solving' when the "
        "user wants help thinking through a logic, math, or analytical problem."
    )
    BRAINSTORMING_PROMPT = (
        "You are a brainstorming partner. Analyze any ideas the user already "
        "provided and add your own relevant ideas about the topic at hand. Speak "
        "with brevity. Do not explain your thought process to the user."
    )
    PROBLEM_SOLVING_PROMPT = (
        "You are a teacher guiding a student through a math or logic problem. Ask "
        "probing questions about the user's prompt, but do not offer solutions. "
        "Speak with brevity. Do not explain your thought process to the user."
    )

    class State(TypedDict, total=False):
        session_id: str
        user_input: str
        route: Literal["brainstorm", "problem_solving"]
        memories: list[str]
        response: str
        history: list[dict[str, str]]

    @dataclass
    class MemoryEntry:
        session_id: str
        text: str
        embedding: list[float]

    def __init__(
        self,
        model: ChatModel,
        embeddings: EmbeddingModel,
        *,
        controller_model: ChatModel | None = None,
        brainstorming_model: ChatModel | None = None,
        problem_solving_model: ChatModel | None = None,
        memory_k: int = 4,
    ) -> None:
        self.controller_model = controller_model or model
        self.brainstorming_model = brainstorming_model or model
        self.problem_solving_model = problem_solving_model or model
        self.embeddings = embeddings
        self.memory_k = memory_k
        self._histories: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._session_locks_guard = Lock()
        self._session_locks: dict[str, Lock] = {}
        self._memory_index: list[RubberDuckyAgentGraph.MemoryEntry] = []
        self.graph = self._build_graph()

    def invoke(self, user_input: str, *, session_id: str = "default") -> dict[str, Any]:
        with self._session_locks_guard:
            session_lock = self._session_locks.setdefault(session_id, Lock())
        with session_lock:
            initial_state: RubberDuckyAgentGraph.State = {
                "session_id": session_id,
                "user_input": user_input,
                "history": list(self._histories.get(session_id, [])),
            }
            result = self.graph.invoke(initial_state)
            self._histories[session_id] = result["history"]
        return dict(result)

    def _build_graph(self):
        graph = StateGraph(self.State)
        graph.add_node("controller", self._controller_node)
        graph.add_node("brainstorm_agent", self._brainstorming_node)
        graph.add_node("problem_solving_agent", self._problem_solving_node)
        graph.add_edge(START, "controller")
        graph.add_conditional_edges(
            "controller",
            self._route_after_controller,
            {
                "brainstorm": "brainstorm_agent",
                "problem_solving": "problem_solving_agent",
            },
        )
        graph.add_edge("brainstorm_agent", END)
        graph.add_edge("problem_solving_agent", END)
        return graph.compile()

    def _controller_node(self, state: dict[str, Any]) -> dict[str, Any]:
        route = self._classify_route(state["user_input"])
        return {
            "route": route,
            "memories": self._recall_memories(state["session_id"], state["user_input"]),
        }

    def _brainstorming_node(self, state: dict[str, Any]) -> dict[str, Any]:
        response = self._invoke_agent(
            self.brainstorming_model,
            self.BRAINSTORMING_PROMPT,
            state["user_input"],
            state.get("history", []),
            state.get("memories", []),
        )
        return self._finalize_turn(state, response)

    def _problem_solving_node(self, state: dict[str, Any]) -> dict[str, Any]:
        response = self._invoke_agent(
            self.problem_solving_model,
            self.PROBLEM_SOLVING_PROMPT,
            state["user_input"],
            state.get("history", []),
            state.get("memories", []),
        )
        return self._finalize_turn(state, response)

    def _route_after_controller(self, state: dict[str, Any]) -> str:
        return state["route"]

    def _classify_route(self, user_input: str) -> Literal["brainstorm", "problem_solving"]:
        try:
            raw_result = self.controller_model.invoke(
                [
                    {"role": "system", "content": self.CONTROLLER_PROMPT},
                    {"role": "user", "content": user_input},
                ]
            )
            normalized = self._stringify_response(raw_result).strip().lower()
            if normalized in {"brainstorm", "problem_solving"}:
                return normalized
        except Exception:
            pass

        brainstorm_keywords = (
            "brainstorm",
            "ideas",
            "creative",
            "name",
            "topics",
            "ways to",
            "generate",
        )
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in brainstorm_keywords):
            return "brainstorm"
        return "problem_solving"

    def _invoke_agent(
        self,
        model: ChatModel,
        system_prompt: str,
        user_input: str,
        history: list[dict[str, str]],
        memories: list[str],
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if memories:
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant conversation memories:\n- " + "\n- ".join(memories),
                }
            )
        if history:
            messages.append(
                {
                    "role": "system",
                    "content": "Recent conversation history:\n"
                    + "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:]),
                }
            )
        messages.append({"role": "user", "content": user_input})

        try:
            return self._stringify_response(model.invoke(messages))
        except Exception:
            prompt = "\n\n".join(message["content"] for message in messages)
            return self._stringify_response(model.invoke(prompt))

    def _finalize_turn(self, state: dict[str, Any], response: str) -> dict[str, Any]:
        history = list(state.get("history", []))
        history.extend(
            [
                {"role": "user", "content": state["user_input"]},
                {"role": "assistant", "content": response},
            ]
        )
        self._store_memory(state["session_id"], state["user_input"])
        self._store_memory(state["session_id"], response)
        return {"response": response, "history": history}

    def _store_memory(self, session_id: str, text: str) -> None:
        self._memory_index.append(
            self.MemoryEntry(
                session_id=session_id,
                text=text,
                embedding=self._embed_text(text),
            )
        )

    def _recall_memories(self, session_id: str, query: str) -> list[str]:
        query_embedding = self._embed_text(query)
        scored = []
        for memory in self._memory_index:
            if memory.session_id != session_id:
                continue
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            scored.append((similarity, memory.text))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for score, text in scored[: self.memory_k] if score > 0]

    def _embed_text(self, text: str) -> list[float]:
        if hasattr(self.embeddings, "embed_documents"):
            return list(self.embeddings.embed_documents([text])[0])
        return list(self.embeddings.embed_query(text))

    @staticmethod
    def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        left_values = list(left)
        right_values = list(right)
        numerator = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = sqrt(sum(value * value for value in left_values))
        right_norm = sqrt(sum(value * value for value in right_values))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _stringify_response(result: Any) -> str:
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

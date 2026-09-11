import unittest
from threading import Lock, Thread
from time import sleep

from rubber_ducky.agent_graph import RubberDuckyAgentGraph


class FakeEmbeddings:
    WORDS = (
        "brainstorm",
        "name",
        "robot",
        "garden",
        "solar",
        "teacher",
        "equation",
        "logic",
    )

    def embed_query(self, text):
        lowered = text.lower()
        return [float(word in lowered) for word in self.WORDS]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if callable(self.response):
            return self.response(payload)
        return self.response


class RubberDuckyAgentGraphTests(unittest.TestCase):
    def test_brainstorm_route_uses_brainstorm_prompt(self):
        controller = FakeModel("brainstorm")
        brainstorming = FakeModel("Try short names like SunSprout or BeamBloom.")
        problem_solving = FakeModel("This should not run.")

        graph = RubberDuckyAgentGraph(
            model=brainstorming,
            embeddings=FakeEmbeddings(),
            controller_model=controller,
            brainstorming_model=brainstorming,
            problem_solving_model=problem_solving,
        )

        result = graph.invoke("Brainstorm names for a solar garden robot.", session_id="ideas")

        self.assertEqual(result["route"], "brainstorm")
        self.assertEqual(result["response"], "Try short names like SunSprout or BeamBloom.")
        system_prompt = brainstorming.calls[0][0]["content"]
        self.assertIn("Analyze any ideas the user already provided", system_prompt)
        self.assertIn("Speak with brevity", system_prompt)
        self.assertEqual(problem_solving.calls, [])

    def test_problem_solving_route_uses_teacher_prompt(self):
        controller = FakeModel("problem_solving")
        brainstorming = FakeModel("This should not run.")
        problem_solving = FakeModel("What changes if you isolate one side of the equation first?")

        graph = RubberDuckyAgentGraph(
            model=problem_solving,
            embeddings=FakeEmbeddings(),
            controller_model=controller,
            brainstorming_model=brainstorming,
            problem_solving_model=problem_solving,
        )

        result = graph.invoke("I am stuck on a logic equation proof.", session_id="math")

        self.assertEqual(result["route"], "problem_solving")
        self.assertEqual(
            result["response"],
            "What changes if you isolate one side of the equation first?",
        )
        system_prompt = problem_solving.calls[0][0]["content"]
        self.assertIn("Ask probing questions", system_prompt)
        self.assertIn("do not offer solutions", system_prompt)
        self.assertEqual(brainstorming.calls, [])

    def test_retrieves_embedding_memories_for_follow_up_turns(self):
        controller = FakeModel("brainstorm")
        brainstorming = FakeModel("Maybe explore playful solar-themed names.")

        graph = RubberDuckyAgentGraph(
            model=brainstorming,
            embeddings=FakeEmbeddings(),
            controller_model=controller,
            brainstorming_model=brainstorming,
        )

        graph.invoke("Brainstorm solar robot names for my garden.", session_id="shared")
        result = graph.invoke("More solar ideas for the garden robot?", session_id="shared")

        self.assertTrue(
            any("solar robot names" in memory.lower() for memory in result["memories"]),
            result["memories"],
        )
        memory_prompt = brainstorming.calls[-1][1]["content"]
        self.assertIn("Relevant conversation memories", memory_prompt)
        controller_prompt = controller.calls[-1][1]["content"]
        self.assertIn("Relevant conversation memories", controller_prompt)

    def test_controller_can_route_using_conversational_memory(self):
        def controller_response(payload):
            prompt = payload[1]["content"]
            if (
                "Relevant conversation memories" in prompt
                and "Brainstorm product names." in prompt
            ):
                return "brainstorm"
            return "problem_solving"

        controller = FakeModel(controller_response)
        brainstorming = FakeModel("Let us try fun brand-name options.")
        problem_solving = FakeModel("Can you rewrite the constraint in symbols?")

        graph = RubberDuckyAgentGraph(
            model=brainstorming,
            embeddings=FakeEmbeddings(),
            controller_model=controller,
            brainstorming_model=brainstorming,
            problem_solving_model=problem_solving,
        )

        graph.invoke("Brainstorm product names.", session_id="shared")
        result = graph.invoke("More name ideas?", session_id="shared")

        self.assertEqual(result["route"], "brainstorm")

    def test_invocations_for_same_session_are_serialized(self):
        active_calls = 0
        max_active_calls = 0
        active_calls_lock = Lock()

        def slow_response(payload):
            nonlocal active_calls, max_active_calls
            with active_calls_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            sleep(0.05)
            with active_calls_lock:
                active_calls -= 1
            if isinstance(payload, list) and payload and payload[0]["content"].startswith(
                "You are a routing controller"
            ):
                return "brainstorm"
            return "Thread-safe response."

        model = FakeModel(slow_response)
        graph = RubberDuckyAgentGraph(
            model=model,
            embeddings=FakeEmbeddings(),
            controller_model=model,
            brainstorming_model=model,
            problem_solving_model=model,
        )

        thread_one = Thread(
            target=graph.invoke,
            args=("Brainstorm product names.",),
            kwargs={"session_id": "shared"},
        )
        thread_two = Thread(
            target=graph.invoke,
            args=("Brainstorm playful ones.",),
            kwargs={"session_id": "shared"},
        )

        thread_one.start()
        thread_two.start()
        thread_one.join()
        thread_two.join()

        self.assertEqual(max_active_calls, 1)


if __name__ == "__main__":
    unittest.main()

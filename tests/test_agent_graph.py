import unittest

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


if __name__ == "__main__":
    unittest.main()

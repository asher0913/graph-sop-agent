import unittest

from graph_sop_agent.core import Document, fixture


class SOPTests(unittest.TestCase):
    def test_exception_routes_and_cites(self):
        answer = fixture().answer("damaged parcel exception")
        self.assertEqual(answer["intent"], "exception")
        self.assertIn("claims team", answer["graph_entities"])
        self.assertTrue(answer["citations"])

    def test_procedure_retrieval(self):
        answer = fixture().answer("steps after label creation")
        self.assertIn("sop-3", answer["citations"])

    def test_rejected_learning_is_ignored(self):
        agent = fixture()
        self.assertFalse(agent.learn(Document("new", "x", "new process"), accepted=False))
        self.assertNotIn("new", [doc.doc_id for doc in agent.documents])

    def test_accepted_learning_is_searchable(self):
        agent = fixture()
        self.assertTrue(agent.learn(Document("new", "Returns", "Return authorization requires manager approval."), True))
        self.assertIn("new", agent.answer("return authorization approval")["citations"])


if __name__ == "__main__":
    unittest.main()

"""
Unit and Integration tests for OmniDoc-RAG
"""

import unittest
from src.retrieval.retriever import expand_query, get_related_subjects, is_context_relevant, is_garbled_ocr
from src.utils.helpers import load_app_config, is_short_query, SUBJECT_METADATA
from src.llm.llm_client import is_ollama_online


class TestOmniDocRAG(unittest.TestCase):

    def test_config_loading(self):
        cfg = load_app_config()
        self.assertIn("app", cfg)
        self.assertIn("database", cfg)
        self.assertIn("retrieval", cfg)

    def test_stm_abbreviation_expansion(self):
        # In STM, CN must map to Control Flow Graph
        expanded = expand_query("explain CN", active_subject="STM Notes")
        self.assertIn("Control Flow Graph", expanded)

        # In STM, JF must map to Junction Flow
        expanded_jf = expand_query("what is JF", active_subject="STM Notes")
        self.assertIn("Junction", expanded_jf)

    def test_ml_abbreviation_expansion(self):
        # In ML, DA must map to Data Analysis
        expanded = expand_query("explain DA", active_subject="ML notes")
        self.assertIn("Data Analysis", expanded)

        # In ML, SVM must map to Support Vector Machine
        expanded_svm = expand_query("what is SVM", active_subject="ML notes")
        self.assertIn("Support Vector Machine", expanded_svm)

    def test_daa_abbreviation_expansion(self):
        # In DAA, DA must map to Design and Analysis of Algorithms
        expanded = expand_query("what is DA", active_subject="DAA Notes")
        self.assertIn("Design and Analysis of Algorithms", expanded)

    def test_acs_lab_expansion(self):
        expanded = expand_query("what is ACS Lab", active_subject="ACS Lab")
        self.assertIn("Advanced Communication Systems", expanded)

    def test_no_cross_contamination(self):
        # Asking CN in Java should not expand to Computer Networks
        expanded = expand_query("what is CN", active_subject="Java")
        self.assertNotIn("Computer Networks", expanded)

    def test_short_query_detection(self):
        self.assertTrue(is_short_query("CN"))
        self.assertTrue(is_short_query("what is DA"))
        self.assertFalse(is_short_query("explain the differences between linear and logistic regression in machine learning"))

    def test_ocr_garbage_detection(self):
        self.assertTrue(is_garbled_ocr("$$#@!!%^& 12 3"))
        self.assertFalse(is_garbled_ocr("A Control Flow Graph (CFG) is a graphical representation of control flow."))

    def test_subject_mismatch_relevance_gate(self):
        # Asking Data Analysis in Java with unrelated context
        fake_java_context = "Java is an object oriented language. Classes and objects encapsulate data and functions."
        is_rel, msg = is_context_relevant("Explain about Data Analysis", fake_java_context, "Java")
        self.assertFalse(is_rel)
        self.assertIn("Java Programming", msg)

    def test_subject_meta_query_relevance(self):
        # Asking about the subject itself should be relevant
        fake_acs_context = "Experiment 1: Time Division Multiplexing. Experiment 2: Optical Fiber Link Setup."
        is_rel, msg = is_context_relevant("What is ACS Lab", fake_acs_context, "ACS Lab")
        self.assertTrue(is_rel)

    def test_ollama_socket_check_never_crashes(self):
        # Must return boolean and never raise an unhandled exception
        online = is_ollama_online(timeout=0.2)
        self.assertIsInstance(online, bool)


if __name__ == "__main__":
    unittest.main()

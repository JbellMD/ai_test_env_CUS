import unittest
from training.code_evaluator import CodeEvaluator

class TestCodeEvaluator(unittest.TestCase):
    def test_evaluator_initialization(self):
        evaluator = CodeEvaluator()
        self.assertIsNotNone(evaluator)
        
    def test_code_evaluation(self):
        evaluator = CodeEvaluator()
        code = "def foo(): pass"
        results = evaluator.evaluate_code(code)
        self.assertIn('safety', results)
        self.assertIn('performance', results)
        self.assertIn('complexity', results)
        
    def test_safety_checks(self):
        evaluator = CodeEvaluator()
        unsafe_code = "import os"
        results = evaluator.evaluate_code(unsafe_code)
        self.assertLess(results['safety'], 1.0)

if __name__ == '__main__':
    unittest.main()
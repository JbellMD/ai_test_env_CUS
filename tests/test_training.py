import unittest
from training.self_writing_trainer import SelfWritingTrainer
from training.code_evaluator import CodeEvaluator
from training.evolutionary_trainer import EvolutionaryTrainer

class TestTrainingComponents(unittest.TestCase):
    def test_self_writing_trainer(self):
        trainer = SelfWritingTrainer(None)
        code = "def foo(): pass"
        metrics = trainer.evaluate_modification(code)
        self.assertIn('performance', metrics)
        
    def test_code_evaluator(self):
        evaluator = CodeEvaluator()
        code = "def foo(): pass"
        results = evaluator.evaluate_code(code)
        self.assertIn('safety', results)
        
    def test_evolutionary_trainer(self):
        trainer = EvolutionaryTrainer()
        population = [{'fitness': i} for i in range(10)]
        new_population = trainer.evolve_population(population)
        self.assertEqual(len(new_population), 10)

if __name__ == '__main__':
    unittest.main()
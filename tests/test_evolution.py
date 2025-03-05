import unittest
from training.evolutionary_trainer import EvolutionaryTrainer

class TestEvolutionaryTrainer(unittest.TestCase):
    def test_evolution_initialization(self):
        trainer = EvolutionaryTrainer()
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.population_size, 10)
        self.assertEqual(trainer.mutation_rate, 0.1)
        self.assertEqual(trainer.crossover_rate, 0.5)
        
    def test_population_evolution(self):
        trainer = EvolutionaryTrainer(population_size=5)
        population = [{'fitness': i} for i in range(5)]
        new_population = trainer.evolve_population(population)
        self.assertEqual(len(new_population), 5)
        
    def test_parent_selection(self):
        trainer = EvolutionaryTrainer()
        population = [{'fitness': i} for i in range(10)]
        parent = trainer._select_parent(population)
        self.assertIn(parent, population)
        
    def test_crossover(self):
        trainer = EvolutionaryTrainer(crossover_rate=1.0)
        parent1 = {'value': 1}
        parent2 = {'value': 2}
        child = trainer._crossover(parent1, parent2)
        self.assertEqual(child['value'], 2)
        
    def test_mutation(self):
        trainer = EvolutionaryTrainer(mutation_rate=1.0)
        individual = {'value': 5}
        mutated = trainer._mutate(individual)
        self.assertNotEqual(individual['value'], mutated['value'])
        
    def test_string_mutation(self):
        trainer = EvolutionaryTrainer()
        original = "hello"
        mutated = trainer._mutate_string(original)
        self.assertEqual(len(original), len(mutated))
        self.assertNotEqual(original, mutated)

if __name__ == '__main__':
    unittest.main()
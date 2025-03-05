import unittest
from models.self_modifying.self_modifying_model import SelfModifyingModel
import torch

class TestSelfModifyingModel(unittest.TestCase):
    def test_model_initialization(self):
        base_model = torch.nn.Linear(10, 1)
        model = SelfModifyingModel(base_model)
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        base_model = torch.nn.Linear(10, 1)
        model = SelfModifyingModel(base_model)
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 1))
        
    def test_architecture_modification(self):
        base_model = torch.nn.Linear(10, 1)
        model = SelfModifyingModel(base_model)
        modified_model = model.modify_architecture({})
        self.assertIsNotNone(modified_model)
        
    def test_training_loop(self):
        base_model = torch.nn.Linear(10, 1)
        training_loop = TrainingLoop(base_model)
        best_model = training_loop.run()
        self.assertIsNotNone(best_model)

import unittest
from models.gcn_model import GCNModel
import torch

class TestGCNModel(unittest.TestCase):
    def test_gcn_model_cpu(self):
        gcn_model = GCNModel(size="100k", num_layers=2, hidden_dim=16, epochs=1, device='cpu')
        gcn_model.train()
        self.assertEqual(gcn_model.device.type, 'cpu')

    def test_gcn_model_cuda(self):
        if torch.cuda.is_available():
            gcn_model = GCNModel(size="100k", num_layers=2, hidden_dim=16, epochs=1, device='cuda')
            gcn_model.train()
            self.assertEqual(gcn_model.device.type, 'cuda')
        else:
            print("CUDA not available; skipping CUDA test.")

    def test_gcn_model_mps(self):
        if torch.backends.mps.is_available():
            gcn_model = GCNModel(size="100k", num_layers=2, hidden_dim=16, epochs=1, device='mps')
            gcn_model.train()
            self.assertEqual(gcn_model.device.type, 'mps')
        else:
            print("MPS not available; skipping MPS test.")

if __name__ == '__main__':
    unittest.main()
import unittest

import torch

from models.fastvit import fastvit_sa12, fastvit_t12


class FastViTFeatureExtractionTest(unittest.TestCase):
    def test_sa12_fixed_c5_forks_after_attention_stage(self):
        model = fastvit_sa12(fork_feat=True, feature_version="fixed_c5")
        self.assertEqual(model.stage_output_indices, (0, 2, 4, 7))
        self.assertEqual(model.out_indices, [0, 2, 4, 7])

    def test_sa12_legacy_retains_historical_feature_indices(self):
        model = fastvit_sa12(fork_feat=True, feature_version="legacy")
        self.assertEqual(model.out_indices, [0, 2, 4, 6])

    def test_t12_indices_do_not_change(self):
        fixed = fastvit_t12(fork_feat=True, feature_version="fixed_c5")
        legacy = fastvit_t12(fork_feat=True, feature_version="legacy")
        self.assertEqual(fixed.out_indices, [0, 2, 4, 6])
        self.assertEqual(fixed.out_indices, legacy.out_indices)

    def test_sa12_attention_receives_gradient_from_fixed_c5(self):
        model = fastvit_sa12(fork_feat=True, feature_version="fixed_c5")
        model.eval()
        features = model(torch.randn(1, 3, 128, 160))
        self.assertEqual(len(features), 4)
        self.assertEqual(
            [feature.shape[-2:] for feature in features],
            [torch.Size([32, 40]), torch.Size([16, 20]), torch.Size([8, 10]), torch.Size([4, 5])],
        )
        features[-1].sum().backward()
        attention_gradients = [
            parameter.grad
            for parameter in model.network[7].parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(attention_gradients)
        self.assertTrue(
            any(torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                for gradient in attention_gradients)
        )


if __name__ == "__main__":
    unittest.main()

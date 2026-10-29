from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_hsi_uda import PatchNPZDataset, SimpleUNet, dice_loss, weighted_ce_loss


class ModelTests(unittest.TestCase):
    def test_odd_spatial_shape_and_supervised_gradients(self):
        torch.manual_seed(0)
        model = SimpleUNet(in_ch=32, num_classes=4, base=8, feat_dim=16)
        image = torch.randn(1, 32, 17, 19)
        logits, features = model(image, return_feat=True)
        self.assertEqual(tuple(logits.shape), (1, 4, 17, 19))
        self.assertEqual(tuple(features.shape), (1, 16, 17, 19))
        target = torch.randint(0, 4, (1, 17, 19))
        target[:, 0] = 255
        loss = weighted_ce_loss(logits, target) + dice_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all().item())
        self.assertGreater(model.enc1.block[0].weight.grad.abs().sum().item(), 0)

    def test_patch_normalization_and_unlabeled_target_interface(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "patch.npz"
            image = np.full((32, 8, 8), 5.0, dtype=np.float32)
            mask = np.ones((8, 8), dtype=np.uint8)
            mask[0, 0] = 99
            np.savez_compressed(path, image=image, mask=mask)
            dataset = PatchNPZDataset([path], np.ones(32), np.full(32, 2.0))
            item = dataset[0]
            np.testing.assert_allclose(item["img"].numpy(), 2.0)
            self.assertEqual(item["mask"][0, 0].item(), 255)
            unlabeled = PatchNPZDataset([path], np.ones(32), np.full(32, 2.0), labeled=False)[0]
            self.assertNotIn("mask", unlabeled)


if __name__ == "__main__":
    unittest.main()

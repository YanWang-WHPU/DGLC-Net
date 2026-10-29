from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from apply_local_consensus_router import apply_router, one_hot_prob, score_prediction


class RouterTests(unittest.TestCase):
    def test_binary_metrics_merge_subclasses_and_ignore_unknowns(self):
        target = np.array([[[1, 2, 3, 3, 0, 255]]], dtype=np.uint8)
        pred = np.array([[[1, 3, 1, 3, 1, 1]]], dtype=np.uint8)
        score = score_prediction(pred, target, ignore_zero_eval=True)
        self.assertEqual(score["binary_counts"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertAlmostEqual(score["oa"], 0.5)
        self.assertAlmostEqual(score["miou"], 1 / 3)
        self.assertAlmostEqual(score["mf1"], 0.5)

    def test_routing_fills_a_hole_and_removes_an_isolated_rice_pixel(self):
        pred = np.full((1, 9, 9), 3, dtype=np.uint8)
        pred[:, 2:7, 2:7] = 1
        pred[0, 4, 4] = 3
        pred[0, 0, 4] = 1
        prob = one_hot_prob(pred)
        prob[0, :, 4, 4] = [0.0, 0.1, 0.2, 0.7]
        expected = pred.copy()
        expected[0, 4, 4] = 2
        expected[0, 0, 4] = 3
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.npz"
            for target in (None, np.zeros_like(pred), np.ones_like(pred)):
                arrays = {"pred": pred, "prob": prob}
                if target is not None:
                    arrays["target"] = target
                np.savez_compressed(path, **arrays)
                routed, returned_target, debug = apply_router(path, 3, 1, 1)
                np.testing.assert_array_equal(routed, expected)
                self.assertEqual(debug["restored_pixels"], 1)
                self.assertEqual(debug["flipped_pixels"], 1)
                if target is None:
                    self.assertIsNone(returned_target)
                else:
                    np.testing.assert_array_equal(returned_target, target)

    def test_empty_valid_set_has_finite_metrics(self):
        target = np.full((1, 3, 3), 255, dtype=np.uint8)
        metrics = score_prediction(np.ones_like(target), target, True)
        self.assertEqual(metrics["oa"], 0.0)
        self.assertEqual(metrics["miou"], 0.0)


if __name__ == "__main__":
    unittest.main()

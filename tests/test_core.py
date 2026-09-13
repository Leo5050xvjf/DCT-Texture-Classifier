import unittest

import cv2
import numpy as np
import torch

from dct_texture.core import (
    aggregate_patch_probabilities,
    dct_basis,
    dct_2d_numpy,
    dct_2d_torch,
    extreme_windows,
)
from dct_texture.model import DCTTextureClassifier
from dct_texture.metrics import best_balanced_threshold, binary_metrics
from dct_texture.robust_models import make_robust_model
from dct_texture.map_model import MapGeneratorUNet
from dct_texture.teacher_map import teacher_maps_for_crops
from build_context_dataset import extract_context
from infer_map_generator import infer_tiled, tile_starts


class CoreTests(unittest.TestCase):
    def test_dct_matches_opencv_and_torch(self) -> None:
        rng = np.random.default_rng(1)
        patches = rng.normal(size=(5, 8, 8)).astype(np.float32)
        expected = np.stack([cv2.dct(patch) for patch in patches])
        actual_numpy = dct_2d_numpy(patches)
        basis = torch.from_numpy(dct_basis())
        actual_torch = dct_2d_torch(torch.from_numpy(patches), basis).numpy()
        np.testing.assert_allclose(actual_numpy, expected, atol=2e-6)
        np.testing.assert_allclose(actual_torch, expected, atol=2e-6)

    def test_extreme_windows(self) -> None:
        values = np.zeros((8, 9), dtype=np.float32)
        values[1:4, 2:5] = 5.0
        values[5:, 6:] = -2.0
        high, low, high_value, low_value = extreme_windows(values, window=3)
        self.assertEqual(high, (2, 1))
        self.assertEqual(low, (6, 5))
        self.assertEqual(high_value, 5.0)
        self.assertEqual(low_value, -2.0)

    def test_probability_aggregation_matches_brute_force(self) -> None:
        grid = np.arange(12, dtype=np.float32).reshape(3, 4)
        actual = aggregate_patch_probabilities(grid, (5, 6), patch_size=3)
        total = np.zeros((5, 6), dtype=np.float32)
        count = np.zeros((5, 6), dtype=np.float32)
        for y in range(3):
            for x in range(4):
                total[y : y + 3, x : x + 3] += grid[y, x]
                count[y : y + 3, x : x + 3] += 1
        np.testing.assert_allclose(actual, total / count, atol=1e-6)

    def test_model_shape(self) -> None:
        model = DCTTextureClassifier()
        self.assertEqual(model(torch.zeros(7, 1, 8, 8)).shape, (7,))

    def test_best_threshold(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = best_balanced_threshold(labels, scores)
        self.assertEqual(binary_metrics(labels, scores, threshold)["balanced_accuracy"], 1.0)

    def test_context_center_and_border_padding(self) -> None:
        gray = np.arange(40 * 50, dtype=np.uint16).reshape(40, 50).astype(np.uint8)
        interior = extract_context(gray, x=20, y=15)
        np.testing.assert_array_equal(interior[12:20, 12:20], gray[15:23, 20:28])
        border = extract_context(gray, x=0, y=0)
        self.assertEqual(border.shape, (32, 32))
        np.testing.assert_array_equal(border[12:20, 12:20], gray[:8, :8])

    def test_robust_model_shapes(self) -> None:
        batch = 3
        dct = torch.zeros(batch, 1, 8, 8)
        patch = torch.zeros(batch, 1, 8, 8)
        context = torch.zeros(batch, 1, 32, 32)
        sigma = torch.zeros(batch)
        self.assertEqual(make_robust_model("dct8")(dct).shape, (batch,))
        self.assertEqual(make_robust_model("dct8_conditional")(dct, sigma).shape, (batch,))
        self.assertEqual(make_robust_model("spatial8")(patch).shape, (batch,))
        self.assertEqual(make_robust_model("spatial32")(context).shape, (batch,))
        self.assertEqual(make_robust_model("hybrid32")(context, dct).shape, (batch,))

    def test_map_generator_shape(self) -> None:
        model = MapGeneratorUNet(base_channels=4)
        self.assertEqual(model(torch.zeros(2, 1, 32, 40)).shape, (2, 32, 40))

    def test_teacher_overlap_map_constant_probability(self) -> None:
        class ZeroTeacher(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(len(x), device=x.device)

        clean = torch.rand(2, 16, 17)
        basis = torch.from_numpy(dct_basis())
        mean = torch.zeros(1, 8, 8)
        std = torch.ones(1, 8, 8)
        actual = teacher_maps_for_crops(clean, ZeroTeacher(), basis, mean, std, patch_batch_size=100)
        np.testing.assert_allclose(actual.numpy(), 0.5, atol=1e-6)

    def test_tiled_generator_preserves_shape_and_constant(self) -> None:
        class ZeroMap(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), x.shape[-2], x.shape[-1]), device=x.device)

        gray = np.full((137, 211), 127, dtype=np.uint8)
        probability, noisy = infer_tiled(
            gray, ZeroMap(), torch.device("cpu"), tile_size=64, overlap=24,
            halo=12, batch_size=5, sigma=0.0, seed=1,
        )
        self.assertEqual(probability.shape, gray.shape)
        self.assertEqual(noisy.shape, gray.shape)
        np.testing.assert_allclose(probability, 0.5, atol=1e-6)
        self.assertEqual(tile_starts(100, 64, 40)[-1], 36)


if __name__ == "__main__":
    unittest.main()

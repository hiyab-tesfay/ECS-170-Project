"""Focused Stage 3 checks against the supplied dataset files."""

import unittest

import torch

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import make_model


class Stage3Tests(unittest.TestCase):
    def test_loaders_and_forward(self):
        for name, count, channels, classes in (
            ("MNIST", 60000, 1, 10), ("ORL", 360, 1, 40), ("CIFAR", 50000, 3, 10)
        ):
            with self.subTest(name=name):
                data = Dataset_Loader(name).load()
                self.assertEqual(len(data["train"]), count)
                self.assertEqual(data["train"].images.shape[1], channels)
                self.assertGreaterEqual(int(data["train"].labels.min()), 0)
                self.assertLess(int(data["train"].labels.max()), classes)
                model, _ = make_model(name)
                image, _ = data["train"][0]
                with torch.no_grad():
                    self.assertEqual(tuple(model(image[None]).shape), (1, classes))

    def test_training_step(self):
        for name, shape in (("MNIST", (1, 28, 28)),
                            ("ORL", (1, 112, 92)),
                            ("CIFAR", (3, 32, 32))):
            with self.subTest(name=name):
                model, _ = make_model(name)
                x = torch.rand(4, *shape)
                y = torch.tensor([0, 1, 2, 3])
                result = model.run(x, y, x, epochs=1, batch_size=2)
                self.assertEqual(len(result["pred_y"]), 4)
                self.assertEqual(len(result["training_loss"]), 1)


if __name__ == "__main__":
    unittest.main()

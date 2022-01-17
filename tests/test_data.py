import math
import os

import pytest
import torch

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
class TestDataset:
    def test_load_traindata(self):
        self.train_dataset = torch.load(f"{_PATH_DATA}/processed/train.pt")
        assert len(self.train_dataset) == math.ceil(25000 / 100)

    def test_load_testdata(self):
        self.test_dataset = torch.load(f"{_PATH_DATA}/processed/test.pt")
        assert len(self.test_dataset) == 5000

    def test_shape_traindata(self):
        self.train_dataset = torch.load(f"{_PATH_DATA}/processed/train.pt")
        for (images, labels) in self.train_dataset:
            assert (100, 1, 28, 28) == images.shape

    def test_labels_traindata(self):
        self.train_dataset = torch.load(f"{_PATH_DATA}/processed/train.pt")
        for (images, labels) in self.train_dataset:
            for label in labels:
                assert labels is not None

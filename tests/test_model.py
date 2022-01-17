import pytest
from tests import _PROJECT_ROOT
from src.models.model import MyAwesomeModel
import torch


class TestModel:
    @pytest.mark.parametrize("test_input,expected", [(torch.randn(1,1,2,28), ValueError), (torch.randn(64,1,28,2), ValueError), (torch.randn(100,3,28,28), ValueError)])
    def test_model_input_shape(self, test_input, expected):
        model = MyAwesomeModel(0.25, 0.5)
        model.train()
        with pytest.raises(expected, match='Expected each x sample to have shape 1,28,28'):
            model.forward(test_input)

    @pytest.mark.parametrize("test_input,expected", [(torch.randn(1,1,28,28), (1,10)), (torch.randn(64,1,28,28), (64,10)), (torch.randn(100,1,28,28), (100,10))])
    def test_model_output(self, test_input, expected):
        model = MyAwesomeModel(0.25, 0.5)
        output = model(test_input)
        assert output.shape == expected
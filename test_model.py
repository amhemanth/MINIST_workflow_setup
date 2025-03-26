import torch
import pytest
from model import MNISTModel, count_parameters
from train import train

def test_model_architecture():
    model = MNISTModel()
    
    # Test 1: Check if model accepts 28x28 input
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), "Output shape is incorrect"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")
    
    # Test 2: Check number of parameters
    num_params = count_parameters(model)
    assert num_params < 100000, f"Model has {num_params} parameters, should be less than 100000"
    
    # Test 3: Check output dimension
    assert output.shape[1] == 10, f"Model output dimension is {output.shape[1]}, should be 10"

def test_model_training():
    # Test 4: Check training accuracy
    accuracy = train()
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be above 80%"

if __name__ == "__main__":
    pytest.main([__file__]) 
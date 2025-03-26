import torch
import pytest
import numpy as np
from model import MNISTModel, count_parameters
from train import train
import os
import time

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
    print(f"Model has {num_params} parameters")
    assert num_params < 25000, f"Model has {num_params} parameters, should be less than 25000"
    
    # Test 3: Check output dimension
    assert output.shape[1] == 10, f"Model output dimension is {output.shape[1]}, should be 10"

def test_model_training():
    # Test 4: Check training accuracy
    accuracy = train()
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be above 95%"

def test_model_inference_speed():
    # Test 5: Check inference speed
    model = MNISTModel()
    model.eval()
    
    # Generate batch of test data
    test_batch = torch.randn(64, 1, 28, 28)
    
    # Warm-up
    with torch.no_grad():
        _ = model(test_batch)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):  # Run multiple times for more accurate measurement
            _ = model(test_batch)
    end_time = time.time()
    
    avg_time_per_batch = (end_time - start_time) / 10
    avg_time_per_image = avg_time_per_batch / 64
    
    print(f"Average inference time per image: {avg_time_per_image*1000:.2f} ms")
    
    # Ensure inference is fast enough for real-time applications
    assert avg_time_per_image < 0.01, f"Inference too slow: {avg_time_per_image*1000:.2f} ms per image, should be < 10ms"

def test_model_robustness():
    # Test 6: Check model robustness to noise
    model = MNISTModel()
    model.eval()
    
    # Load a small batch from MNIST for testing
    from torchvision import datasets, transforms
    test_dataset = datasets.MNIST('./data', train=False, download=True, 
                                 transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
    
    # Get a batch of test data
    data, targets = next(iter(test_loader))
    
    # Get predictions on clean data
    with torch.no_grad():
        clean_output = model(data)
        _, clean_preds = clean_output.max(1)
        clean_correct = clean_preds.eq(targets).sum().item()
    
    # Add noise to the data
    noise_levels = [0.1, 0.2, 0.3]
    noise_accuracies = []
    
    for noise_level in noise_levels:
        noisy_data = data + noise_level * torch.randn_like(data)
        noisy_data = torch.clamp(noisy_data, 0, 1)
        
        with torch.no_grad():
            noisy_output = model(noisy_data)
            _, noisy_preds = noisy_output.max(1)
            noisy_correct = noisy_preds.eq(targets).sum().item()
            
        noise_accuracies.append(noisy_correct / len(targets) * 100)
    
    # Check if accuracy doesn't drop too much with moderate noise
    print(f"Clean accuracy: {clean_correct/len(targets)*100:.2f}%")
    print(f"Accuracy with noise levels {noise_levels}: {noise_accuracies}")
    
    # Model should maintain at least 70% accuracy with 0.1 noise
    assert noise_accuracies[0] > 70, f"Model not robust to noise: {noise_accuracies[0]:.2f}% accuracy with noise level 0.1"

def test_model_save_load():
    # Test 7: Check if model can be saved and loaded correctly
    model = MNISTModel()
    
    # Generate random input
    test_input = torch.randn(1, 1, 28, 28)
    
    # Get output before saving
    model.eval()
    with torch.no_grad():
        output_before = model(test_input)
    
    # Save the model
    save_path = 'test_model.pth'
    torch.save(model.state_dict(), save_path)
    
    # Load the model
    loaded_model = MNISTModel()
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    
    # Get output after loading
    with torch.no_grad():
        output_after = loaded_model(test_input)
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
    
    # Check if outputs are identical
    assert torch.allclose(output_before, output_after), "Model outputs differ after save and load"

if __name__ == "__main__":
    pytest.main([__file__]) 
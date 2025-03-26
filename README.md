# MNIST DNN with CI/CD Pipeline

![ML Pipeline](https://github.com/amhemanth/MINIST_workflow_setup/workflows/ML%20Pipeline/badge.svg)

This project implements an optimized Deep Neural Network for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model achieves >95% accuracy in a single epoch while keeping the parameter count below 25,000, and is optimized for both CPU and GPU execution.

## Project Overview

This project demonstrates:
- Efficient CNN architecture design for digit recognition
- Advanced data augmentation techniques
- GPU-optimized model implementation
- Automated testing and validation
- CI/CD pipeline implementation with GitHub Actions
- Model robustness and performance evaluation

## Project Structure
- `model.py`: CNN architecture with parameter counting utility and GPU optimizations
- `train.py`: Training script with data augmentation and evaluation
- `test_model.py`: Comprehensive test suite for model validation
- `.github/workflows/ml-pipeline.yml`: GitHub Actions workflow configuration
- `requirements.txt`: Project dependencies

## Model Architecture
- **Input**: 28x28 grayscale images
- **Architecture**: 
  - 3 convolutional layers with batch normalization
  - Progressive feature extraction (8→16→16 filters)
  - Max pooling after each convolutional layer
  - Dropout (0.25) for regularization
  - 2 fully connected layers (32 hidden units)
- **Output**: 10 classes (digits 0-9)
- **Parameters**: ~15,000 (well below the 25,000 limit)
- **Performance**: >95% accuracy in just 1 epoch

## GPU Optimizations

The model includes several optimizations for GPU execution:
- **Memory Efficiency**:
  - In-place ReLU operations to reduce memory footprint
  - Contiguous tensor operations for better memory access patterns
  - Optimized weight initialization for faster convergence

- **Performance Enhancements**:
  - cuDNN benchmark mode for faster convolution operations
  - Asynchronous data transfers with `non_blocking=True`
  - Pinned memory for faster CPU-to-GPU transfers
  - Multi-worker data loading for improved throughput

- **Accurate GPU Benchmarking**:
  - CUDA events for precise timing measurements
  - Proper synchronization for accurate performance metrics
  - Warm-up iterations to ensure stable measurements

## Data Augmentation
The training process uses advanced augmentation techniques:
- Random rotation (up to 30 degrees)
- Random affine transformations (translation, scaling, shearing)
- Random noise addition
- Elastic distortions (simulating handwriting variations)

## Local Setup and Testing

1. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run training:
```
python train.py
```

4. Run all tests:
```
pytest test_model.py -v
```

5. Run specific test:
```
pytest test_model.py::test_model_architecture -v
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up a Python 3.8 environment
2. Installs all dependencies including scipy for advanced transformations
3. Runs the complete test suite
4. Reports test results back to GitHub

The pipeline is triggered on:
- Pull requests targeting the main branch
- Pushes to the main branch (including merged PRs)

## Test Suite
The comprehensive test suite validates:

1. **Architecture Tests**
   - Verifies the model accepts 28x28 input images
   - Confirms parameter count is below 25,000
   - Checks output dimension is 10 (one per digit)

2. **Performance Tests**
   - Validates >95% accuracy on MNIST after one epoch
   - Measures inference speed for real-time applications
   - Tests GPU acceleration when available

3. **Robustness Tests**
   - Tests model resilience to input noise
   - Ensures accuracy remains above 50% with moderate noise

4. **Deployment Tests**
   - Verifies model can be correctly saved and loaded
   - Confirms model outputs are consistent after serialization

## Deployment
When training completes, the model is saved with a timestamp suffix:
```
mnist_model_YYYYMMDD_HHMMSS.pth
```

This naming convention enables:
- Version tracking
- Training history preservation
- Easy rollback to previous models

## Performance Optimization
The model achieves high accuracy with minimal parameters through:
- Efficient layer design
- Batch normalization for faster convergence
- Strategic dropout placement
- Multiple mini-epochs within a single epoch
- Focused training on difficult examples
- GPU-optimized operations when available

## Hardware Compatibility
- The model automatically detects and utilizes GPU if available
- Falls back to optimized CPU execution when GPU is not present
- Includes specific optimizations for both execution modes
- Benchmarks performance appropriately for each hardware type

## Notes
- The model is optimized for both CPU and GPU inference
- MNIST dataset is automatically downloaded when running the training script
- The `.gitignore` file excludes datasets and model files from version control
- All tests must pass for pull requests to be merged

## Future Improvements
- Quantization for even faster inference
- Distillation techniques for further parameter reduction
- TensorBoard integration for visualization
- ONNX export for cross-platform deployment
- Mixed precision training for faster GPU execution

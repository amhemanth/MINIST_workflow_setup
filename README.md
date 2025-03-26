# MNIST DNN with CI/CD Pipeline

This project implements an optimized Deep Neural Network for MNIST digit classification with a CI/CD pipeline.

## Project Structure
- `model.py`: Contains the CNN architecture
- `train.py`: Training script
- `test_model.py`: Test cases for model validation
- `.github/workflows/ml-pipeline.yml`: GitHub Actions workflow
- `requirements.txt`: Dependencies for the project

## Model Architecture
- **Input**: 28x28 grayscale images
- **Architecture**: 
  - 2 convolutional layers with batch normalization
  - Dropout for regularization
  - 2 fully connected layers
- **Output**: 10 classes (digits 0-9)
- **Parameters**: Less than 25,000
- **Performance**: >95% accuracy in just 1 epoch

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

4. Run tests:
```
pytest test_model.py -v
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up a Python environment
2. Installs dependencies
3. Runs model tests
4. Validates model architecture and performance

## Test Cases
The pipeline validates that:
1. The model accepts 28x28 input images
2. The model has fewer than 25,000 parameters
3. The model outputs 10 classes (for digits 0-9)
4. The model achieves >95% accuracy after training for 1 epoch

## Deployment
When training completes, the model is saved with a timestamp suffix in the format:
```
mnist_model_YYYYMMDD_HHMMSS.pth
```

This allows tracking of when each model version was trained.

## Notes
- The model is designed to run on CPU
- MNIST dataset is automatically downloaded when running the training script
- The `.gitignore` file is configured to exclude the dataset and model files from version control

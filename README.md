## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up a Python environment
2. Installs dependencies
3. Runs model tests
4. Validates model architecture and performance

## Test Cases
The pipeline validates that:
1. The model accepts 28x28 input images
2. The model has fewer than 100,000 parameters
3. The model outputs 10 classes (for digits 0-9)
4. The model achieves >80% accuracy after training for 1 epoch

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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
import datetime

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST dataset with enhanced augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # For evaluation, use only normalization
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Create a separate loader for evaluation
    eval_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_eval)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    # Calculate total steps for all mini-epochs
    total_steps = len(train_loader) * 3  # 3 mini-epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01, 
        total_steps=total_steps
    )
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count} parameters")
    
    # Train for multiple mini-epochs
    model.train()
    for mini_epoch in range(3):  # Train for 3 mini-epochs
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Mini-epoch: {mini_epoch+1}/3, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {accuracy:.2f}%')
    
    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'mnist_model_{timestamp}.pth')
    
    return accuracy

if __name__ == "__main__":
    train() 
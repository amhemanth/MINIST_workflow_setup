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
    transform = transforms.Compose([
        transforms.RandomRotation(15),  # Increased rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine transforms
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)  # Smaller batch size
    
    # Initialize model, loss, and optimizer
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)  # Increased learning rate, added weight decay
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=1)  # Added scheduler
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count} parameters")
    
    # Train for one epoch with multiple passes over difficult examples
    model.train()
    correct = 0
    total = 0
    
    # First pass over all data
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
            print(f'Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    # Second pass focusing on difficult examples
    difficult_data = []
    difficult_targets = []
    
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            # Find misclassified examples
            incorrect_mask = ~predicted.eq(target)
            if incorrect_mask.sum() > 0:
                difficult_data.append(data[incorrect_mask])
                difficult_targets.append(target[incorrect_mask])
    
    # If we found difficult examples, train on them
    if difficult_data:
        print("Training on difficult examples...")
        model.train()
        
        # Combine all difficult examples
        difficult_data = torch.cat(difficult_data)
        difficult_targets = torch.cat(difficult_targets)
        
        # Create a new dataloader for difficult examples
        difficult_dataset = torch.utils.data.TensorDataset(difficult_data, difficult_targets)
        difficult_loader = torch.utils.data.DataLoader(difficult_dataset, batch_size=32, shuffle=True)
        
        # Train for 3 passes on difficult examples
        for epoch in range(3):
            for batch_idx, (data, target) in enumerate(difficult_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in train_loader:
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
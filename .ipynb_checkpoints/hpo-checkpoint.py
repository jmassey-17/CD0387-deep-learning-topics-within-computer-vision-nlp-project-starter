# Import dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from smdebug import modes
from smdebug.pytorch import get_hook
from torchvision import datasets, models
import logging
import os
import sys

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize debugging hook
hook = get_hook(create_if_not_exists=True)

def test(model, test_loader, criterion):
    '''
    Test function to evaluate model performance on the test dataset
    '''
    model.to(device)
    running_loss = 0
    correct = 0
    total = 0
    model.eval()

    # Disable gradient computation for testing
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    total_loss = running_loss / len(test_loader)
    accuracy = correct / total
    print(f"Testing Loss: {total_loss:.4f}, Testing Accuracy: {accuracy:.4f}")
    logger.info(f"Testing Loss: {total_loss:.4f}, Testing Accuracy: {accuracy:.4f}")
    if hook:
        hook.record_tensor_value("Testing Loss", total_loss)
        hook.record_tensor_value("Testing Accuracy", accuracy)


def train(model, train_loader, criterion, optimizer, epochs):
    '''
    Train function to optimize model on training dataset
    '''
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Save intermediate training metrics to hook for debugging
            if hook and hook.has_default_writer():
                hook.record_tensor_value("Training Loss", train_loss / len(train_loader))
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")
    return model


def net(num_classes):
    '''
    Initializes and returns a pretrained model with modified output layer
    '''
    model = models.resnet50(pretrained=True)
    
    # Freeze parameters in the pretrained model
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Update the final fully connected layer to match the number of classes
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), 
        nn.ReLU(),  
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    # Register model to debugging hook
    if hook:
        hook.register_module(model)
    return model


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Add this to handle truncated images

def create_data_loaders(data_path, batch_size):
    '''
    Create and return data loaders for training, testing, and validation
    '''
    # Define data paths
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    valid_path = os.path.join(data_path, 'valid')
    
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # Define datasets
    trainset = datasets.ImageFolder(
        root=train_path,
        transform=transform_train
    )
    testset = datasets.ImageFolder(
        root=test_path,
        transform=transform_test
    )
    validset = datasets.ImageFolder(
        root=valid_path,
        transform=transform_test
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, valid_loader



def main(args):
    '''
    Main function to initialize model, train, test, and save it
    '''
    logger.info("Starting training job")
    
    # Initialize model
    model = net(num_classes=133)
    
    # Define loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load data
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_path, args.batch_size)
    
    # Train the model
    model = train(model, train_loader, loss_criterion, optimizer, args.epochs)
    
    # Test the model
    test(model, test_loader, loss_criterion)
    
    # Save the model
    save_path = os.path.join(args.model_dir, 'model.pth')
    logger.info(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    logger.info("Model saved successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Specify hyperparameters for training
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training")
    parser.add_argument("--epochs", type=int, default=5, metavar="E", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate")
    parser.add_argument("--data-path", type=str, default=os.getenv("SM_CHANNEL_TRAINING"), metavar="DD", help="Data directory")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"], help="model directory")
    
    args = parser.parse_args()
    
    main(args)

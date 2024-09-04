import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

# Function to handle command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Train an image classifier model.")
    
    # Required positional argument for the dataset directory
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset.')
    
    # Optional arguments with default values
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the model checkpoint.')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'], help='Model architecture (e.g., vgg16, densenet121).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available.')
    
    return parser.parse_args()

# Function to load the data and apply transformations
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
    }

    return dataloaders, image_datasets

# Function to build the model based on the selected architecture
def build_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088  # Input size for VGG16
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features  # Input size for DenseNet121
    else:
        raise ValueError('Architecture not recognized. Use vgg16 or densenet121.')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                validation_loss, accuracy = validate_model(model, dataloaders['valid'], criterion, device)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                
                running_loss = 0
                model.train()

# Function to validate the model
def validate_model(model, validloader, criterion, device):
    model.eval()
    validation_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            validation_loss += batch_loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return validation_loss, accuracy

# Function to save the trained model checkpoint
def save_checkpoint(model, image_datasets, save_dir, arch, hidden_units, learning_rate, epochs):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_dir)

# Main function to tie everything together
def main():
    args = get_input_args()
    dataloaders, image_datasets = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    save_checkpoint(model, image_datasets, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs)

if __name__ == '__main__':
    main()

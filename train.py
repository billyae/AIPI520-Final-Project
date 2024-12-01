import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from src.data.data_loader import ReflexDataset
from src.models.deep_learning import MultiLabelXrayClassifier
from src.utils.evaluation import evaluate_model, plot_metrics
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description='Train RefleX classification model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to directory containing images')
    parser.add_argument('--labels_file', type=str, required=True,
                      help='Path to labels CSV file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create datasets
    dataset = ReflexDataset(
        root_dir=args.data_dir,
        labels_file=args.labels_file,
        transform=transform
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model
    model = MultiLabelXrayClassifier().to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(dataloader):
            inputs = data['image'].to(device)
            labels = data['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    # Save model
    torch.save(model.state_dict(), 'reflex_multilabel_model.pth')
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, device)
    print(f"Final metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")

if __name__ == '__main__':
    main()

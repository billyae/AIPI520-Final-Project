import torch.nn as nn

class MultiLabelXrayClassifier(nn.Module):
    """Custom CNN for multi-label X-ray diffraction image classification.
    
    Architecture designed for 1024x1024 input images and 7 binary labels.
    """
    def __init__(self):
        super(MultiLabelXrayClassifier, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate size after convolutions
        self.fc_input_size = 512 * (1024 // (2**5)) * (1024 // (2**5))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 7),  # 7 output nodes for 7 labels
            nn.Sigmoid()  # Sigmoid for multi-label
        )

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 1024, 1024)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 7)
        """
        x = self.features(x)
        x = x.view(-1, self.fc_input_size)
        x = self.classifier(x)
        return x
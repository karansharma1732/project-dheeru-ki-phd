import torch
import torch.nn as nn
import torchvision.models as models

class MedicalImageCNN(nn.Module):
    """Custom CNN for medical image classification"""
    
    def __init__(self, num_classes=2):
        super(MedicalImageCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(architecture="resnet18", num_classes=2, pretrained=True, freeze_backbone=False):
    """Get model architecture"""
    
    if architecture == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif architecture == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    elif architecture == "custom_cnn":
        model = MedicalImageCNN(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Freeze backbone if requested
    if freeze_backbone and pretrained:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    return model
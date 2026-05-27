import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy

class FederatedClient:
    """Federated Learning Client"""
    
    def __init__(self, client_id, model, device, learning_rate=0.001, local_epochs=5):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, train_loader):
        """Train local model"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            epoch_losses.append(avg_loss)
        
        return {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'num_samples': total,
            'loss': sum(epoch_losses) / len(epoch_losses),
            'accuracy': accuracy
        }
    
    def evaluate(self, val_loader):
        """Evaluate local model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def set_model_parameters(self, state_dict):
        """Update local model with global parameters"""
        self.model.load_state_dict(state_dict)
    
    def get_model_parameters(self):
        """Get local model parameters"""
        return copy.deepcopy(self.model.state_dict())
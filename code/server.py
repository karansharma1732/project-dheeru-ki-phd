import torch
import numpy as np
from tqdm import tqdm
import copy
from .client import FederatedClient
from .aggregation import aggregate_models

class FederatedServer:
    """Federated Learning Server"""
    
    def __init__(self, model, device, num_clients, num_rounds, 
                 aggregation_method="fedavg", client_fraction=1.0):
        self.global_model = model.to(device)
        self.device = device
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.client_fraction = client_fraction
        
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'client_stats': []
        }
    
    def train(self, client_loaders, learning_rate=0.001, local_epochs=5):
        """Run federated learning training"""
        
        print(f"Starting Federated Learning with {self.num_clients} clients")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Aggregation method: {self.aggregation_method}")
        
        # Initialize clients
        clients = {}
        for client_id in client_loaders.keys():
            client_model = copy.deepcopy(self.global_model)
            clients[client_id] = FederatedClient(
                client_id, client_model, self.device,
                learning_rate, local_epochs
            )
        
        # Training loop
        for round_num in range(self.num_rounds):
            print(f"\n=== Round {round_num + 1}/{self.num_rounds} ===")
            
            # Select clients for this round
            num_selected = max(1, int(self.num_clients * self.client_fraction))
            selected_clients = np.random.choice(
                list(clients.keys()), num_selected, replace=False
            )
            
            # Client training
            client_updates = []
            client_weights = []
            client_stats = {}
            
            for client_id in tqdm(selected_clients, desc="Client Training"):
                # Set global model parameters
                clients[client_id].set_model_parameters(
                    self.global_model.state_dict()
                )
                
                # Train locally
                train_result = clients[client_id].train(
                    client_loaders[client_id]['train']
                )
                
                # Evaluate locally
                val_result = clients[client_id].evaluate(
                    client_loaders[client_id]['val']
                )
                
                # Collect updates
                client_updates.append(train_result['model_state'])
                client_weights.append(train_result['num_samples'])
                
                client_stats[client_id] = {
                    'train_loss': train_result['loss'],
                    'train_accuracy': train_result['accuracy'],
                    'val_loss': val_result['loss'],
                    'val_accuracy': val_result['accuracy']
                }
            
            # Aggregate models
            global_state = aggregate_models(
                self.aggregation_method,
                client_updates,
                client_weights
            )
            
            # Update global model
            self.global_model.load_state_dict(global_state)
            
            # Calculate and store metrics
            avg_train_loss = np.mean([s['train_loss'] for s in client_stats.values()])
            avg_train_acc = np.mean([s['train_accuracy'] for s in client_stats.values()])
            avg_val_loss = np.mean([s['val_loss'] for s in client_stats.values()])
            avg_val_acc = np.mean([s['val_accuracy'] for s in client_stats.values()])
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_accuracy'].append(avg_train_acc)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_accuracy'].append(avg_val_acc)
            self.history['client_stats'].append(client_stats)
            
            print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Train Acc: {avg_train_acc:.2f}%")
            print(f"Avg Val Loss: {avg_val_loss:.4f} | Avg Val Acc: {avg_val_acc:.2f}%")
        
        return self.history
    
    def evaluate_global_model(self, test_loader):
        """Evaluate global model on test data"""
        self.global_model.eval()
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
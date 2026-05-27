import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DataPartitioner:
    """Partition data for federated learning (IID and Non-IID scenarios)"""
    
    def __init__(self, image_paths, labels, num_clients, partition_strategy="iid", 
                 alpha=0.5, train_split=0.8, val_split=0.1):
        self.image_paths = image_paths
        self.labels = labels
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.alpha = alpha
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split
    
    def partition(self):
        """Partition data according to strategy"""
        # First split into train+val and test
        train_val_idx, test_idx = train_test_split(
            np.arange(len(self.image_paths)),
            test_size=self.test_split,
            stratify=self.labels,
            random_state=42
        )
        
        # Store test data
        test_data = {
            'paths': self.image_paths[test_idx],
            'labels': self.labels[test_idx]
        }
        
        # Partition train+val data among clients
        if self.partition_strategy == "iid":
            client_data = self._partition_iid(train_val_idx)
        elif self.partition_strategy == "non_iid_label_skew":
            client_data = self._partition_non_iid_label_skew(train_val_idx)
        elif self.partition_strategy == "non_iid_quantity_skew":
            client_data = self._partition_non_iid_quantity_skew(train_val_idx)
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")
        
        # Split each client's data into train and val
        client_data_split = {}
        for client_id, indices in client_data.items():
            client_labels = self.labels[indices]
            
            val_size = self.val_split / (self.train_split + self.val_split)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                stratify=client_labels,
                random_state=42
            )
            
            client_data_split[client_id] = (
                self.image_paths[train_idx],
                self.labels[train_idx],
                self.image_paths[val_idx],
                self.labels[val_idx]
            )
        
        return client_data_split, test_data
    
    def _partition_iid(self, indices):
        """IID partitioning - randomly distribute data equally"""
        np.random.shuffle(indices)
        client_data = {}
        
        samples_per_client = len(indices) // self.num_clients
        
        for i in range(self.num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < self.num_clients - 1 else len(indices)
            client_data[f"client_{i}"] = indices[start:end]
        
        return client_data
    
    def _partition_non_iid_label_skew(self, indices):
        """Non-IID with label distribution skew using Dirichlet distribution"""
        labels = self.labels[indices]
        num_classes = len(np.unique(labels))
        
        client_data = {f"client_{i}": [] for i in range(self.num_clients)}
        
        # For each class, use Dirichlet distribution to assign to clients
        for class_id in range(num_classes):
            class_indices = indices[labels == class_id]
            np.random.shuffle(class_indices)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (proportions * len(class_indices)).astype(int)
            
            # Adjust for rounding errors
            proportions[-1] = len(class_indices) - proportions[:-1].sum()
            
            # Assign data to clients
            start = 0
            for client_id, proportion in enumerate(proportions):
                end = start + proportion
                client_data[f"client_{client_id}"].extend(class_indices[start:end])
                start = end
        
        # Convert to numpy arrays
        for client_id in client_data:
            client_data[client_id] = np.array(client_data[client_id])
        
        return client_data
    
    def _partition_non_iid_quantity_skew(self, indices):
        """Non-IID with quantity skew - different clients have different amounts of data"""
        # Use power law distribution for quantity skew
        proportions = np.random.power(2, self.num_clients)
        proportions = proportions / proportions.sum()
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()
        
        np.random.shuffle(indices)
        client_data = {}
        
        start = 0
        for i, proportion in enumerate(proportions):
            end = start + proportion
            client_data[f"client_{i}"] = indices[start:end]
            start = end
        
        return client_data
    
    def analyze_partition(self, client_data):
        """Analyze and visualize data distribution across clients"""
        stats = {}
        
        for client_id, (train_paths, train_labels, _, _) in client_data.items():
            unique, counts = np.unique(train_labels, return_counts=True)
            stats[client_id] = {
                'total_samples': len(train_labels),
                'class_distribution': dict(zip(unique, counts))
            }
        
        return stats
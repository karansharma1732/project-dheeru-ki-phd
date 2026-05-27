import torch
import copy
from collections import OrderedDict

class FederatedAggregator:
    """Federated Learning Aggregation Strategies"""
    
    @staticmethod
    def fedavg(client_updates, client_weights=None):
        """
        FedAvg: Federated Averaging
        Weighted average of client models based on number of samples
        """
        if client_weights is None:
            # Equal weights
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            # Normalize weights
            total_samples = sum(client_weights)
            client_weights = [w / total_samples for w in client_weights]
        
        # Initialize global model with first client's parameters
        global_state = OrderedDict()
        for key in client_updates[0].keys():
            global_state[key] = torch.zeros_like(client_updates[0][key])
        
        # Weighted aggregation
        for client_state, weight in zip(client_updates, client_weights):
            for key in global_state.keys():
                global_state[key] += client_state[key] * weight
        
        return global_state
    
    @staticmethod
    def weighted_average(client_updates, client_weights):
        """Simple weighted average"""
        return FederatedAggregator.fedavg(client_updates, client_weights)
    
    @staticmethod
    def fedprox(client_updates, global_model, mu=0.01):
        """
        FedProx: Adds proximal term to handle system heterogeneity
        """
        # Similar to FedAvg but with proximal regularization
        # This is typically handled during client training
        return FederatedAggregator.fedavg(client_updates)

def aggregate_models(aggregation_method, client_updates, client_weights=None):
    """Main aggregation function"""
    aggregator = FederatedAggregator()
    
    if aggregation_method == "fedavg":
        return aggregator.fedavg(client_updates, client_weights)
    elif aggregation_method == "weighted_avg":
        return aggregator.weighted_average(client_updates, client_weights)
    elif aggregation_method == "fedprox":
        return aggregator.fedprox(client_updates, None)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
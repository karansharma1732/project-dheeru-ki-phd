import sys
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_medical_data, MedicalImageDataset, get_transforms
from data.data_partitioner import DataPartitioner
from models.cnn_models import get_model
from federated.server import FederatedServer
from data.data_loader import create_client_dataloaders

def load_config(config_path="config/config.yaml"):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # Device configuration
    device = torch.device(config['experiment']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading medical image data...")
    image_paths, labels = load_medical_data(
        config['data']['data_dir'],
        config['data']['dataset_type']
    )
    print(f"Loaded {len(image_paths)} images with {len(np.unique(labels))} classes")
    
    # Partition data
    print(f"Partitioning data using {config['federated']['partition_strategy']}...")
    partitioner = DataPartitioner(
        image_paths, labels,
        num_clients=config['federated']['num_clients'],
        partition_strategy=config['federated']['partition_strategy'],
        alpha=config['federated']['alpha'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    client_data, test_data = partitioner.partition()
    
    # Analyze partition
    partition_stats = partitioner.analyze_partition(client_data)
    print("\nData Partition Statistics:")
    for client_id, stats in partition_stats.items():
        print(f"{client_id}: {stats['total_samples']} samples, "
              f"distribution: {stats['class_distribution']}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    client_loaders = create_client_dataloaders(
        client_data,
        config['federated']['batch_size'],
        config['data']['image_size']
    )
    
    # Create test loader
    test_dataset = MedicalImageDataset(
        test_data['paths'], test_data['labels'],
        transform=get_transforms(config['data']['image_size'], is_train=False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['federated']['batch_size'],
        shuffle=False, num_workers=4
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(
        architecture=config['model']['architecture'],
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Initialize federated server
    print("\nInitializing federated server...")
    server = FederatedServer(
        model=model,
        device=device,
        num_clients=config['federated']['num_clients'],
        num_rounds=config['federated']['num_rounds'],
        aggregation_method=config['federated']['aggregation_method'],
        client_fraction=config['federated']['client_fraction']
    )
    
    # Train
    print("\nStarting federated training...")
    history = server.train(
        client_loaders,
        learning_rate=config['federated']['learning_rate'],
        local_epochs=config['federated']['local_epochs']
    )
    
    # Evaluate on test set
    print("\nEvaluating global model on test set...")
    test_results = server.evaluate_global_model(test_loader)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    # Save results
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'global_model.pth')
    torch.save(server.global_model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save history
    history_path = os.path.join(output_dir, 'training_history.pt')
    torch.save(history, history_path)
    print(f"Training history saved to: {history_path}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
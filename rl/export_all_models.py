#!/usr/bin/env python3
"""
Export all trained models to ONNX for web deployment.

Usage:
    uv run export_all_models.py
"""

import os
import torch
import numpy as np

from dqn_agent import DQN
from dqn_cnn_agent import DQN_CNN


def export_model(model, model_name: str, output_dir: str, is_cnn: bool = False):
    """Export a model to ONNX format."""
    model.eval()
    
    if is_cnn:
        dummy_input = torch.randn(1, 16, 4, 4, dtype=torch.float32)
    else:
        dummy_input = torch.randn(1, 16, dtype=torch.float32)
    
    output_path = os.path.join(output_dir, f'{model_name}.onnx')
    
    print(f"Exporting {model_name}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['q_values'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        }
    )
    
    size_kb = os.path.getsize(output_path) / 1024
    data_file = output_path + '.data'
    if os.path.exists(data_file):
        size_kb += os.path.getsize(data_file) / 1024
    
    print(f"  Exported: {output_path} ({size_kb:.1f} KB)")
    return output_path


def main():
    checkpoint_dir = 'checkpoints'
    output_dir = '../web/public/models'
    os.makedirs(output_dir, exist_ok=True)
    
    models_to_export = [
        # (checkpoint_name, output_name, is_cnn, model_class)
        ('best_model.pt', 'dqn_10k', False, DQN),
        ('dqn_basic_100k_best.pt', 'dqn_100k', False, DQN),
        ('dqn_shaped_best.pt', 'dqn_shaped', False, DQN),
        ('dqn_cnn_best.pt', 'dqn_cnn', True, DQN_CNN),
    ]
    
    exported = []
    
    for checkpoint_name, output_name, is_cnn, model_class in models_to_export:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {output_name}: {checkpoint_path} not found")
            continue
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model = model_class()
            model.load_state_dict(checkpoint['policy_net'])
            
            export_model(model, output_name, output_dir, is_cnn)
            exported.append(output_name)
            
        except Exception as e:
            print(f"Error exporting {output_name}: {e}")
    
    print(f"\nExported {len(exported)} models: {', '.join(exported)}")
    
    # Generate model manifest for web
    manifest = {
        'models': [
            {
                'id': 'dqn_10k',
                'name': 'Basic DQN (10k)',
                'description': 'Basic DQN trained for 10,000 episodes',
                'file': 'dqn_10k.onnx',
                'type': 'mlp'
            },
            {
                'id': 'dqn_100k', 
                'name': 'Basic DQN (100k)',
                'description': 'Basic DQN trained for 100,000 episodes',
                'file': 'dqn_100k.onnx',
                'type': 'mlp'
            },
            {
                'id': 'dqn_shaped',
                'name': 'Reward Shaped DQN',
                'description': 'DQN with advanced reward shaping (50k episodes)',
                'file': 'dqn_shaped.onnx',
                'type': 'mlp'
            },
            {
                'id': 'dqn_cnn',
                'name': 'CNN DQN',
                'description': 'CNN architecture with reward shaping (50k episodes)',
                'file': 'dqn_cnn.onnx',
                'type': 'cnn'
            }
        ]
    }
    
    import json
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated manifest: {manifest_path}")


if __name__ == "__main__":
    main()

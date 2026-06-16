#!/usr/bin/env python3
"""
Export all trained models to ONNX for web deployment.

Handles both old (pre-Dueling) and new (Dueling) architectures by
auto-detecting from checkpoint state dict keys.

Usage:
    uv run export_all_models.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dqn_agent import DuelingDQN
from dqn_cnn_agent import DQN_CNN


# Old architectures for backward compatibility with existing checkpoints
class OldDQN(nn.Module):
    """Original MLP DQN: 16 -> 256 -> 256 -> 128 -> 4"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class OldDQN_CNN(nn.Module):
    """Original CNN DQN: 2 conv layers + FC"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, padding=0)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def auto_detect_model(checkpoint_path: str, is_cnn: bool):
    """Auto-detect architecture from checkpoint state dict and return loaded model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['policy_net']

    if is_cnn:
        if 'bn1.weight' in state_dict:
            model = DQN_CNN()
        else:
            model = OldDQN_CNN()
    else:
        if 'shared1.weight' in state_dict:
            model = DuelingDQN()
        else:
            model = OldDQN()

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_model(model, model_name: str, output_dir: str, is_cnn: bool = False):
    """Export a model to ONNX format."""
    model.eval()

    if is_cnn:
        dummy_input = torch.randn(1, 16, 4, 4, dtype=torch.float32)
    else:
        dummy_input = torch.randn(1, 16, dtype=torch.float32)

    output_path = os.path.join(output_dir, f'{model_name}.onnx')

    print(f"Exporting {model_name} ({model.__class__.__name__})...")

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

    # Models to export: (checkpoint, output_name, is_cnn)
    # Includes both old and new checkpoints
    models_to_export = [
        # Old models (backward compat)
        ('dqn_shaped_best.pt', 'dqn_shaped', False),
        ('dqn_cnn_best.pt', 'dqn_cnn', True),
        # New Dueling models
        ('dueling_shaped_best.pt', 'dqn_shaped', False),
        ('dueling_cnn_best.pt', 'dqn_cnn', True),
    ]

    exported = []

    # Export new models first (override old ones), fall back to old
    seen = set()
    for checkpoint_name, output_name, is_cnn in reversed(models_to_export):
        if output_name in seen:
            continue

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            continue

        try:
            model = auto_detect_model(checkpoint_path, is_cnn)
            export_model(model, output_name, output_dir, is_cnn)
            exported.append(output_name)
            seen.add(output_name)
        except Exception as e:
            print(f"Error exporting {output_name} from {checkpoint_name}: {e}")

    print(f"\nExported {len(exported)} models: {', '.join(exported)}")

    # Generate model manifest for web
    manifest = {
        'models': [
            {
                'id': 'dqn_shaped',
                'name': 'Dueling DQN (Shaped)',
                'description': 'Dueling DQN with corner/monotonicity reward shaping',
                'file': 'dqn_shaped.onnx',
                'type': 'mlp'
            },
            {
                'id': 'dqn_cnn',
                'name': 'Dueling CNN DQN',
                'description': 'Dueling CNN with one-hot encoding and reward shaping',
                'file': 'dqn_cnn.onnx',
                'type': 'cnn'
            }
        ]
    }

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated manifest: {manifest_path}")


if __name__ == "__main__":
    main()

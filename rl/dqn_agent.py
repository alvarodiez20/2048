"""
Deep Q-Network (DQN) Agent for 2048

This module implements a Dueling DQN agent with:
- Dueling architecture (separate value and advantage streams)
- Experience replay buffer
- Target network with soft updates
- Epsilon-greedy exploration with slower decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import math

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 500000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for 2048.
    
    Uses separate value and advantage streams for better action evaluation.
    
    Architecture:
    - Input: 16 normalized tile values (log2/17)
    - Shared: 512 -> 512
    - Value stream: 256 -> 1
    - Advantage stream: 256 -> 4
    - Output: 4 Q-values via Q = V + (A - mean(A))
    """
    
    def __init__(self, input_size: int = 16, hidden_size: int = 512, output_size: int = 4):
        super().__init__()
        
        # Shared feature layers
        self.shared1 = nn.Linear(input_size, hidden_size)
        self.shared2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream
        self.value1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value2 = nn.Linear(hidden_size // 2, 1)
        
        # Advantage stream
        self.advantage1 = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))
        
        # Value stream
        v = F.relu(self.value1(x))
        v = self.value2(v)
        
        # Advantage stream
        a = F.relu(self.advantage1(x))
        a = self.advantage2(a)
        
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q


# Backward-compatible alias so export_onnx.py and old checkpoints still work
DQN = DuelingDQN


class DQNAgent:
    """
    DQN Agent for playing 2048.
    
    Uses:
    - Dueling DQN architecture
    - Experience replay (500k buffer)
    - Target network (soft updates)
    - Slower epsilon-greedy exploration (500k decay)
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500000,
        batch_size: int = 256,
        buffer_size: int = 500000,
        target_update_freq: int = 1000,
        device: str = None
    ):
        # Device selection: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🚀 Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("🚀 Using NVIDIA GPU (CUDA)")
            else:
                self.device = torch.device("cpu")
                print("⚠️  Using CPU (slow)")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = DuelingDQN().to(self.device)
        self.target_net = DuelingDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Counters
        self.steps = 0
        self.updates = 0
    
    @property
    def epsilon(self) -> float:
        """Current epsilon for exploration."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-1.0 * self.steps / self.epsilon_decay)
    
    def select_action(self, state: np.ndarray, legal_actions: List[bool], 
                      training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Normalized board state (16 values)
            legal_actions: Boolean mask of legal actions [Up, Down, Left, Right]
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Selected action index (0-3)
        """
        valid_actions = [i for i, legal in enumerate(legal_actions) if legal]
        
        if not valid_actions:
            return 0  # Shouldn't happen, but just in case
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze()
            
            # Mask invalid actions with very negative values
            for i in range(4):
                if i not in valid_actions:
                    q_values[i] = float('-inf')
            
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Store an experience in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.steps += 1
    
    def update(self) -> Optional[float]:
        """
        Perform one step of training.
        
        Returns:
            Loss value if update was performed, None otherwise
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        experiences = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            # Use policy net to select actions
            next_actions = self.policy_net(next_states).argmax(1)
            # Use target net to evaluate
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.updates += 1
        
        # Update target network
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """Save the agent's state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'updates': self.updates
        }, path)
    
    def load(self, path: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']


if __name__ == "__main__":
    # Quick test
    agent = DQNAgent()
    print(f"Device: {agent.device}")
    print(f"Policy network: {agent.policy_net}")
    
    # Test action selection
    state = np.random.rand(16).astype(np.float32)
    legal = [True, True, True, True]
    action = agent.select_action(state, legal)
    print(f"Selected action: {action}")

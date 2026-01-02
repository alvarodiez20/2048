"""
DQN Agent with CNN Architecture for 2048

Uses a convolutional neural network that treats the board as a 4x4 grid
with one-hot encoded channels for each tile value.
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
    
    def __init__(self, capacity: int = 100000):
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


def encode_board_onehot(board: np.ndarray) -> np.ndarray:
    """
    Encode board as one-hot tensor: (16 channels, 4, 4)
    Channel 0 = empty, Channel i = tile value 2^i (i=1..15)
    """
    result = np.zeros((16, 4, 4), dtype=np.float32)
    for i in range(16):
        row, col = i // 4, i % 4
        val = board[i]
        if val == 0:
            result[0, row, col] = 1.0
        else:
            channel = int(np.log2(val))
            if channel < 16:
                result[channel, row, col] = 1.0
    return result


class DQN_CNN(nn.Module):
    """
    CNN-based Deep Q-Network for 2048.
    
    Architecture:
    - Input: 16 channels x 4x4 (one-hot encoded board)
    - Conv layers extract spatial features
    - Fully connected layers output Q-values
    """
    
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(16, 128, kernel_size=2, padding=0)  # 4x4 -> 3x3
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, padding=0)  # 3x3 -> 2x2
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, padding=0)  # 2x2 -> 1x1
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 4)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 16, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten to (batch, 128)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent_CNN:
    """
    DQN Agent with CNN architecture and improved reward shaping.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 100000,
        batch_size: int = 256,
        buffer_size: int = 100000,
        target_update_freq: int = 1000,
        use_reward_shaping: bool = True,
        device: str = None
    ):
        # Device selection: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🚀 Using Apple Silicon GPU (MPS) for CNN")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("🚀 Using NVIDIA GPU (CUDA)")
            else:
                self.device = torch.device("cpu")
                print("⚠️  Using CPU (slow)")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = DQN_CNN().to(self.device)
        self.target_net = DQN_CNN().to(self.device)
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
        self.use_reward_shaping = use_reward_shaping
        
        # Counters
        self.steps = 0
        self.updates = 0
    
    @property
    def epsilon(self) -> float:
        """Current epsilon for exploration."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-1.0 * self.steps / self.epsilon_decay)
    
    def encode_state(self, board: np.ndarray) -> np.ndarray:
        """Encode board state for CNN input."""
        return encode_board_onehot(board)
    
    def select_action(self, state: np.ndarray, legal_actions: List[bool], 
                      training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        valid_actions = [i for i, legal in enumerate(legal_actions) if legal]
        
        if not valid_actions:
            return 0
        
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze()
            
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
        """Perform one step of training."""
        if len(self.buffer) < self.batch_size:
            return None
        
        experiences = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.updates += 1
        
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


def shape_reward(game, result: dict, action: int) -> float:
    """
    Advanced reward shaping for better learning.
    
    Bonuses:
    - Points from merges (base reward)
    - Corner bonus: max tile in corner
    - Monotonicity bonus: tiles arranged in decreasing order
    - Empty cells bonus: more empty cells is good
    - Penalty for invalid moves
    """
    reward = float(result['reward'])
    
    if not result['changed']:
        return -10.0  # Penalty for invalid move
    
    board = game.board if hasattr(game, 'board') else list(game._game.board()) if hasattr(game, '_game') else game.board()
    if not isinstance(board, list):
        board = list(board)
    
    max_tile = max(board)
    empty_count = board.count(0)
    
    # Corner bonus: reward keeping max tile in corner
    corners = [0, 3, 12, 15]
    if any(board[c] == max_tile for c in corners):
        reward += max_tile * 0.1
    
    # Edge bonus: second highest should be on edge
    if max_tile > 2:
        edges = [1, 2, 4, 7, 8, 11, 13, 14] + corners
        sorted_tiles = sorted([t for t in board if t > 0], reverse=True)
        if len(sorted_tiles) > 1:
            second_highest = sorted_tiles[1]
            if any(board[e] == second_highest for e in edges):
                reward += second_highest * 0.05
    
    # Empty cells bonus
    reward += empty_count * 2
    
    # Monotonicity in snake pattern (top-left starting)
    # Row 0: left to right, Row 1: right to left, etc.
    snake = [board[0], board[1], board[2], board[3],
             board[7], board[6], board[5], board[4],
             board[8], board[9], board[10], board[11],
             board[15], board[14], board[13], board[12]]
    
    monotonic_score = 0
    for i in range(len(snake) - 1):
        if snake[i] >= snake[i + 1]:
            monotonic_score += 1
    reward += monotonic_score * 0.5
    
    # Big merge bonus
    if result['reward'] >= 512:
        reward += result['reward'] * 0.5
    
    return reward


if __name__ == "__main__":
    # Quick test
    agent = DQNAgent_CNN()
    print(f"Device: {agent.device}")
    print(f"CNN Network:\n{agent.policy_net}")
    
    # Test forward pass
    dummy_board = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 0, 0, 0, 0, 0])
    state = agent.encode_state(dummy_board)
    print(f"Encoded state shape: {state.shape}")
    
    action = agent.select_action(state, [True, True, True, True])
    print(f"Selected action: {action}")

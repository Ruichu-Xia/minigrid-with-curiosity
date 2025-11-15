"""
RIDE: Rewarding Impact-Driven Exploration for PPO-LSTM

Implementation based on:
Raileanu, R., & Rocktäschel, T. (2020). 
RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments.
ICLR 2020.

The intrinsic reward is computed as the L2 distance between consecutive state representations:
r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||_2

where φ is a CNN feature extractor (state embedding network).

CRITICAL IMPLEMENTATION DETAILS (from paper and official implementation):
1. State embedding network is a SEPARATE CNN that processes observations independently
2. Observations are normalized to [0, 1] before being fed to the state embedding network
3. State representations MUST be detached when computing intrinsic rewards (prevents exploitation)
4. The state embedding network is trained through forward/inverse dynamics losses
5. Intrinsic reward uses episodic state visitation normalization: r / sqrt(N(s))
6. Policy network uses 2-layer LSTM with 1024 hidden size and FC layers (1024->1024)
7. Total reward = r_extrinsic + β * r_intrinsic, where β is typically 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import defaultdict

from models.ppo_lstm import PPOLSTMActorCritic, RolloutStorage, PPOLSTMAgent


class StateEmbeddingNet(nn.Module):
    """
    State embedding network for RIDE - matches original implementation.
    
    This is a SEPARATE CNN (not shared with policy) that extracts state representations 
    for intrinsic reward computation.
    
    Architecture matches original RIDE implementation for MiniGrid:
    - 3 conv layers: 32 -> 32 -> 128 channels
    - stride=2, padding=1 for all layers
    - ELU activations
    - Observations are normalized to [0, 1] before processing
    
    Based on MinigridStateEmbeddingNet from original RIDE code.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int]):
        super().__init__()
        h, w, c = obs_shape
        self.observation_shape = obs_shape
        
        # CNN architecture matching original RIDE: 32 -> 32 -> 128 channels
        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        
        # Initialize weights using orthogonal initialization (matching original)
        for m in self.feat_extract:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))  # gain for ReLU/ELU
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract state representation from observation.
        
        CRITICAL: Observations are normalized to [0, 1] (matching original RIDE).
        Policy network does NOT normalize, but state embedding does.
        
        Args:
            obs: (batch, H, W, C) or (batch, seq_len, H, W, C) - uint8 [0, 255]
        
        Returns:
            state_rep: (batch, feature_dim) or (batch, seq_len, feature_dim)
        """
        is_sequence = len(obs.shape) == 5
        
        if not is_sequence:
            obs = obs.unsqueeze(1)  # (batch, 1, H, W, C)
        
        T, B = obs.shape[:2]
        
        # Handle case where B=0 (empty batch) - this can happen with seq_len=1 sequences
        if B == 0:
            # Return empty tensor with correct shape
            # Get observation dimensions (H, W, C) from shape if available
            if len(obs.shape) >= 5:
                h, w, c = obs.shape[2:5]
            elif len(obs.shape) >= 4:
                h, w, c = obs.shape[1:4]
            else:
                # Fallback: use default MiniGrid observation shape (7, 7, 3)
                h, w, c = 7, 7, 3
            # After 3 conv layers with stride=2: H/8, W/8, 128 channels
            feature_dim = max(1, (h // 8)) * max(1, (w // 8)) * 128
            
            if is_sequence:
                return torch.zeros(T, B, feature_dim, device=obs.device, dtype=torch.float32)
            else:
                return torch.zeros(B, feature_dim, device=obs.device, dtype=torch.float32)
        
        # Flatten time and batch dimensions
        x = obs.reshape(-1, *obs.shape[2:])  # (T*B, H, W, C)
        
        # CRITICAL: Normalize to [0, 1] (matching original RIDE)
        x = x.float() / 255.0
        
        # Transpose to (T*B, C, H, W) for CNN
        x = x.permute(0, 3, 1, 2)
        
        # Extract features
        x = self.feat_extract(x)  # (T*B, 128, H', W')
        
        # Flatten spatial dimensions
        state_embedding = x.reshape(T, B, -1)  # (T, B, feature_dim)
        
        # Note: Original RIDE does not normalize state embeddings here
        # The embeddings are trained through forward/inverse dynamics losses
        # which naturally encourages diverse, meaningful representations
        
        if not is_sequence:
            state_embedding = state_embedding.squeeze(0)  # (B, feature_dim)
        
        return state_embedding


class ForwardDynamicsNet(nn.Module):
    """
    Forward dynamics model for RIDE - matches original implementation.
    
    Predicts the next state embedding from the current state embedding and action.
    This model is used to train the state embedding network to learn predictable
    representations that capture the dynamics of the environment.
    
    Architecture matches MinigridForwardDynamicsNet from original RIDE:
    - Input: state_embedding (128-dim) + action (one-hot, num_actions)
    - Hidden: Linear(128 + num_actions -> 256) + ReLU
    - Output: Linear(256 -> 128) (predicted next state embedding)
    """
    
    def __init__(self, num_actions: int, state_embedding_dim: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.state_embedding_dim = state_embedding_dim
        
        # Forward dynamics network
        self.forward_dynamics = nn.Sequential(
            nn.Linear(state_embedding_dim + num_actions, 256),
            nn.ReLU(),
        )
        
        # Output layer
        self.fd_out = nn.Linear(256, state_embedding_dim)
        
        # Initialize weights using orthogonal initialization (matching original)
        for m in self.forward_dynamics:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))  # gain for ReLU
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.fd_out.weight, gain=0.01)
        nn.init.constant_(self.fd_out.bias, 0)
    
    def forward(self, state_embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state embedding from current state embedding and action.
        
        Args:
            state_embedding: (batch, seq_len, state_embedding_dim) or (batch, state_embedding_dim)
            action: (batch, seq_len) or (batch,) - action indices
        
        Returns:
            pred_next_state_emb: (batch, seq_len, state_embedding_dim) or (batch, state_embedding_dim)
        """
        is_sequence = len(state_embedding.shape) == 3
        
        if not is_sequence:
            state_embedding = state_embedding.unsqueeze(1)  # (batch, 1, state_embedding_dim)
            action = action.unsqueeze(1)  # (batch, 1)
        
        batch_size, seq_len = state_embedding.shape[:2]
        
        # One-hot encode actions
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()  # (batch, seq_len, num_actions)
        
        # Concatenate state embedding and action
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)  # (batch, seq_len, state_embedding_dim + num_actions)
        
        # Flatten for processing
        inputs_flat = inputs.reshape(-1, inputs.shape[-1])  # (batch * seq_len, state_embedding_dim + num_actions)
        
        # Forward pass
        hidden = self.forward_dynamics(inputs_flat)  # (batch * seq_len, 256)
        pred_next_state_emb = self.fd_out(hidden)  # (batch * seq_len, state_embedding_dim)
        
        # Reshape back
        pred_next_state_emb = pred_next_state_emb.reshape(batch_size, seq_len, -1)  # (batch, seq_len, state_embedding_dim)
        
        if not is_sequence:
            pred_next_state_emb = pred_next_state_emb.squeeze(1)  # (batch, state_embedding_dim)
        
        return pred_next_state_emb


class InverseDynamicsNet(nn.Module):
    """
    Inverse dynamics model for RIDE - matches original implementation.
    
    Predicts the action that was taken given the current and next state embeddings.
    This model is used to train the state embedding network to learn representations
    that capture action-relevant information.
    
    Architecture matches MinigridInverseDynamicsNet from original RIDE:
    - Input: state_embedding (128-dim) + next_state_embedding (128-dim)
    - Hidden: Linear(2 * 128 -> 256) + ReLU
    - Output: Linear(256 -> num_actions) (action logits)
    """
    
    def __init__(self, num_actions: int, state_embedding_dim: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.state_embedding_dim = state_embedding_dim
        
        # Inverse dynamics network
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(2 * state_embedding_dim, 256),
            nn.ReLU(),
        )
        
        # Output layer
        self.id_out = nn.Linear(256, num_actions)
        
        # Initialize weights using orthogonal initialization (matching original)
        for m in self.inverse_dynamics:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))  # gain for ReLU
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.id_out.weight, gain=0.01)
        nn.init.constant_(self.id_out.bias, 0)
    
    def forward(self, state_embedding: torch.Tensor, next_state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict action from current and next state embeddings.
        
        Args:
            state_embedding: (batch, seq_len, state_embedding_dim) or (batch, state_embedding_dim)
            next_state_embedding: (batch, seq_len, state_embedding_dim) or (batch, state_embedding_dim)
        
        Returns:
            action_logits: (batch, seq_len, num_actions) or (batch, num_actions)
        """
        is_sequence = len(state_embedding.shape) == 3
        
        if not is_sequence:
            state_embedding = state_embedding.unsqueeze(1)  # (batch, 1, state_embedding_dim)
            next_state_embedding = next_state_embedding.unsqueeze(1)  # (batch, 1, state_embedding_dim)
        
        batch_size, seq_len = state_embedding.shape[:2]
        
        # Concatenate state embeddings
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)  # (batch, seq_len, 2 * state_embedding_dim)
        
        # Flatten for processing
        inputs_flat = inputs.reshape(-1, inputs.shape[-1])  # (batch * seq_len, 2 * state_embedding_dim)
        
        # Forward pass
        hidden = self.inverse_dynamics(inputs_flat)  # (batch * seq_len, 256)
        action_logits = self.id_out(hidden)  # (batch * seq_len, num_actions)
        
        # Reshape back
        action_logits = action_logits.reshape(batch_size, seq_len, -1)  # (batch, seq_len, num_actions)
        
        if not is_sequence:
            action_logits = action_logits.squeeze(1)  # (batch, num_actions)
        
        return action_logits


class RIDEActorCritic(nn.Module):
    """
    Actor-Critic network with LSTM for RIDE - matches original MinigridPolicyNet architecture.
    
    Architecture (matching original RIDE):
    - CNN: 3 conv layers (32 -> 32 -> 32 channels)
    - FC: 2 fully connected layers (1024 -> 1024) with ReLU
    - LSTM: 2 layers with 1024 hidden size
    - Heads: Actor and Critic linear layers
    
    CRITICAL: The state embedding network is completely separate from the policy network
    (matching original RIDE implementation). It's trained through the optimizer but uses
    separate forward passes with normalized observations.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int, 
                 hidden_size: int = 1024, num_layers: int = 2):
        """
        Args:
            obs_shape: (H, W, C) observation shape
            num_actions: Number of actions
            hidden_size: LSTM hidden size (default 1024, matching original RIDE)
            num_layers: Number of LSTM layers (default 2, matching original RIDE)
        """
        super().__init__()
        h, w, c = obs_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN feature extractor (matching original RIDE)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()
        )
        
        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).shape[1]
        
        # FC layers between CNN and LSTM (matching original RIDE)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        
        # LSTM core (2 layers, 1024 hidden size, matching original RIDE)
        self.core = nn.LSTM(1024, hidden_size, num_layers, batch_first=True)
        
        # Actor and Critic heads
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Create SEPARATE state embedding network (not shared with policy)
        # This matches the original RIDE implementation
        self.state_embedding_net = StateEmbeddingNet(obs_shape)
        
        # Initialize weights using orthogonal initialization (matching original RIDE)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization (matching original RIDE)."""
        for name, param in self.named_parameters():
            if 'state_embedding_net' in name:
                # State embedding net has its own initialization
                continue
            if 'weight' in name:
                if 'core' in name or 'lstm' in name:
                    # LSTM weights: orthogonal initialization
                    nn.init.orthogonal_(param)
                elif 'conv' in name:
                    # Conv weights: orthogonal with sqrt(2) gain for ELU
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif 'fc' in name:
                    # FC weights: orthogonal with sqrt(2) gain for ReLU
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    # Actor/Critic heads: small orthogonal initialization
                    nn.init.orthogonal_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, obs: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor],
                continuation_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for policy computation.
        
        NOTE: State embedding is NOT computed here to avoid unnecessary computation.
        It's only computed when needed (for intrinsic rewards during collection,
        or for dynamics losses during training).
        
        CRITICAL: Policy network does NOT normalize observations (matching original RIDE).
        Observations are converted to float but not divided by 255.
        
        Args:
            obs: (batch, H, W, C) or (batch, seq_len, H, W, C) - uint8 [0, 255]
            hidden_state: (h, c) tuple where each is (num_layers, batch, hidden_size)
            continuation_mask: (batch, seq_len) Optional mask where 1=continue, 0=done
        
        Returns:
            logits: (batch, seq_len, num_actions) or (batch, num_actions)
            value: (batch, seq_len, 1) or (batch, 1)
            new_hidden: (h, c) tuple
        """
        is_sequence = len(obs.shape) == 5
        
        if not is_sequence:
            # Single timestep: add sequence dimension
            obs = obs.unsqueeze(1)  # (batch, 1, H, W, C)
        
        batch_size, seq_len = obs.shape[:2]
        
        # Convert to float but DO NOT normalize (matching original RIDE)
        obs_flat = obs.reshape(-1, *obs.shape[2:]).float()  # (batch * seq_len, H, W, C)
        
        # Reshape for CNN: (batch * seq_len, C, H, W)
        obs_flat = obs_flat.permute(0, 3, 1, 2)
        
        # Extract features through CNN
        features = self.conv(obs_flat)  # (batch * seq_len, conv_out_size)
        
        # FC layers
        features = self.fc(features)  # (batch * seq_len, 1024)
        
        # Reshape for LSTM: (batch, seq_len, 1024)
        features = features.reshape(batch_size, seq_len, -1)
        
        # Process through LSTM
        if continuation_mask is not None:
            # Handle episode boundaries by processing with mask
            lstm_out, new_hidden = self._forward_with_mask(features, hidden_state, continuation_mask)
        else:
            # Standard LSTM processing
            lstm_out, new_hidden = self.core(features, hidden_state)
        
        # Actor and Critic heads
        logits = self.actor(lstm_out)  # (batch, seq_len, num_actions)
        value = self.critic(lstm_out)   # (batch, seq_len, 1)
        
        if not is_sequence:
            # Remove sequence dimension for single timestep
            logits = logits.squeeze(1)
            value = value.squeeze(1)
        
        return logits, value, new_hidden
    
    def _forward_with_mask(self, features: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor],
                          continuation_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequences with continuation masks to reset hidden states at episode boundaries.
        
        Args:
            features: (batch, seq_len, feature_dim)
            hidden_state: (h, c) tuple where each is (num_layers, batch, hidden_size)
            continuation_mask: (batch, seq_len) - 1 where episode continues, 0 where it ends
        
        Returns:
            lstm_out: (batch, seq_len, hidden_size)
            new_hidden: (h, c) tuple with final hidden states
        """
        batch_size, seq_len, _ = features.shape
        h, c = hidden_state
        
        # Check if any sequences have dones - if not, use efficient batched processing
        if torch.all(continuation_mask == 1.0):
            # No episode boundaries in this batch - use efficient LSTM
            return self.core(features, (h, c))
        
        # Process with masking (less efficient but handles episode boundaries)
        outputs = []
        for t in range(seq_len):
            # Reset hidden state where episodes ended in previous step
            if t > 0:
                # continuation_mask[:, t-1] = 0 means episode ended at t-1
                # So we reset hidden state before processing t
                mask = continuation_mask[:, t-1].reshape(1, batch_size, 1)  # (1, batch, 1)
                h = h * mask
                c = c * mask
            
            # Process single timestep
            out, (h, c) = self.core(features[:, t:t+1], (h, c))
            outputs.append(out)
        
        lstm_out = torch.cat(outputs, dim=1)
        return lstm_out, (h, c)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
    
    def get_state_representation(self, obs: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Extract state representation from observation using SEPARATE state embedding network.
        
        CRITICAL IMPLEMENTATION DETAILS (matching original RIDE):
        1. State embedding network is SEPARATE from policy network (not shared)
        2. Observations are normalized to [0, 1] before being fed to state embedding
        3. Policy network does NOT normalize observations (handled in parent class)
        4. State representations are detached when computing intrinsic rewards
           (prevents agent from manipulating features for higher rewards)
        5. State embedding network is trained through the optimizer (it's in parameters())
           even though we detach the output for intrinsic rewards
        
        Args:
            obs: (batch, H, W, C) or (batch, seq_len, H, W, C) - uint8 [0, 255]
            detach: If True, detach from computation graph (for intrinsic reward computation)
        
        Returns:
            state_rep: (batch, feature_dim) or (batch, seq_len, feature_dim)
        """
        # Use separate state embedding network
        # This network normalizes observations internally
        state_rep = self.state_embedding_net(obs)
        
        # Detach if needed for intrinsic reward computation
        # This prevents gradients from intrinsic rewards, but the network parameters
        # are still part of self.parameters() and get updated by the optimizer
        if detach:
            state_rep = state_rep.detach()
        
        return state_rep


class RIDERolloutStorage(RolloutStorage):
    """
    Rollout storage for RIDE that tracks state representations for intrinsic reward computation.
    """
    
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.current_episode['state_reps'] = []
    
    def add(self, obs: np.ndarray, action: int, log_prob: float, value: float,
            reward: float, done: bool, hidden_state: Tuple[torch.Tensor, torch.Tensor],
            next_hidden_state: Tuple[torch.Tensor, torch.Tensor], next_obs: np.ndarray,
            state_rep: Optional[torch.Tensor] = None):
        """
        Add a single timestep to current episode.
        
        Args:
            state_rep: State representation for this observation (for intrinsic reward)
        """
        super().add(obs, action, log_prob, value, reward, done, 
                   hidden_state, next_hidden_state, next_obs)
        
        if state_rep is not None:
            # Optimize: Only detach and move to CPU (already detached, so no need to clone)
            # Move to CPU to avoid keeping GPU memory
            self.current_episode['state_reps'].append(state_rep.detach().cpu())
        else:
            self.current_episode['state_reps'].append(None)
    
    def finish_episode(self):
        """Mark current episode as complete and start new one."""
        super().finish_episode()
        self.current_episode['state_reps'] = []
    
    def get_sequences(self, max_seq_len: int = 128) -> List[Dict]:
        """Get sequences with state representations."""
        sequences = []
        
        for episode in self.episodes:
            ep_len = len(episode['obs'])
            
            # Split episode into chunks
            for start_idx in range(0, ep_len, max_seq_len):
                end_idx = min(start_idx + max_seq_len, ep_len)
                
                # Get initial hidden state for this sequence
                init_hidden = episode['hidden_states'][start_idx]
                
                # Get final hidden state for this sequence (for bootstrapping)
                if end_idx < ep_len:
                    final_hidden = episode['hidden_states'][end_idx]
                    next_obs = episode['obs'][end_idx]
                else:
                    final_hidden = episode['final_hidden']
                    next_obs = episode.get('next_obs', None)
                
                # Get state representations for this sequence
                state_reps = []
                if 'state_reps' in episode and len(episode['state_reps']) >= end_idx:
                    state_reps = [sr for sr in episode['state_reps'][start_idx:end_idx] if sr is not None]
                
                seq = {
                    'obs': np.array(episode['obs'][start_idx:end_idx]),
                    'actions': np.array(episode['actions'][start_idx:end_idx]),
                    'log_probs': np.array(episode['log_probs'][start_idx:end_idx]),
                    'values': np.array(episode['values'][start_idx:end_idx]),
                    'rewards': np.array(episode['rewards'][start_idx:end_idx]),
                    'dones': np.array(episode['dones'][start_idx:end_idx]),
                    'init_hidden': init_hidden,
                    'final_hidden': final_hidden,
                    'next_obs': next_obs,
                    'state_reps': state_reps,
                }
                
                sequences.append(seq)
        
        return sequences
    
    def clear(self):
        """Clear all stored episodes."""
        super().clear()
        self.current_episode['state_reps'] = []


class RIDEAgent(PPOLSTMAgent):
    """
    PPO agent with RIDE (Rewarding Impact-Driven Exploration).
    
    RIDE adds intrinsic rewards based on the change in state representation:
    r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||_2
    
    The total reward is: r_total = r_extrinsic + β * r_intrinsic
    where β is the intrinsic reward coefficient (typically 0.1 in original RIDE).
    
    According to the RIDE paper and original implementation:
    - State representations are computed using a SEPARATE CNN (state embedding network)
    - State embedding network has architecture: 3 conv layers (32 -> 32 -> 128 channels)
    - State embedding normalizes observations to [0, 1]; policy network does NOT normalize
    - State representations are detached when computing intrinsic rewards
    - In original RIDE, state embedding is trained through forward/inverse dynamics losses
    - Intrinsic rewards use episodic state visitation normalization: r / sqrt(N(s))
    - Policy network uses 2-layer LSTM with 1024 hidden size and FC layers (1024->1024)
    """
    
    def __init__(self, env, device: str = 'cpu', lr: float = 3e-4, 
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 ppo_epochs: int = 4, ppo_minibatch_size: int = 4,
                 ppo_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 max_seq_len: int = 128, hidden_size: int = 1024,
                 clip_value_loss: bool = True,
                 intrinsic_reward_coef: float = 0.1,
                 forward_loss_coef: float = 0.1,
                 inverse_loss_coef: float = 0.1,
                 state_embedding_dim: int = 128,
                 use_intrinsic_normalization: bool = True):
        """
        Args:
            intrinsic_reward_coef: Coefficient β for intrinsic rewards (default 0.1, matching RIDE paper)
            forward_loss_coef: Coefficient for forward dynamics loss (default 0.1)
            inverse_loss_coef: Coefficient for inverse dynamics loss (default 0.1)
            state_embedding_dim: Dimension of state embeddings (default 128, matching original RIDE)
        """
        # Initialize parent class
        super().__init__(
            env, device, lr, gamma, gae_lambda, ppo_epochs, ppo_minibatch_size,
            ppo_epsilon, value_coef, entropy_coef, max_grad_norm, max_seq_len,
            hidden_size, clip_value_loss
        )
        
        # Replace actor-critic with RIDE version
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.n
        self.actor_critic = RIDEActorCritic(
            obs_shape, num_actions, hidden_size=hidden_size
        ).to(self.device)
        
        # Compute actual state embedding dimension from the network
        # The state embedding dimension depends on the observation shape
        with torch.no_grad():
            dummy_obs = torch.zeros(1, *obs_shape, dtype=torch.uint8).to(self.device)
            dummy_state_emb = self.actor_critic.get_state_representation(dummy_obs, detach=True)
            actual_state_embedding_dim = dummy_state_emb.shape[-1]
        
        # Create forward and inverse dynamics models with actual dimension
        self.forward_dynamics_model = ForwardDynamicsNet(
            num_actions, state_embedding_dim=actual_state_embedding_dim
        ).to(self.device)
        
        self.inverse_dynamics_model = InverseDynamicsNet(
            num_actions, state_embedding_dim=actual_state_embedding_dim
        ).to(self.device)
        
        # Store actual dimension
        self.state_embedding_dim = actual_state_embedding_dim
        
        # Separate optimizers for each component (matching original RIDE)
        # Policy optimizer (excludes state embedding to prevent interference)
        policy_params = [p for n, p in self.actor_critic.named_parameters() 
                        if 'state_embedding_net' not in n]
        self.optimizer = optim.Adam(policy_params, lr=lr)
        
        # State embedding optimizer (trained through dynamics losses)
        self.state_embedding_optimizer = optim.Adam(
            self.actor_critic.state_embedding_net.parameters(), lr=lr
        )
        self.forward_dynamics_optimizer = optim.Adam(
            self.forward_dynamics_model.parameters(), lr=lr
        )
        self.inverse_dynamics_optimizer = optim.Adam(
            self.inverse_dynamics_model.parameters(), lr=lr
        )
        
        # Replace storage with RIDE version
        self.storage = RIDERolloutStorage(self.device)
        
        # RIDE-specific parameters
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.forward_loss_coef = forward_loss_coef
        self.inverse_loss_coef = inverse_loss_coef
        self.use_intrinsic_normalization = use_intrinsic_normalization
        
        # Episodic state visitation tracking for intrinsic reward normalization
        # Maps state representation (as tuple of rounded values) to visitation count
        self.episode_state_visits = {}  # Reset at start of each episode
        self.current_episode_id = 0
        
        # state_embedding_dim will be set after computing actual dimension
    
    def compute_forward_dynamics_loss(self, pred_next_emb: torch.Tensor, 
                                     next_emb: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute forward dynamics loss (L2 norm between predicted and actual next state embedding).
        
        Matches original RIDE implementation:
        forward_dynamics_loss = ||pred_next_emb - next_emb||_2
        
        CRITICAL: Uses mask to exclude invalid/padded timesteps and episode boundaries.
        
        Args:
            pred_next_emb: (batch, seq_len, state_embedding_dim) - predicted next state embedding
            next_emb: (batch, seq_len, state_embedding_dim) - actual next state embedding
            mask: (batch, seq_len) - mask where 1 = valid, 0 = invalid (padded or episode boundary)
        
        Returns:
            Loss scalar (only over valid timesteps)
        """
        # Compute L2 norm per timestep
        forward_dynamics_loss = torch.norm(pred_next_emb - next_emb, dim=2, p=2)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # Only compute loss over valid timesteps
            forward_dynamics_loss = forward_dynamics_loss * mask
            # Mean over valid timesteps only
            valid_count = mask.sum()
            if valid_count > 0:
                return forward_dynamics_loss.sum() / valid_count
            else:
                return torch.tensor(0.0, device=forward_dynamics_loss.device)
        
        # No mask: mean over all timesteps
        return torch.mean(forward_dynamics_loss)
    
    def compute_inverse_dynamics_loss(self, pred_actions: torch.Tensor, 
                                     true_actions: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute inverse dynamics loss (cross-entropy for action prediction).
        
        Matches original RIDE implementation using NLL loss.
        
        CRITICAL: Uses mask to exclude invalid/padded timesteps and episode boundaries.
        
        Args:
            pred_actions: (batch, seq_len, num_actions) - predicted action logits
            true_actions: (batch, seq_len) - true action indices
            mask: (batch, seq_len) - mask where 1 = valid, 0 = invalid (padded or episode boundary)
        
        Returns:
            Loss scalar (only over valid timesteps)
        """
        # Flatten for loss computation
        batch_size, seq_len = pred_actions.shape[:2]
        # Use reshape instead of view for potentially non-contiguous tensors (from slicing)
        pred_actions_flat = pred_actions.reshape(-1, pred_actions.shape[-1])  # (batch * seq_len, num_actions)
        true_actions_flat = true_actions.reshape(-1)  # (batch * seq_len)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)  # (batch * seq_len)
            # Only compute loss over valid timesteps
            valid_indices = mask_flat.bool()
            if valid_indices.sum() > 0:
                pred_actions_valid = pred_actions_flat[valid_indices]
                true_actions_valid = true_actions_flat[valid_indices]
                # Cross-entropy loss over valid timesteps only
                inverse_dynamics_loss = F.nll_loss(
                    F.log_softmax(pred_actions_valid, dim=-1),
                    target=true_actions_valid,
                    reduction='mean'
                )
                return inverse_dynamics_loss
            else:
                return torch.tensor(0.0, device=pred_actions.device)
        
        # No mask: mean over all timesteps
        inverse_dynamics_loss = F.nll_loss(
            F.log_softmax(pred_actions_flat, dim=-1),
            target=true_actions_flat,
            reduction='mean'
        )
        return inverse_dynamics_loss
    
    def compute_intrinsic_reward(self, state_rep: torch.Tensor, 
                                prev_state_rep: Optional[torch.Tensor],
                                done: bool = False) -> float:
        """
        Compute intrinsic reward as L2 distance between consecutive state representations.
        
        According to RIDE paper: r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||_2
        
        With episodic state visitation normalization (matching original RIDE):
        r_intrinsic_normalized = r_intrinsic / sqrt(N(s))
        where N(s) is the number of times state s has been visited in the current episode.
        
        CRITICAL: 
        - State representations are detached to prevent gradient flow
        - This prevents the agent from learning to manipulate features for higher intrinsic rewards
        - Original RIDE uses episodic state visitation normalization to encourage exploration
        
        Args:
            state_rep: Current state representation (feature_dim,) - DETACHED
            prev_state_rep: Previous state representation (feature_dim,) or None - DETACHED
            done: Whether the episode has ended (used to reset visitation counts)
        
        Returns:
            Intrinsic reward (scalar, normalized L2 distance if normalization enabled)
        """
        # Reset state visitation counts at episode start
        if done:
            self.episode_state_visits = {}
            self.current_episode_id += 1
        
        if prev_state_rep is None:
            # First timestep: no intrinsic reward
            return 0.0
        
        # Compute L2 distance between state representations
        # Both should be detached to prevent gradient flow
        # r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||_2
        diff = state_rep - prev_state_rep
        control_reward = torch.norm(diff, p=2).item()
        
        # Apply episodic state visitation normalization (matching original RIDE)
        # CRITICAL: Normalize by visit count of CURRENT state (s_t), not next state (s_{t+1})
        # In original RIDE: intrinsic_rewards = count_rewards * control_rewards
        # where count_rewards = 1/sqrt(N(s_t)) and control_rewards = ||φ(s_{t+1}) - φ(s_t)||_2
        if self.use_intrinsic_normalization:
            # Create a hashable representation of the PREVIOUS state (s_t) for visitation tracking
            # This is the state we're transitioning FROM, which determines the normalization factor
            if prev_state_rep.is_cuda:
                prev_state_rep_cpu = prev_state_rep.cpu()
            else:
                prev_state_rep_cpu = prev_state_rep
            
            # Round to 1 decimal place for state grouping (matching original implementation)
            # This groups nearby states together so visitation counts grow as intended
            prev_state_rep_rounded = torch.round(prev_state_rep_cpu * 10).int().view(-1).numpy()
            # Use hash of the array data for efficient lookup
            prev_state_key = hash(prev_state_rep_rounded.tobytes())
            
            # Get visit count for the CURRENT state (s_t) BEFORE incrementing
            # This matches original RIDE: count_rewards = 1/sqrt(N(s_t))
            if prev_state_key not in self.episode_state_visits:
                self.episode_state_visits[prev_state_key] = 0
            
            # Get the count BEFORE incrementing (for this transition)
            visit_count = self.episode_state_visits[prev_state_key]
            
            # Increment visitation count for the CURRENT state (s_t) after using it
            # This ensures next time we visit s_t, the count will be higher
            self.episode_state_visits[prev_state_key] += 1
            
            # Also track the next state (s_{t+1}) for future transitions
            # But don't use it for normalization of THIS reward
            # CRITICAL BUG FIX: Use prev_state_rep (s_t) for normalization, not state_rep (s_{t+1})
            # In original RIDE: count_rewards = 1/sqrt(N(s_t)) where s_t is the CURRENT state
            if prev_state_rep.is_cuda:
                state_rep_cpu = state_rep.cpu()
            else:
                state_rep_cpu = state_rep
            state_rep_rounded = torch.round(state_rep_cpu * 10).int().view(-1).numpy()
            next_state_key = hash(state_rep_rounded.tobytes())
            if next_state_key not in self.episode_state_visits:
                self.episode_state_visits[next_state_key] = 0
            
            # Normalize by sqrt of visitation count of CURRENT state (s_t)
            # This reduces intrinsic reward for frequently visited states, encouraging exploration
            # Formula: intrinsic_reward = control_reward / sqrt(N(s_t))
            if visit_count > 0:
                count_reward = 1.0 / (np.sqrt(visit_count) + 1e-8)
            else:
                count_reward = 1.0  # First visit: no normalization
            intrinsic_reward = control_reward * count_reward
        else:
            intrinsic_reward = control_reward
        
        return intrinsic_reward
    
    def select_action(self, obs: np.ndarray, hidden_state: Tuple[torch.Tensor, torch.Tensor],
                     deterministic: bool = False) -> Tuple[int, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select action (same as parent class, no state representation needed here).
        
        Returns:
            action, log_prob, value, new_hidden_state
        """
        with torch.no_grad():
            # Optimize: Use torch.from_numpy instead of FloatTensor (avoids copy)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            logits, value, new_hidden = self.actor_critic(obs_tensor, hidden_state)
            
            dist = Categorical(logits=logits)
            
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item(), new_hidden
    
    def collect_rollout(self, num_steps: int, render: bool = False) -> Dict:
        """
        Collect rollout with RIDE intrinsic rewards.
        """
        self.actor_critic.eval()
        
        obs, _ = self.env.reset()
        hidden_state = self.actor_critic.init_hidden(1, self.device)
        
        episode_rewards = []
        episode_lengths = []
        episode_intrinsic_rewards = []
        current_episode_reward = 0
        current_episode_intrinsic_reward = 0
        current_episode_length = 0
        
        for step in range(num_steps):
            if render:
                self.env.render()
            
            # Get current state representation (DETACHED for intrinsic reward)
            # Optimize: Convert to tensor once and reuse
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            current_state_rep = self.actor_critic.get_state_representation(obs_tensor, detach=True).squeeze(0)
            
            # Select action
            action, log_prob, value, new_hidden = self.select_action(obs, hidden_state)
            
            # Take step in environment
            next_obs, extrinsic_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Get next state representation (DETACHED for intrinsic reward)
            # Optimize: Convert to tensor once and reuse
            next_obs_tensor = torch.from_numpy(next_obs).float().unsqueeze(0).to(self.device)
            next_state_rep = self.actor_critic.get_state_representation(next_obs_tensor, detach=True).squeeze(0)
            
            # Compute intrinsic reward as change in state representation
            # r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||_2
            # CRITICAL: State representations are detached to prevent agent from
            # learning to manipulate features for higher intrinsic rewards
            # Pass done flag for episodic state visitation normalization
            intrinsic_reward = self.compute_intrinsic_reward(next_state_rep, current_state_rep, done=done)
            
            # Total reward = extrinsic + intrinsic
            total_reward = extrinsic_reward + self.intrinsic_reward_coef * intrinsic_reward
            
            # Store transition with current state representation
            self.storage.add(obs, action, log_prob, value, total_reward, done,
                           hidden_state, new_hidden, next_obs, state_rep=current_state_rep)
            
            current_episode_reward += extrinsic_reward  # Track extrinsic only for stats
            current_episode_intrinsic_reward += intrinsic_reward
            current_episode_length += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_intrinsic_rewards.append(current_episode_intrinsic_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_intrinsic_reward = 0
                current_episode_length = 0
                
                # Reset state visitation counts for new episode
                # (also done in compute_intrinsic_reward, but reset here too for safety)
                self.episode_state_visits = {}
                
                obs, _ = self.env.reset()
                hidden_state = self.actor_critic.init_hidden(1, self.device)
            else:
                obs = next_obs
                hidden_state = new_hidden
        
        # Finish any incomplete episode
        self.storage.finish_episode()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_intrinsic_rewards': episode_intrinsic_rewards,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'mean_intrinsic_reward': np.mean(episode_intrinsic_rewards) if episode_intrinsic_rewards else 0,
        }
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update with forward and inverse dynamics losses (matching original RIDE).
        
        The state embedding network is trained through forward/inverse dynamics losses,
        which encourages it to learn predictable, action-relevant representations.
        """
        self.actor_critic.train()
        self.forward_dynamics_model.train()
        self.inverse_dynamics_model.train()
        
        # Get sequences with correct hidden states
        sequences = self.storage.get_sequences(self.max_seq_len)
        sequences = self.storage.compute_advantages(
            sequences, self.gamma, self.gae_lambda, self.actor_critic
        )
        
        # Normalize advantages across all sequences
        all_advantages = np.concatenate([seq['advantages'] for seq in sequences])
        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std()
        for seq in sequences:
            seq['advantages'] = (seq['advantages'] - adv_mean) / (adv_std + 1e-8)
        
        # Training statistics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_forward_loss = 0
        total_inverse_loss = 0
        num_updates = 0
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle sequences
            np.random.shuffle(sequences)
            
            # Process in minibatches
            for i in range(0, len(sequences), self.ppo_minibatch_size):
                batch_sequences = sequences[i:i + self.ppo_minibatch_size]
                
                # Pad sequences to same length for batching
                batch = self._prepare_batch(batch_sequences)
                
                # Forward pass for policy
                logits, values, _ = self.actor_critic(
                    batch['obs'],
                    batch['init_hidden'],
                    continuation_mask=batch['continuation_mask']
                )
                
                # Flatten for loss computation
                logits = logits[batch['valid_mask']]
                values = values[batch['valid_mask']].squeeze(-1)
                
                # Compute PPO losses
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch['actions'][batch['valid_mask']])
                entropy = dist.entropy().mean()
                
                # PPO actor loss
                ratio = (new_log_probs - batch['old_log_probs'][batch['valid_mask']]).exp()
                advantages = batch['advantages'][batch['valid_mask']]
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                returns = batch['returns'][batch['valid_mask']]
                critic_loss = (returns - values).pow(2).mean()
                
                # PPO total loss
                ppo_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # ===== Forward and Inverse Dynamics Losses =====
                # Get state embeddings for current and next observations (WITHOUT detach for training)
                # We need obs[:-1] and obs[1:] for state embeddings
                obs_batch = batch['obs']  # (batch, seq_len, H, W, C)
                actions_batch = batch['actions']  # (batch, seq_len)
                valid_mask = batch['valid_mask']  # (batch, seq_len) - True for valid, False for padded
                continuation_mask = batch['continuation_mask']  # (batch, seq_len) - 1.0 = continue, 0.0 = episode ended
                
                # Convert masks to tensors if they're numpy arrays (for tensor operations)
                # Note: valid_mask can be used as numpy for indexing, but we need tensors for boolean ops
                if isinstance(valid_mask, np.ndarray):
                    valid_mask_tensor = torch.from_numpy(valid_mask).to(self.device)
                else:
                    valid_mask_tensor = valid_mask
                if isinstance(continuation_mask, np.ndarray):
                    continuation_mask_tensor = torch.from_numpy(continuation_mask).to(self.device)
                else:
                    continuation_mask_tensor = continuation_mask
                
                # Get current and next observations (excluding last timestep for next_obs)
                batch_size, seq_len = obs_batch.shape[:2]
                
                # CRITICAL: Skip dynamics losses if seq_len <= 1 (no transitions possible)
                if seq_len <= 1:
                    # No transitions to compute dynamics losses for
                    forward_dynamics_loss = torch.tensor(0.0, device=self.device)
                    inverse_dynamics_loss = torch.tensor(0.0, device=self.device)
                else:
                    # Current observations: all except last timestep
                    current_obs = obs_batch[:, :-1]  # (batch, seq_len-1, H, W, C)
                    next_obs = obs_batch[:, 1:]  # (batch, seq_len-1, H, W, C)
                    current_actions = actions_batch[:, :-1]  # (batch, seq_len-1)
                    
                    # CRITICAL: Create mask for dynamics losses
                    # A transition (s_t, a_t, s_{t+1}) is valid if:
                    # 1. Both t and t+1 are valid (not padded) - use valid_mask
                    # 2. Episode didn't end at t (continuation_mask[t] = 1.0)
                    #    (If episode ended at t, then s_{t+1} is from a new episode, so transition is invalid)
                    
                    valid_mask_current = valid_mask_tensor[:, :-1]  # (batch, seq_len-1) - valid for current timestep
                    valid_mask_next = valid_mask_tensor[:, 1:]  # (batch, seq_len-1) - valid for next timestep
                    continuation_mask_current = continuation_mask_tensor[:, :-1]  # (batch, seq_len-1) - episode continues after current
                    
                    # Transition is valid if both timesteps are valid AND episode continues
                    # Use tensor operations for boolean logic
                    dynamics_mask = (
                        valid_mask_current.bool() & 
                        valid_mask_next.bool() & 
                        (continuation_mask_current > 0.5)
                    ).float()  # (batch, seq_len-1)
                    
                    # Get state embeddings (WITHOUT detach - we want gradients for training)
                    # Optimize: Batch both current and next observations together for efficiency
                    # Concatenate along sequence dimension, process, then split
                    batch_size, seq_len_minus_1 = current_obs.shape[:2]
                    combined_obs = torch.cat([current_obs, next_obs], dim=1)  # (batch, 2*(seq_len-1), H, W, C)
                    combined_state_emb = self.actor_critic.get_state_representation(combined_obs, detach=False)  # (batch, 2*(seq_len-1), state_embedding_dim)
                    # Split back
                    state_emb = combined_state_emb[:, :seq_len_minus_1]  # (batch, seq_len-1, state_embedding_dim)
                    next_state_emb = combined_state_emb[:, seq_len_minus_1:]  # (batch, seq_len-1, state_embedding_dim)
                    
                    # Forward dynamics: predict next state embedding from current state + action
                    pred_next_state_emb = self.forward_dynamics_model(state_emb, current_actions)  # (batch, seq_len-1, state_embedding_dim)
                    
                    # Inverse dynamics: predict action from current and next state embeddings
                    pred_actions = self.inverse_dynamics_model(state_emb, next_state_emb)  # (batch, seq_len-1, num_actions)
                    
                    # Compute dynamics losses with proper masking
                    # CRITICAL: Only compute losses over valid transitions (not padded, not across episode boundaries)
                    forward_dynamics_loss = self.compute_forward_dynamics_loss(
                        pred_next_state_emb, next_state_emb, mask=dynamics_mask
                    )
                    inverse_dynamics_loss = self.compute_inverse_dynamics_loss(
                        pred_actions, current_actions, mask=dynamics_mask
                    )
                
                # Total loss = PPO loss + dynamics losses
                # NOTE: Dynamics losses train the state embedding network, which is separate from policy
                # The state embedding network gets gradients through these losses
                total_loss = (ppo_loss + 
                             self.forward_loss_coef * forward_dynamics_loss +
                             self.inverse_loss_coef * inverse_dynamics_loss)
                
                # Only update state embedding through dynamics losses, not through PPO loss
                # This prevents the dynamics losses from interfering with policy learning
                
                # ===== Optimization =====
                # Zero gradients
                self.optimizer.zero_grad()
                self.state_embedding_optimizer.zero_grad()
                self.forward_dynamics_optimizer.zero_grad()
                self.inverse_dynamics_optimizer.zero_grad()
                
                # Backward pass
                total_loss.backward()
                
                # Clip gradients separately
                # Policy gradients (exclude state embedding)
                policy_params = [p for n, p in self.actor_critic.named_parameters() 
                               if 'state_embedding_net' not in n]
                if policy_params:
                    torch.nn.utils.clip_grad_norm_(policy_params, self.max_grad_norm)
                
                # State embedding and dynamics model gradients
                torch.nn.utils.clip_grad_norm_(self.actor_critic.state_embedding_net.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.forward_dynamics_model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.inverse_dynamics_model.parameters(), self.max_grad_norm)
                
                # Update all optimizers
                self.optimizer.step()
                self.state_embedding_optimizer.step()
                self.forward_dynamics_optimizer.step()
                self.inverse_dynamics_optimizer.step()
                
                # Track statistics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_forward_loss += forward_dynamics_loss.item()
                total_inverse_loss += inverse_dynamics_loss.item()
                num_updates += 1
        
        # Clear storage
        self.storage.clear()
        
        # Avoid division by zero
        if num_updates == 0:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'entropy': 0.0,
                'forward_dynamics_loss': 0.0,
                'inverse_dynamics_loss': 0.0,
                'num_sequences': len(sequences),
                'num_updates': 0
            }
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'forward_dynamics_loss': total_forward_loss / num_updates,
            'inverse_dynamics_loss': total_inverse_loss / num_updates,
            'num_sequences': len(sequences),
            'num_updates': num_updates
        }
    
    def save(self, path: str):
        """Save model checkpoint including RIDE-specific parameters and dynamics models."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'forward_dynamics_state_dict': self.forward_dynamics_model.state_dict(),
            'inverse_dynamics_state_dict': self.inverse_dynamics_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_embedding_optimizer_state_dict': self.state_embedding_optimizer.state_dict(),
            'forward_dynamics_optimizer_state_dict': self.forward_dynamics_optimizer.state_dict(),
            'inverse_dynamics_optimizer_state_dict': self.inverse_dynamics_optimizer.state_dict(),
            'intrinsic_reward_coef': self.intrinsic_reward_coef,
            'forward_loss_coef': self.forward_loss_coef,
            'inverse_loss_coef': self.inverse_loss_coef,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint including RIDE-specific parameters and dynamics models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        if 'forward_dynamics_state_dict' in checkpoint:
            self.forward_dynamics_model.load_state_dict(checkpoint['forward_dynamics_state_dict'])
            self.inverse_dynamics_model.load_state_dict(checkpoint['inverse_dynamics_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'state_embedding_optimizer_state_dict' in checkpoint:
                self.state_embedding_optimizer.load_state_dict(checkpoint['state_embedding_optimizer_state_dict'])
                self.forward_dynamics_optimizer.load_state_dict(checkpoint['forward_dynamics_optimizer_state_dict'])
                self.inverse_dynamics_optimizer.load_state_dict(checkpoint['inverse_dynamics_optimizer_state_dict'])
        if 'intrinsic_reward_coef' in checkpoint:
            self.intrinsic_reward_coef = checkpoint['intrinsic_reward_coef']
        if 'forward_loss_coef' in checkpoint:
            self.forward_loss_coef = checkpoint['forward_loss_coef']
        if 'inverse_loss_coef' in checkpoint:
            self.inverse_loss_coef = checkpoint['inverse_loss_coef']

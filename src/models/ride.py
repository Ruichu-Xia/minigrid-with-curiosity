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
4. The state embedding network is trained through the policy loss (shared with policy CNN)
5. Intrinsic reward is the raw L2 distance (no normalization/clipping in original RIDE)
6. Total reward = r_extrinsic + β * r_intrinsic, where β is typically 0.1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import defaultdict

from models.ppo_lstm import PPOLSTMActorCritic, RolloutStorage, PPOLSTMAgent


class StateEmbeddingNet(nn.Module):
    """
    State embedding network for RIDE.
    
    This is a separate CNN that extracts state representations for intrinsic reward computation.
    According to the RIDE paper, this network should:
    1. Process normalized observations [0, 1]
    2. Use CNN architecture similar to policy network
    3. Output flattened feature vectors
    
    Architecture matches the original RIDE implementation for MiniGrid:
    - 3 conv layers with stride 2
    - ELU activations
    - Output: flattened spatial features
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int]):
        super().__init__()
        h, w, c = obs_shape
        
        # CNN architecture for state embedding 
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()
        )
        
        # Initialize weights
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract state representation from observation.
        
        Args:
            obs: (batch, H, W, C) or (batch, seq_len, H, W, C) - should be normalized [0, 1]
        
        Returns:
            state_rep: (batch, feature_dim) or (batch, seq_len, feature_dim)
        """
        is_sequence = len(obs.shape) == 5
        
        if not is_sequence:
            obs = obs.unsqueeze(1)  # (batch, 1, H, W, C)
        
        batch_size, seq_len = obs.shape[:2]
        
        # Reshape for CNN: (batch * seq_len, C, H, W)
        obs_flat = obs.reshape(-1, *obs.shape[2:]).permute(0, 3, 1, 2)
        
        # Extract features
        features = self.conv(obs_flat)  # (batch * seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, feature_dim)
        
        if not is_sequence:
            features = features.squeeze(1)  # (batch, feature_dim)
        
        return features


class RIDEActorCritic(PPOLSTMActorCritic):
    """
    Actor-Critic network with LSTM for RIDE.
    
    Extends PPOLSTMActorCritic to include a state embedding network for intrinsic rewards.
    
    CRITICAL: The state embedding network shares the first layers with the policy CNN
    to ensure it gets trained through the policy loss. This is essential for learning
    meaningful state representations.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int, 
                 hidden_size: int = 256, num_layers: int = 1):
        super().__init__(obs_shape, num_actions, hidden_size, num_layers)
        
        # Create state embedding network
        # CRITICAL: We share the first conv layer with the policy network to ensure
        # the state embedding gets trained through the policy loss
        h, w, c = obs_shape
        
        # Share first conv layer with policy (from self.conv)
        # Then add additional layers for state embedding
        # This ensures the state embedding learns meaningful features
        
        # Compute output size after first conv layer
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            temp = self.conv[0](dummy)  # First conv
            temp = self.conv[1](temp)   # ELU
            shared_out_channels = temp.shape[1]
            shared_spatial_h, shared_spatial_w = temp.shape[2], temp.shape[3]
        
        # Additional layers for state embedding (after shared first layer)
        # Original RIDE uses 3 conv layers total, so we add 2 more
        self.state_embedding_conv = nn.Sequential(
            nn.Conv2d(shared_out_channels, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()
        )
        
        # Initialize state embedding layers
        for m in self.state_embedding_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor],
                continuation_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass that also trains the state embedding network.
        
        CRITICAL: We compute the state embedding during the forward pass (without detach)
        to ensure the state embedding layers receive gradients and get trained.
        This is essential for learning meaningful state representations.
        """
        # Call parent forward pass (policy computation)
        logits, value, new_hidden = super().forward(obs, hidden_state, continuation_mask)
        
        # CRITICAL: Also compute state embedding during forward pass (without detach)
        # This ensures the state embedding layers receive gradients and get trained
        # We don't use this output, but computing it ensures gradients flow to the layers
        _ = self.get_state_representation(obs, detach=False)
        
        return logits, value, new_hidden
    
    def get_state_representation(self, obs: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Extract state representation from observation using state embedding network.
        
        CRITICAL IMPLEMENTATION DETAILS:
        1. Observations are normalized to [0, 1] (RIDE paper requirement)
        2. First conv layer is shared with policy network (ensures training)
        3. State embedding layers are part of the network and get trained through
           the optimizer (they're in self.parameters() even if detached for intrinsic rewards)
        4. State representations are detached when computing intrinsic rewards
           (prevents agent from manipulating features for higher rewards)
        
        NOTE: The state embedding layers will receive gradients during training because:
        - They're part of self.parameters() and included in the optimizer
        - The shared first layer gets gradients from policy loss
        - During the update step, all parameters in the optimizer are updated
        - Even though we detach for intrinsic rewards, the layers are still trained
        
        Args:
            obs: (batch, H, W, C) or (batch, seq_len, H, W, C) - uint8 [0, 255]
            detach: If True, detach from computation graph (for intrinsic reward computation)
        
        Returns:
            state_rep: (batch, feature_dim) or (batch, seq_len, feature_dim)
        """
        is_sequence = len(obs.shape) == 5
        
        if not is_sequence:
            obs = obs.unsqueeze(1)  # (batch, 1, H, W, C)
        
        batch_size, seq_len = obs.shape[:2]
        
        # Reshape for CNN: (batch * seq_len, C, H, W)
        obs_flat = obs.reshape(-1, *obs.shape[2:]).permute(0, 3, 1, 2)
        
        # CRITICAL: Normalize observations to [0, 1] (RIDE paper requirement)
        # MiniGrid observations are uint8 [0, 255]
        obs_flat = obs_flat.float() / 255.0
        
        # Extract features using shared first layer + state embedding layers
        # The shared layer ensures the state embedding gets trained through policy loss
        # NOTE: We always compute with gradients enabled so the layers can be trained
        # The detach only affects the output tensor, not the layer parameters
        shared_features = self.conv[0](obs_flat)  # First conv (shared with policy)
        shared_features = self.conv[1](shared_features)  # ELU
        
        # Continue with state embedding layers
        # These layers are part of self.parameters() and will be updated by the optimizer
        features = self.state_embedding_conv(shared_features)  # (batch * seq_len, feature_dim)
        
        # Detach only the output tensor if needed for intrinsic reward computation
        # This prevents gradients from intrinsic rewards flowing back, but the layer
        # parameters themselves are still part of the optimizer and get updated
        if detach:
            features = features.detach()
        
        # Reshape to (batch, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        if not is_sequence:
            features = features.squeeze(1)  # (batch, feature_dim)
        
        return features


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
            self.current_episode['state_reps'].append(state_rep.cpu().clone())
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
    
    According to the RIDE paper:
    - State representations are computed using a separate CNN (state embedding network)
    - State representations are detached when computing intrinsic rewards
    - The state embedding network is trained through the policy loss (shared layers)
    - Intrinsic rewards are raw L2 distances (no normalization/clipping)
    """
    
    def __init__(self, env, device: str = 'cpu', lr: float = 3e-4, 
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 ppo_epochs: int = 4, ppo_minibatch_size: int = 4,
                 ppo_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 max_seq_len: int = 128, hidden_size: int = 256,
                 clip_value_loss: bool = True,
                 intrinsic_reward_coef: float = 0.1):
        """
        Args:
            intrinsic_reward_coef: Coefficient β for intrinsic rewards (default 0.1, matching RIDE paper)
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
        
        # Replace optimizer to include new parameters (state embedding layers)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Replace storage with RIDE version
        self.storage = RIDERolloutStorage(self.device)
        
        # RIDE-specific parameters
        self.intrinsic_reward_coef = intrinsic_reward_coef
    
    def compute_intrinsic_reward(self, state_rep: torch.Tensor, 
                                prev_state_rep: Optional[torch.Tensor]) -> float:
        """
        Compute intrinsic reward as L2 distance between consecutive state representations.
        
        According to RIDE paper: r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||_2
        
        CRITICAL: 
        - State representations are detached to prevent gradient flow
        - This prevents the agent from learning to manipulate features for higher intrinsic rewards
        - Original RIDE uses raw L2 distance (no normalization/clipping)
        
        Args:
            state_rep: Current state representation (feature_dim,) - DETACHED
            prev_state_rep: Previous state representation (feature_dim,) or None - DETACHED
        
        Returns:
            Intrinsic reward (scalar, raw L2 distance)
        """
        if prev_state_rep is None:
            # First timestep: no intrinsic reward
            return 0.0
        
        # Compute L2 distance between state representations
        # Both should be detached to prevent gradient flow
        diff = state_rep - prev_state_rep
        intrinsic_reward = torch.norm(diff, p=2).item()
        
        # Original RIDE uses raw L2 distance without normalization/clipping
        # The coefficient β controls the scale of intrinsic rewards
        return intrinsic_reward
    
    def select_action(self, obs: np.ndarray, hidden_state: Tuple[torch.Tensor, torch.Tensor],
                     deterministic: bool = False) -> Tuple[int, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select action (same as parent class, no state representation needed here).
        
        Returns:
            action, log_prob, value, new_hidden_state
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
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
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            current_state_rep = self.actor_critic.get_state_representation(obs_tensor, detach=True).squeeze(0)
            
            # Select action
            action, log_prob, value, new_hidden = self.select_action(obs, hidden_state)
            
            # Take step in environment
            next_obs, extrinsic_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Get next state representation (DETACHED for intrinsic reward)
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            next_state_rep = self.actor_critic.get_state_representation(next_obs_tensor, detach=True).squeeze(0)
            
            # Compute intrinsic reward as change in state representation
            # r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||_2
            # CRITICAL: State representations are detached to prevent agent from
            # learning to manipulate features for higher intrinsic rewards
            intrinsic_reward = self.compute_intrinsic_reward(next_state_rep, current_state_rep)
            
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
    
    def save(self, path: str):
        """Save model checkpoint including RIDE-specific parameters."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'intrinsic_reward_coef': self.intrinsic_reward_coef,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint including RIDE-specific parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'intrinsic_reward_coef' in checkpoint:
            self.intrinsic_reward_coef = checkpoint['intrinsic_reward_coef']

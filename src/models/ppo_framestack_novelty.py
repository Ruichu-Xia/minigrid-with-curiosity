import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

from curiosity_modules.novelty import NoveltyApproachReward


class FrameStack:
    """
    Utility class to stack frames for temporal context.
    Maintains a fixed-size buffer of recent observations.
    """
    
    def __init__(self, num_frames: int):
        """
        Args:
            num_frames: Number of frames to stack
        """
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, obs: np.ndarray) -> np.ndarray:
        """
        Reset with initial observation (repeated num_frames times).
        
        Args:
            obs: Initial observation (H, W, C)
        
        Returns:
            Stacked frames (num_frames, H, W, C)
        """
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self.get_stacked()
    
    def update(self, obs: np.ndarray) -> np.ndarray:
        """
        Add new observation and return stacked frames.
        
        Args:
            obs: New observation (H, W, C)
        
        Returns:
            Stacked frames (num_frames, H, W, C)
        """
        self.frames.append(obs)
        return self.get_stacked()
    
    def get_stacked(self) -> np.ndarray:
        """
        Get current stacked frames.
        
        Returns:
            Stacked frames (num_frames, H, W, C)
        """
        return np.array(list(self.frames))


class PPOConvActorCritic(nn.Module):
    """
    Actor-Critic network with frame stacking for PPO.
    Simpler and more stable than LSTM for many tasks.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int,
                 num_frames: int = 4, hidden_size: int = 256):
        """
        Args:
            obs_shape: Single frame shape (H, W, C)
            num_actions: Number of discrete actions
            num_frames: Number of frames to stack
            hidden_size: Size of feedforward hidden layers
        """
        super().__init__()
        h, w, c = obs_shape
        self.num_frames = num_frames
        
        # CNN feature extractor (input channels = C * num_frames)
        self.conv = nn.Sequential(
            nn.Conv2d(c * num_frames, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()
        )
        
        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c * num_frames, h, w)
            conv_out_size = self.conv(dummy).shape[1]
        
        # Feedforward layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU()
        )
        
        # Actor and Critic heads
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if module == self.actor:
                    # Small initialization for policy head
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: (batch, num_frames, H, W, C) or (batch, num_frames * C, H, W)
        
        Returns:
            logits: (batch, num_actions)
            value: (batch, 1)
        """
        # Handle different input formats
        if len(obs.shape) == 5:
            # (batch, num_frames, H, W, C) -> (batch, num_frames * C, H, W)
            batch_size, num_frames, h, w, c = obs.shape
            obs = obs.permute(0, 1, 4, 2, 3)  # (batch, num_frames, C, H, W)
            obs = obs.reshape(batch_size, num_frames * c, h, w)
        
        # Extract features
        features = self.conv(obs)
        features = self.fc(features)
        
        # Actor and Critic outputs
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value


class RolloutBuffer:
    """
    Simple buffer for storing and retrieving PPO rollout data.
    No sequence handling needed with feedforward architecture.
    """
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.next_observations = []
    
    def add(self, obs: np.ndarray, action: int, log_prob: float, 
            value: float, reward: float, done: bool, next_obs: np.ndarray):
        """Add a single timestep to buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_observations.append(next_obs)
    
    def get(self) -> Dict[str, np.ndarray]:
        """
        Get all stored data as numpy arrays.
        
        Returns:
            Dictionary with all rollout data
        """
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'next_observations': np.array(self.next_observations)
        }
    
    def clear(self):
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_observations.clear()
    
    def __len__(self):
        return len(self.observations)


class PPOFrameStackAgent:
    """
    PPO agent with frame stacking for temporal context.
    Simpler and more stable than LSTM-based approaches.
    """
    
    def __init__(self, env, device: str = 'cpu', lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 ppo_epochs: int = 4, ppo_batch_size: int = 64,
                 ppo_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 num_frames: int = 4, hidden_size: int = 256,
                 clip_value_loss: bool = True,
                 # Curiosity parameters
                 use_curiosity: bool = False,
                 curiosity_approach_scale: float = 0.3,
                 curiosity_interaction_scale: float = 1.0,
                 extrinsic_reward_scale: float = 10.0,
                 intrinsic_reward_scale: float = 0.1):
        """
        Args:
            env: Gym environment
            device: Device for torch tensors ('cpu' or 'cuda')
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            ppo_epochs: Number of PPO update epochs per rollout
            ppo_batch_size: Minibatch size for PPO updates
            ppo_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            num_frames: Number of frames to stack
            hidden_size: Size of feedforward hidden layers
            clip_value_loss: Whether to clip value function updates
            use_curiosity: Whether to use intrinsic curiosity rewards
            curiosity_approach_scale: Scale for approach rewards
            curiosity_interaction_scale: Scale for interaction rewards
            extrinsic_reward_scale: Scale for environment rewards
            intrinsic_reward_scale: Scale for curiosity rewards
        """
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epsilon = ppo_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_frames = num_frames
        self.clip_value_loss = clip_value_loss
        self.extrinsic_reward_scale = extrinsic_reward_scale
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # Initialize curiosity module
        self.use_curiosity = use_curiosity
        if self.use_curiosity:
            self.curiosity_module = NoveltyApproachReward(
                novelty_reward_scale=curiosity_approach_scale,
                interaction_reward_scale=curiosity_interaction_scale
            )
            print(f"🔍 Curiosity enabled: approach={curiosity_approach_scale}, "
                  f"interaction={curiosity_interaction_scale}")
        else:
            self.curiosity_module = None
            print("🔍 Curiosity disabled (baseline)")
        
        # Get environment specs
        obs_shape = env.observation_space.shape  # (H, W, C)
        num_actions = env.action_space.n
        
        # Initialize network
        self.actor_critic = PPOConvActorCritic(
            obs_shape=obs_shape,
            num_actions=num_actions,
            num_frames=num_frames,
            hidden_size=hidden_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Initialize buffer
        self.buffer = RolloutBuffer()
        
        # Frame stacker
        self.frame_stack = FrameStack(num_frames)
    
    def select_action(self, stacked_obs: np.ndarray, 
                     deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action given stacked observations.
        
        Args:
            stacked_obs: (num_frames, H, W, C)
            deterministic: Whether to select action deterministically
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            # Add batch dimension and convert to tensor
            obs_tensor = torch.FloatTensor(stacked_obs).unsqueeze(0).to(self.device)
            
            # Forward pass
            logits, value = self.actor_critic(obs_tensor)
            
            # Sample action
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def collect_rollout(self, num_steps: int, render: bool = False) -> Dict:
        """
        Collect rollout with optional curiosity-driven intrinsic rewards.
        
        Args:
            num_steps: Number of environment steps to collect
            render: Whether to render the environment
        
        Returns:
            Dictionary with rollout statistics
        """
        self.actor_critic.eval()
        
        # Reset environment and frame stack
        obs, _ = self.env.reset()
        stacked_obs = self.frame_stack.reset(obs)
        
        # Episode tracking
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # Curiosity tracking
        if self.use_curiosity:
            episode_intrinsic_rewards = []
            episode_extrinsic_rewards = []
            episode_scaled_extrinsic_rewards = []
            episode_approach_rewards = []
            episode_interaction_rewards = []
            current_intrinsic = 0
            current_extrinsic = 0
            current_scaled_extrinsic = 0
            current_approach = 0
            current_interaction = 0
            total_interactions = 0
        
        for step in range(num_steps):
            if render:
                self.env.render()
            
            # Select action
            action, log_prob, value = self.select_action(stacked_obs)
            
            # Environment step
            next_obs, env_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Update frame stack
            next_stacked_obs = self.frame_stack.update(next_obs)
            
            # Compute total reward (extrinsic + intrinsic)
            scaled_env_reward = env_reward * self.extrinsic_reward_scale
            total_reward = scaled_env_reward
            
            if self.use_curiosity:
                # Compute intrinsic reward based on single-frame observations
                intrinsic_reward, curiosity_info = self.curiosity_module.compute_reward(
                    obs, next_obs, action,
                    terminated=terminated,
                    truncated=truncated,
                    env_reward=scaled_env_reward
                )
                intrinsic_reward *= self.intrinsic_reward_scale
                total_reward = scaled_env_reward + intrinsic_reward
                
                # Track statistics
                current_intrinsic += intrinsic_reward
                current_extrinsic += env_reward
                current_scaled_extrinsic += scaled_env_reward
                current_approach += curiosity_info['approach_reward']
                current_interaction += curiosity_info['interaction_reward']
                total_interactions += curiosity_info['num_interactions']
            
            # Store transition
            self.buffer.add(
                stacked_obs, action, log_prob, value,
                total_reward, done, next_stacked_obs
            )
            
            # Update episode statistics
            current_episode_reward += total_reward
            current_episode_length += 1
            
            if done:
                # Record episode statistics
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                
                if self.use_curiosity:
                    episode_intrinsic_rewards.append(current_intrinsic)
                    episode_extrinsic_rewards.append(current_extrinsic)
                    episode_scaled_extrinsic_rewards.append(current_scaled_extrinsic)
                    episode_approach_rewards.append(current_approach)
                    episode_interaction_rewards.append(current_interaction)
                    current_intrinsic = 0
                    current_extrinsic = 0
                    current_scaled_extrinsic = 0
                    current_approach = 0
                    current_interaction = 0
                
                # Reset episode
                current_episode_reward = 0
                current_episode_length = 0
                obs, _ = self.env.reset()
                stacked_obs = self.frame_stack.reset(obs)
                
                # Reset curiosity episode tracking (but NOT interaction history)
                if self.use_curiosity:
                    self.curiosity_module.reset_episode()
            else:
                obs = next_obs
                stacked_obs = next_stacked_obs
        
        # Compile statistics
        stats = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'num_episodes': len(episode_rewards)
        }
        
        if self.use_curiosity:
            mean_intrinsic = np.mean(episode_intrinsic_rewards) if episode_intrinsic_rewards else 0
            mean_extrinsic = np.mean(episode_extrinsic_rewards) if episode_extrinsic_rewards else 0
            mean_scaled_extrinsic = np.mean(episode_scaled_extrinsic_rewards) if episode_scaled_extrinsic_rewards else 0
            mean_total = mean_intrinsic + mean_scaled_extrinsic
            
            stats.update({
                'mean_intrinsic_reward': mean_intrinsic,
                'mean_extrinsic_reward': mean_extrinsic,
                'mean_scaled_extrinsic_reward': mean_scaled_extrinsic,
                'mean_approach_reward': np.mean(episode_approach_rewards) if episode_approach_rewards else 0,
                'mean_interaction_reward': np.mean(episode_interaction_rewards) if episode_interaction_rewards else 0,
                'total_interactions': total_interactions,
                'intrinsic_ratio': mean_intrinsic / mean_total if mean_total > 0 else 0,
                'unique_objects_interacted': len(self.curiosity_module.novelty_tracker.interacted_objects),
                'total_interaction_count': sum(self.curiosity_module.novelty_tracker.interaction_counts.values())
            })
        
        return stats
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: (num_steps,)
            values: (num_steps,)
            dones: (num_steps,)
            next_values: (num_steps,)
        
        Returns:
            advantages: (num_steps,)
            returns: (num_steps,)
        """
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(num_steps)):
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γλ * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.
        
        Returns:
            Dictionary with training statistics
        """
        self.actor_critic.train()
        
        # Get rollout data
        data = self.buffer.get()
        num_steps = len(self.buffer)
        
        if num_steps == 0:
            return {}
        
        # Convert to tensors
        observations = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        old_values = torch.FloatTensor(data['values']).to(self.device)
        rewards = torch.FloatTensor(data['rewards']).to(self.device)
        dones = torch.FloatTensor(data['dones']).to(self.device)
        next_observations = torch.FloatTensor(data['next_observations']).to(self.device)
        
        # Compute next values for GAE
        with torch.no_grad():
            _, next_values = self.actor_critic(next_observations)
            next_values = next_values.squeeze(-1)
        
        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_updates = 0
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Generate random minibatches
            indices = np.arange(num_steps)
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_steps, self.ppo_batch_size):
                end_idx = min(start_idx + self.ppo_batch_size, num_steps)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                logits, values = self.actor_critic(batch_obs)
                values = values.squeeze(-1)
                
                # Compute new log probabilities and entropy
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO actor loss with clipping
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Clip fraction (diagnostic)
                clip_fraction = ((ratio - 1.0).abs() > self.ppo_epsilon).float().mean()
                
                # Critic loss with optional clipping
                if self.clip_value_loss:
                    # Clip value function updates
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.ppo_epsilon,
                        self.ppo_epsilon
                    )
                    critic_loss_unclipped = (values - batch_returns).pow(2)
                    critic_loss_clipped = (values_clipped - batch_returns).pow(2)
                    critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                else:
                    critic_loss = (values - batch_returns).pow(2).mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction.item()
                num_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'clip_fraction': total_clip_fraction / num_updates,
            'num_transitions': num_steps,
            'num_updates': num_updates
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Model loaded from {path}")
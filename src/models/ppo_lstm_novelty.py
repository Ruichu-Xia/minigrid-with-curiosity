import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from curiosity_modules.novelty import NoveltyApproachReward


class PPOLSTMActorCritic(nn.Module):
    """
    Actor-Critic network with LSTM for PPO.
    Designed for efficient batch processing of sequences.
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int, 
                 hidden_size: int = 256, num_layers: int = 1):
        super().__init__()
        h, w, c = obs_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN feature extractor
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
        
        # LSTM layer
        self.lstm = nn.LSTM(conv_out_size, hidden_size, num_layers, batch_first=True)
        
        # Actor and Critic heads
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'conv' in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    nn.init.orthogonal_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, obs: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor],
                continuation_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass supporting both single timestep and sequence processing.
        
        Args:
            obs: (batch, seq_len, H, W, C) or (batch, H, W, C)
            hidden_state: (h, c) tuple where each is (num_layers, batch, hidden_size)
            continuation_mask: (batch, seq_len) Optional mask where 1=continue, 0=done
                              When provided, resets hidden states at episode boundaries
        
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
        
        # Reshape for CNN: (batch * seq_len, C, H, W)
        obs_flat = obs.reshape(-1, *obs.shape[2:]).permute(0, 3, 1, 2)
        
        # Extract features
        features = self.conv(obs_flat)  # (batch * seq_len, conv_out_size)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, conv_out_size)
        
        # Process through LSTM
        if continuation_mask is not None:
            # Handle episode boundaries by processing with mask
            lstm_out, new_hidden = self._forward_with_mask(features, hidden_state, continuation_mask)
        else:
            # Standard LSTM processing (efficient for sequences without internal dones)
            lstm_out, new_hidden = self.lstm(features, hidden_state)
        
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
        Handles continuation sequences correctly: if a sequence has no dones, 
        it will maintain the hidden state from init_hidden throughout.
        
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
            return self.lstm(features, (h, c))
        
        # Process with masking (less efficient but handles episode boundaries)
        outputs = []
        for t in range(seq_len):
            # Reset hidden state where episodes ended in previous step
            if t > 0:
                # continuation_mask[:, t-1] = 0 means episode ended at t-1
                # So we reset hidden state before processing t
                mask = continuation_mask[:, t-1].view(1, batch_size, 1)  # (1, batch, 1)
                h = h * mask
                c = c * mask
            
            # Process single timestep
            out, (h, c) = self.lstm(features[:, t:t+1], (h, c))
            outputs.append(out)
        
        lstm_out = torch.cat(outputs, dim=1)
        return lstm_out, (h, c)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


class RolloutStorage:
    """
    Efficient storage for PPO rollouts preserving temporal structure.
    Stores complete episodes/trajectories for proper LSTM training.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.episodes = []
        self.current_episode = defaultdict(list)
    
    def add(self, obs: np.ndarray, action: int, log_prob: float, value: float,
            reward: float, done: bool, hidden_state: Tuple[torch.Tensor, torch.Tensor],
            next_hidden_state: Tuple[torch.Tensor, torch.Tensor], next_obs: np.ndarray):
        """
        Add a single timestep to current episode.
        
        Args:
            hidden_state: Hidden state BEFORE processing this observation
            next_hidden_state: Hidden state AFTER processing this observation
            next_obs: Next observation (for accurate GAE bootstrapping)
        """
        self.current_episode['obs'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['log_probs'].append(log_prob)
        self.current_episode['values'].append(value)
        self.current_episode['rewards'].append(reward)
        self.current_episode['dones'].append(done)
        
        # Store hidden state at this timestep (input to network)
        self.current_episode['hidden_states'].append((
            hidden_state[0].cpu().clone(),
            hidden_state[1].cpu().clone()
        ))
        
        # Always update the final hidden state and next_obs (used for bootstrapping)
        self.current_episode['final_hidden'] = (
            next_hidden_state[0].cpu().clone(),
            next_hidden_state[1].cpu().clone()
        )
        self.current_episode['next_obs'] = next_obs  # For GAE bootstrap
        
        if done:
            self.finish_episode()
    
    def finish_episode(self):
        """Mark current episode as complete and start new one."""
        if len(self.current_episode['obs']) > 0:
            self.episodes.append(dict(self.current_episode))
            self.current_episode = defaultdict(list)
    
    def get_sequences(self, max_seq_len: int = 128) -> List[Dict]:
        """
        Split episodes into fixed-length sequences for efficient training.
        Uses stored hidden states to maintain temporal continuity.
        
        Args:
            max_seq_len: Maximum sequence length for truncated BPTT
        
        Returns:
            List of sequence dictionaries with correct initial hidden states
        """
        sequences = []
        
        for episode in self.episodes:
            ep_len = len(episode['obs'])
            
            # Split episode into chunks
            for start_idx in range(0, ep_len, max_seq_len):
                end_idx = min(start_idx + max_seq_len, ep_len)
                
                # Get initial hidden state for this sequence
                # This is the hidden state that was INPUT to obs[start_idx]
                init_hidden = episode['hidden_states'][start_idx]
                
                # Get final hidden state for this sequence (for bootstrapping)
                if end_idx < ep_len:
                    # Middle of episode: use hidden state that was INPUT to obs[end_idx]
                    final_hidden = episode['hidden_states'][end_idx]
                    # Get the next observation for accurate bootstrapping
                    next_obs = episode['obs'][end_idx]
                else:
                    # End of episode: use the final hidden state after processing last obs
                    final_hidden = episode['final_hidden']
                    # Get the next observation (stored during collection)
                    next_obs = episode.get('next_obs', None)
                
                seq = {
                    'obs': np.array(episode['obs'][start_idx:end_idx]),
                    'actions': np.array(episode['actions'][start_idx:end_idx]),
                    'log_probs': np.array(episode['log_probs'][start_idx:end_idx]),
                    'values': np.array(episode['values'][start_idx:end_idx]),
                    'rewards': np.array(episode['rewards'][start_idx:end_idx]),
                    'dones': np.array(episode['dones'][start_idx:end_idx]),
                    'init_hidden': init_hidden,
                    'final_hidden': final_hidden,  # Store for efficient bootstrapping
                    'next_obs': next_obs  # For accurate GAE bootstrap value
                }
                
                sequences.append(seq)
        
        return sequences
    
    def compute_advantages(self, sequences: List[Dict], gamma: float, gae_lambda: float,
                          actor_critic: PPOLSTMActorCritic) -> List[Dict]:
        """
        Compute GAE advantages for all sequences.
        Properly handles episode boundaries with accurate bootstrapping.
        """
        for seq in sequences:
            rewards = torch.FloatTensor(seq['rewards']).to(self.device)
            values = torch.FloatTensor(seq['values']).to(self.device)
            dones = torch.FloatTensor(seq['dones']).to(self.device)
            
            # Compute next value for bootstrapping
            if seq['dones'][-1]:
                # Episode ended, no bootstrapping
                next_value = 0.0
            else:
                # Episode continues - compute value for next state
                if seq['next_obs'] is not None:
                    # We have the next observation - compute its value with correct hidden state
                    with torch.no_grad():
                        next_obs_tensor = torch.FloatTensor(seq['next_obs']).unsqueeze(0).to(self.device)
                        h, c = seq['final_hidden']
                        h, c = h.to(self.device), c.to(self.device)
                        
                        _, next_value_tensor, _ = actor_critic(next_obs_tensor, (h, c))
                        next_value = next_value_tensor.item()
                else:
                    # Fallback: use last value (less accurate but acceptable)
                    next_value = values[-1].item()
            
            # Compute GAE
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_val = next_value
                else:
                    next_val = values[t + 1]
                
                # TD error
                delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
                
                # GAE accumulation
                gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
                advantages[t] = gae
            
            returns = advantages + values
            
            seq['advantages'] = advantages.cpu().numpy()
            seq['returns'] = returns.cpu().numpy()
        
        return sequences
    
    def clear(self):
        """Clear all stored episodes."""
        self.episodes = []
        self.current_episode = defaultdict(list)


class PPOLSTMAgentNovelty:
    """
    Production-quality PPO agent with LSTM and optional curiosity-driven exploration.
    """
    
    def __init__(self, env, device: str = 'cpu', lr: float = 3e-4, 
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 ppo_epochs: int = 4, ppo_minibatch_size: int = 4,
                 ppo_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 max_seq_len: int = 128, hidden_size: int = 256,
                 clip_value_loss: bool = True,
                 # ✅ NEW: Curiosity parameters
                 use_curiosity: bool = False,
                 curiosity_approach_scale: float = 0.3,
                 curiosity_interaction_scale: float = 1.0,
                 extrinsic_reward_scale: float = 10.0,
                 intrinsic_reward_scale: float = 0.1):
        """
        Args:
            ... existing args ...
            use_curiosity: Whether to use intrinsic curiosity rewards
            curiosity_approach_scale: Scale for approach rewards (getting closer)
            curiosity_interaction_scale: Scale for interaction rewards (pickup/toggle)
        """
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.ppo_minibatch_size = ppo_minibatch_size
        self.ppo_epsilon = ppo_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.max_seq_len = max_seq_len
        self.clip_value_loss = clip_value_loss

        self.extrinsic_reward_scale = extrinsic_reward_scale
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # ✅ NEW: Initialize curiosity module
        self.use_curiosity = use_curiosity
        if self.use_curiosity:
            self.curiosity_module = NoveltyApproachReward(
                novelty_reward_scale=curiosity_approach_scale,
                interaction_reward_scale=curiosity_interaction_scale
            )
            print(f"🔍 Curiosity enabled: approach={curiosity_approach_scale}, interaction={curiosity_interaction_scale}")
        else:
            self.curiosity_module = None
            print("🔍 Curiosity disabled (baseline)")
        
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.n
        
        self.actor_critic = PPOLSTMActorCritic(
            obs_shape, num_actions, hidden_size=hidden_size
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.storage = RolloutStorage(self.device)
    
    def select_action(self, obs: np.ndarray, hidden_state: Tuple[torch.Tensor, torch.Tensor],
                     deterministic: bool = False) -> Tuple[int, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """Select action given observation and hidden state."""
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
        Collect rollout with optional curiosity-driven intrinsic rewards.
        
        ✅ IMPORTANT: Curiosity module maintains history ACROSS episodes.
        Episode reset does NOT clear interaction counts.
        """
        self.actor_critic.eval()
        
        obs, _ = self.env.reset()
        hidden_state = self.actor_critic.init_hidden(1, self.device)
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # ✅ NEW: Track curiosity statistics
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
            
            action, log_prob, value, new_hidden = self.select_action(obs, hidden_state)
            next_obs, env_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if truncated and not terminated:
                env_reward = -0.1
            
            # ✅ Compute intrinsic reward if curiosity enabled
            scaled_env_reward = env_reward * self.extrinsic_reward_scale
            total_reward = scaled_env_reward
            
            if self.use_curiosity:
                intrinsic_reward, curiosity_info = self.curiosity_module.compute_reward(
                    obs, next_obs, action,
                    terminated=terminated,
                    truncated=truncated,
                    env_reward=scaled_env_reward
                )
                intrinsic_reward = intrinsic_reward * self.intrinsic_reward_scale
                total_reward = scaled_env_reward + intrinsic_reward
                
                # Track detailed statistics
                current_intrinsic += intrinsic_reward
                current_extrinsic += env_reward
                current_scaled_extrinsic += scaled_env_reward
                current_approach += curiosity_info['approach_reward']
                current_interaction += curiosity_info['interaction_reward']
                total_interactions += curiosity_info['num_interactions']
            
            # Store transition with total reward (extrinsic + intrinsic)
            self.storage.add(obs, action, log_prob, value, total_reward, done, 
                           hidden_state, new_hidden, next_obs)
            
            current_episode_reward += total_reward
            current_episode_length += 1
            
            if done:
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
                    current_approach = 0
                    current_interaction = 0
                
                current_episode_reward = 0
                current_episode_length = 0
                
                obs, _ = self.env.reset()
                hidden_state = self.actor_critic.init_hidden(1, self.device)
                
                # ✅ IMPORTANT: Episode reset does NOT clear curiosity history
                # The curiosity module remembers interactions across episodes
                if self.use_curiosity:
                    self.curiosity_module.reset_episode()  # No-op, but explicit
            else:
                obs = next_obs
                hidden_state = new_hidden
        
        # Finish any incomplete episode
        self.storage.finish_episode()
        
        # ✅ Return enhanced statistics
        stats = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0
        }
        
        if self.use_curiosity:
            mean_intrinsic = np.mean(episode_intrinsic_rewards) if episode_intrinsic_rewards else 0
            mean_extrinsic = np.mean(episode_extrinsic_rewards) if episode_extrinsic_rewards else 0
            mean_scaled_extrinsic = np.mean(episode_scaled_extrinsic_rewards) if episode_scaled_extrinsic_rewards else 0
            mean_total = mean_intrinsic + mean_extrinsic
            
            stats.update({
                'mean_intrinsic_reward': mean_intrinsic,
                'mean_extrinsic_reward': mean_extrinsic,
                'mean_scaled_extrinsic_reward': mean_scaled_extrinsic,
                'mean_approach_reward': np.mean(episode_approach_rewards) if episode_approach_rewards else 0,
                'mean_interaction_reward': np.mean(episode_interaction_rewards) if episode_interaction_rewards else 0,
                'total_interactions': total_interactions,
                'intrinsic_ratio': mean_intrinsic / mean_total if mean_total > 0 else 0,
                # ✅ NEW: Report curiosity state
                'unique_objects_interacted': len(self.curiosity_module.novelty_tracker.interacted_objects),
                'total_interaction_count': sum(self.curiosity_module.novelty_tracker.interaction_counts.values())
            })
        
        return stats
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.
        Properly handles temporal dependencies through sequence-based training.
        
        Returns:
            Dictionary with training statistics
        """
        self.actor_critic.train()
        
        # Get sequences with correct hidden states (no recomputation needed!)
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
                
                # Forward pass with proper hidden state handling
                logits, values, _ = self.actor_critic(
                    batch['obs'],
                    batch['init_hidden'],
                    continuation_mask=batch['continuation_mask']  # Renamed for clarity
                )
                
                # Flatten for loss computation
                logits = logits[batch['valid_mask']]
                values = values[batch['valid_mask']].squeeze(-1)
                
                # Compute losses
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch['actions'][batch['valid_mask']])
                entropy = dist.entropy().mean()
                
                # PPO actor loss
                ratio = (new_log_probs - batch['old_log_probs'][batch['valid_mask']]).exp()
                advantages = batch['advantages'][batch['valid_mask']]
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss with optional clipping
                returns = batch['returns'][batch['valid_mask']]
                if self.clip_value_loss:
                    # Clip value function updates (helps stability)
                    old_values = batch_sequences[0]['values']  # Need to store old values
                    # For now, use unclipped loss - proper implementation needs old values stored
                    critic_loss = (returns - values).pow(2).mean()
                else:
                    critic_loss = (returns - values).pow(2).mean()
                
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
                num_updates += 1
        
        # Clear storage
        self.storage.clear()
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'num_sequences': len(sequences),
            'num_updates': num_updates
        }
    
    def _prepare_batch(self, sequences: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch of sequences with padding for efficient processing.
        Correctly handles continuation sequences from the same episode.
        
        Args:
            sequences: List of sequence dictionaries
        
        Returns:
            Batch dictionary with padded tensors
        """
        batch_size = len(sequences)
        max_len = max(len(seq['obs']) for seq in sequences)
        
        # Get shapes
        obs_shape = sequences[0]['obs'][0].shape
        
        # Initialize padded tensors
        obs = np.zeros((batch_size, max_len, *obs_shape), dtype=np.float32)
        actions = np.zeros((batch_size, max_len), dtype=np.int64)
        old_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
        advantages = np.zeros((batch_size, max_len), dtype=np.float32)
        returns = np.zeros((batch_size, max_len), dtype=np.float32)
        continuation_mask = np.ones((batch_size, max_len), dtype=np.float32)  # 1 = continue, 0 = done
        valid_mask = np.zeros((batch_size, max_len), dtype=bool)
        
        # Collect initial hidden states
        init_hiddens_h = []
        init_hiddens_c = []
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq['obs'])
            
            obs[i, :seq_len] = seq['obs']
            actions[i, :seq_len] = seq['actions']
            old_log_probs[i, :seq_len] = seq['log_probs']
            advantages[i, :seq_len] = seq['advantages']
            returns[i, :seq_len] = seq['returns']
            valid_mask[i, :seq_len] = True
            
            # Build continuation mask: 1 where episode continues, 0 where it ended
            # This mask is used by LSTM to reset hidden states at episode boundaries
            # When done[t]=True, mask[t]=0, which causes hidden state reset before t+1
            for t in range(seq_len):
                if seq['dones'][t]:
                    # Episode ended at timestep t
                    continuation_mask[i, t] = 0.0
                else:
                    # Episode continues
                    continuation_mask[i, t] = 1.0
            
            # Initial hidden state (correctly stored during collection)
            # This is the hidden state that was INPUT to obs[start_idx] during collection
            h, c = seq['init_hidden']
            init_hiddens_h.append(h)
            init_hiddens_c.append(c)
        
        # Stack hidden states
        init_hidden_h = torch.cat(init_hiddens_h, dim=1).to(self.device)  # (num_layers, batch, hidden)
        init_hidden_c = torch.cat(init_hiddens_c, dim=1).to(self.device)
        
        return {
            'obs': torch.FloatTensor(obs).to(self.device),
            'actions': torch.LongTensor(actions).to(self.device),
            'old_log_probs': torch.FloatTensor(old_log_probs).to(self.device),
            'advantages': torch.FloatTensor(advantages).to(self.device),
            'returns': torch.FloatTensor(returns).to(self.device),
            'continuation_mask': torch.FloatTensor(continuation_mask).to(self.device),  # Renamed for clarity
            'valid_mask': valid_mask,  # Padding mask for loss computation
            'init_hidden': (init_hidden_h, init_hidden_c)
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
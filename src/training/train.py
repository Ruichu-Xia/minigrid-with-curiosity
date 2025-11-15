import os
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.ppo_lstm import PPOLSTMAgent
from models.ride import RIDEAgent


class ExperimentTracker:
    """Dead simple tracker - just collects data and plots at the end."""
    
    def __init__(self, experiment_name: str, log_dir: str = "./logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Store data in memory
        self.data = {
            'frames': [],
            'mean_reward': [],
            'mean_length': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
        }
        
        self.total_frames = 0
        self.best_mean_reward = -float('inf')
        
        print(f"Logging to: {self.log_dir}")
    
    def log_config(self, config: Dict):
        """Save config as JSON."""
        with open(self.log_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def log(self, steps: int, mean_reward: float, mean_length: float,
            actor_loss: float, critic_loss: float, entropy: float,
            forward_dynamics_loss: Optional[float] = None,
            inverse_dynamics_loss: Optional[float] = None):
        """Log training data."""
        self.total_frames += steps
        
        self.data['frames'].append(self.total_frames)
        self.data['mean_reward'].append(mean_reward)
        self.data['mean_length'].append(mean_length)
        self.data['actor_loss'].append(actor_loss)
        self.data['critic_loss'].append(critic_loss)
        self.data['entropy'].append(entropy)
        
        # Optional RIDE-specific losses
        if forward_dynamics_loss is not None:
            if 'forward_dynamics_loss' not in self.data:
                self.data['forward_dynamics_loss'] = []
            self.data['forward_dynamics_loss'].append(forward_dynamics_loss)
        
        if inverse_dynamics_loss is not None:
            if 'inverse_dynamics_loss' not in self.data:
                self.data['inverse_dynamics_loss'] = []
            self.data['inverse_dynamics_loss'].append(inverse_dynamics_loss)
    
    def is_best_model(self, mean_reward: float) -> bool:
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            return True
        return False
    
    def save_data(self):
        """Save data as JSON."""
        with open(self.log_dir / "training_data.json", 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def plot(self):
        """Generate training plots."""
        if len(self.data['frames']) == 0:
            print("No data to plot yet!")
            return

        frames = np.array(self.data['frames'])
        
        # Create simple plot showing current progress
        plt.figure(figsize=(10, 6))
        plt.plot(frames, self.data['mean_reward'], linewidth=2.5, color='#2E86AB')
        plt.xlabel('Environment Frames', fontsize=14, fontweight='bold')
        plt.ylabel('Average Return', fontsize=14, fontweight='bold')
        plt.title(f'{self.experiment_name} - Training Progress', 
                 fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if self.best_mean_reward > -float('inf'):
            plt.axhline(y=self.best_mean_reward, color='red', linestyle='--', 
                       alpha=0.5, linewidth=2, label=f'Best: {self.best_mean_reward:.2f}')
            plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking so training can continue
        plt.pause(0.001)  # Allow plot to render

    def save_plot(self):
        if len(self.data['frames']) == 0:
            print("No data to plot!")
            return
        
        frames = np.array(self.data['frames'])
        
        # Plot 1: Comprehensive 4-panel plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Average Return
        axes[0, 0].plot(frames, self.data['mean_reward'], linewidth=2, color='#2E86AB')
        axes[0, 0].set_xlabel('Environment Frames', fontsize=12)
        axes[0, 0].set_ylabel('Average Return', fontsize=12)
        axes[0, 0].set_title('Training Performance', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        if self.best_mean_reward > -float('inf'):
            axes[0, 0].axhline(y=self.best_mean_reward, color='red', linestyle='--', 
                              alpha=0.5, label=f'Best: {self.best_mean_reward:.2f}')
            axes[0, 0].legend()
        
        # Episode Length
        axes[0, 1].plot(frames, self.data['mean_length'], linewidth=2, color='#A23B72')
        axes[0, 1].set_xlabel('Environment Frames', fontsize=12)
        axes[0, 1].set_ylabel('Average Episode Length', fontsize=12)
        axes[0, 1].set_title('Episode Length', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Losses
        axes[1, 0].plot(frames, self.data['actor_loss'], linewidth=2, 
                       label='Actor Loss', color='#F18F01')
        axes[1, 0].plot(frames, self.data['critic_loss'], linewidth=2, 
                       label='Critic Loss', color='#C73E1D')
        axes[1, 0].set_xlabel('Environment Frames', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Training Losses', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        axes[1, 1].plot(frames, self.data['entropy'], linewidth=2, color='#6A994E')
        axes[1, 1].set_xlabel('Environment Frames', fontsize=12)
        axes[1, 1].set_ylabel('Entropy', fontsize=12)
        axes[1, 1].set_title('Policy Entropy', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comprehensive_plot_path = self.log_dir / "training_curves.png"
        plt.savefig(comprehensive_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Comprehensive plot saved to: {comprehensive_plot_path}")
        
        # Plot 2: Simple return vs frames plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frames, self.data['mean_reward'], linewidth=2.5, color='#2E86AB')
        ax.set_xlabel('Environment Frames', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Return', fontsize=14, fontweight='bold')
        ax.set_title(f'{self.experiment_name} - Training Performance', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if self.best_mean_reward > -float('inf'):
            ax.axhline(y=self.best_mean_reward, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label=f'Best: {self.best_mean_reward:.2f}')
            ax.legend(fontsize=12)
        
        simple_plot_path = self.log_dir / "return_vs_frames.png"
        plt.savefig(simple_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Main plot saved to: {simple_plot_path}")


def train_ppo_lstm(
    env,
    experiment_name: str,
    num_iterations: int = 1000,
    steps_per_iteration: int = 2048,
    save_interval: int = 50,
    print_interval: int = 10, 
    checkpoint_dir: str = "../checkpoints",
    log_dir: str = "../runs",
    # Agent hyperparameters
    device: str = "cpu",
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ppo_epochs: int = 4,
    ppo_minibatch_size: int = 64,
    ppo_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 2,
    max_seq_len: int = 128,
    hidden_size: int = 256,
    clip_value_loss: bool = False,
) -> 'PPOLSTMAgent':
    """
    Train a PPO agent with LSTM on a given environment.
    
    Args:
        env: Gym environment
        experiment_name: Name for the experiment (used for logging)
        num_iterations: Number of training iterations
        steps_per_iteration: Environment steps per iteration
        save_interval: Save checkpoint every N iterations
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for experiment logs
        device: Device to train on ('cpu', 'cuda', or 'mps')
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        ppo_epochs: Number of PPO update epochs
        ppo_minibatch_size: Minibatch size for PPO updates
        ppo_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm
        max_seq_len: Maximum sequence length for TBPTT
        hidden_size: LSTM hidden size
        clip_value_loss: Whether to clip value loss
    
    Returns:
        Trained agent
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_experiment_name = f"{experiment_name}_{timestamp}"
    
    checkpoint_path = Path(checkpoint_dir) / full_experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'env_name': str(env.unwrapped.spec.id) if hasattr(env, 'spec') else 'unknown',
        'num_iterations': num_iterations,
        'steps_per_iteration': steps_per_iteration,
        'save_interval': save_interval,
        'device': device,
        'lr': lr,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'ppo_epochs': ppo_epochs,
        'ppo_minibatch_size': ppo_minibatch_size,
        'ppo_epsilon': ppo_epsilon,
        'value_coef': value_coef,
        'entropy_coef': entropy_coef,
        'max_grad_norm': max_grad_norm,
        'max_seq_len': max_seq_len,
        'hidden_size': hidden_size,
        'clip_value_loss': clip_value_loss,
    }

    # Initialize tracker
    tracker = ExperimentTracker(experiment_name=full_experiment_name, log_dir=log_dir)
    tracker.log_config(config)

    # Initialize agent
    agent = PPOLSTMAgent(
        env=env,
        device=device,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ppo_epochs=ppo_epochs,
        ppo_minibatch_size=ppo_minibatch_size,
        ppo_epsilon=ppo_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        clip_value_loss=clip_value_loss,
    )
    
    print(f"\n{'='*60}")
    print(f"Starting training: {full_experiment_name}")
    print(f"{'='*60}\n")
    
    # Training loop
    for iteration in range(num_iterations):
        # Collect rollouts
        rollout_stats = agent.collect_rollout(steps_per_iteration)
        
        # Update policy
        train_stats = agent.update()
        
        tracker.log(
            steps=steps_per_iteration,
            mean_reward=rollout_stats['mean_reward'],
            mean_length=rollout_stats['mean_length'],
            actor_loss=train_stats['actor_loss'],
            critic_loss=train_stats['critic_loss'],
            entropy=train_stats['entropy']
        )
        
        if tracker.is_best_model(rollout_stats['mean_reward']):
            agent.save(str(checkpoint_path / "best_model.pt"))
        
        if (iteration + 1) % print_interval == 0 or iteration == 0:
            print(f"[{iteration + 1:4d}/{num_iterations}] "
                  f"Frames: {tracker.total_frames:7,} | "
                  f"Reward: {rollout_stats['mean_reward']:7.2f} | "
                  f"Length: {rollout_stats['mean_length']:6.1f} | "
                  f"Loss: {train_stats['actor_loss']:.4f}/{train_stats['critic_loss']:.4f}")

            # tracker.plot()
        
        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            agent.save(str(checkpoint_path / f"checkpoint_{iteration + 1}.pt"))
        
    # Save final model
    agent.save(str(checkpoint_path / "final_model.pt"))
    
    # Save data and generate plots
    tracker.save_data()
    tracker.save_plot() 
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"  Best reward: {tracker.best_mean_reward:.2f}")
    print(f"  Total frames: {tracker.total_frames:,}")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  Plots: {tracker.log_dir}")
    print(f"{'='*60}\n")
    
    return agent


def train_ride(
    env,
    experiment_name: str,
    num_iterations: int = 1000,
    steps_per_iteration: int = 2048,
    save_interval: int = 50,
    print_interval: int = 10, 
    checkpoint_dir: str = "../checkpoints",
    log_dir: str = "../runs",
    # Agent hyperparameters
    device: str = "cpu",
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ppo_epochs: int = 4,
    ppo_minibatch_size: int = 64,
    ppo_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 2,
    max_seq_len: int = 128,
    hidden_size: int = 1024,  # Default 1024 matching original RIDE (2-layer LSTM)
    clip_value_loss: bool = False,
    # RIDE-specific hyperparameters
    intrinsic_reward_coef: float = 0.1,  # Default from RIDE paper
    forward_loss_coef: float = 0.1,  # Coefficient for forward dynamics loss
    inverse_loss_coef: float = 0.1,  # Coefficient for inverse dynamics loss
    use_intrinsic_normalization: bool = True,  # Episodic state visitation normalization
) -> 'RIDEAgent':
    """
    Train a RIDE (Rewarding Impact-Driven Exploration) agent with PPO and LSTM.
    
    RIDE adds intrinsic rewards based on state representation changes:
    r_intrinsic = ||φ(s_t) - φ(s_{t-1})||_2
    r_total = r_extrinsic + β * r_intrinsic
    
    Args:
        env: Gym environment
        experiment_name: Name for the experiment (used for logging)
        num_iterations: Number of training iterations
        steps_per_iteration: Environment steps per iteration
        save_interval: Save checkpoint every N iterations
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for experiment logs
        device: Device to train on ('cpu', 'cuda', or 'mps')
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        ppo_epochs: Number of PPO update epochs
        ppo_minibatch_size: Minibatch size for PPO updates
        ppo_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm
        max_seq_len: Maximum sequence length for TBPTT
        hidden_size: LSTM hidden size (default 1024, matching original RIDE)
        clip_value_loss: Whether to clip value loss
        intrinsic_reward_coef: Coefficient β for intrinsic rewards (default 0.1)
        forward_loss_coef: Coefficient for forward dynamics loss (default 0.1)
        inverse_loss_coef: Coefficient for inverse dynamics loss (default 0.1)
        use_intrinsic_normalization: Whether to use episodic state visitation normalization (default True)
    
    Returns:
        Trained RIDE agent
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_experiment_name = f"{experiment_name}_{timestamp}"
    
    checkpoint_path = Path(checkpoint_dir) / full_experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'env_name': str(env.unwrapped.spec.id) if hasattr(env, 'spec') else 'unknown',
        'num_iterations': num_iterations,
        'steps_per_iteration': steps_per_iteration,
        'save_interval': save_interval,
        'device': device,
        'lr': lr,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'ppo_epochs': ppo_epochs,
        'ppo_minibatch_size': ppo_minibatch_size,
        'ppo_epsilon': ppo_epsilon,
        'value_coef': value_coef,
        'entropy_coef': entropy_coef,
        'max_grad_norm': max_grad_norm,
        'max_seq_len': max_seq_len,
        'hidden_size': hidden_size,
        'clip_value_loss': clip_value_loss,
        'intrinsic_reward_coef': intrinsic_reward_coef,
        'forward_loss_coef': forward_loss_coef,
        'inverse_loss_coef': inverse_loss_coef,
        'use_intrinsic_normalization': use_intrinsic_normalization,
        'method': 'RIDE',
    }

    # Initialize tracker
    tracker = ExperimentTracker(experiment_name=full_experiment_name, log_dir=log_dir)
    tracker.log_config(config)

    # Initialize RIDE agent
    agent = RIDEAgent(
        env=env,
        device=device,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ppo_epochs=ppo_epochs,
        ppo_minibatch_size=ppo_minibatch_size,
        ppo_epsilon=ppo_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        clip_value_loss=clip_value_loss,
        intrinsic_reward_coef=intrinsic_reward_coef,
        forward_loss_coef=forward_loss_coef,
        inverse_loss_coef=inverse_loss_coef,
        use_intrinsic_normalization=use_intrinsic_normalization,
    )
    
    print(f"\n{'='*60}")
    print(f"Starting RIDE training: {full_experiment_name}")
    print(f"Intrinsic reward coefficient: {intrinsic_reward_coef}")
    print(f"{'='*60}\n")
    
    # Training loop
    for iteration in range(num_iterations):
        # Collect rollouts
        rollout_stats = agent.collect_rollout(steps_per_iteration)
        
        # Update policy
        train_stats = agent.update()
        
        tracker.log(
            steps=steps_per_iteration,
            mean_reward=rollout_stats['mean_reward'],
            mean_length=rollout_stats['mean_length'],
            actor_loss=train_stats['actor_loss'],
            critic_loss=train_stats['critic_loss'],
            entropy=train_stats['entropy'],
            forward_dynamics_loss=train_stats.get('forward_dynamics_loss', 0),
            inverse_dynamics_loss=train_stats.get('inverse_dynamics_loss', 0)
        )
        
        if tracker.is_best_model(rollout_stats['mean_reward']):
            agent.save(str(checkpoint_path / "best_model.pt"))
        
        if (iteration + 1) % print_interval == 0 or iteration == 0:
            mean_intrinsic = rollout_stats.get('mean_intrinsic_reward', 0)
            forward_loss = train_stats.get('forward_dynamics_loss', 0)
            inverse_loss = train_stats.get('inverse_dynamics_loss', 0)
            print(f"[{iteration + 1:4d}/{num_iterations}] "
                  f"Frames: {tracker.total_frames:7,} | "
                  f"Reward: {rollout_stats['mean_reward']:7.2f} | "
                  f"Intrinsic: {mean_intrinsic:6.3f} | "
                  f"Length: {rollout_stats['mean_length']:6.1f} | "
                  f"Loss: {train_stats['actor_loss']:.4f}/{train_stats['critic_loss']:.4f} | "
                  f"FD: {forward_loss:.4f} ID: {inverse_loss:.4f}")

            # tracker.plot()
        
        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            agent.save(str(checkpoint_path / f"checkpoint_{iteration + 1}.pt"))
        
    # Save final model
    agent.save(str(checkpoint_path / "final_model.pt"))
    
    # Save data and generate plots
    tracker.save_data()
    tracker.save_plot() 
    
    print(f"\n{'='*60}")
    print(f"✓ RIDE training complete!")
    print(f"  Best reward: {tracker.best_mean_reward:.2f}")
    print(f"  Total frames: {tracker.total_frames:,}")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  Plots: {tracker.log_dir}")
    print(f"{'='*60}\n")
    
    return agent


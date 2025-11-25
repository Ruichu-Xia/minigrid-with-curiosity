import json
import torch
from pathlib import Path
import time
import numpy as np
from envs.env import MiniGridEnvWrapper


def evaluate_agent(
    agent_class,
    env: MiniGridEnvWrapper,
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 10,
    max_steps_per_episode: int = 100,
    deterministic: bool = False,
    render: bool = True,
    render_delay: float = 0.01,
    device: str = None,
) -> dict[str, float]:
    """
    Evaluate the agent's performance on the environment.
    """
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    if device is not None:
        eval_device = device
    else:
        eval_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    agent = agent_class(
        env=env,
        device=eval_device,
        lr=config.get('lr', 3e-4),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        ppo_epochs=config.get('ppo_epochs', 4),
        ppo_minibatch_size=config.get('ppo_minibatch_size', 64),
        ppo_epsilon=config.get('ppo_epsilon', 0.2),
        value_coef=config.get('value_coef', 0.5),
        entropy_coef=config.get('entropy_coef', 0.01),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        max_seq_len=config.get('max_seq_len', 128),
        hidden_size=config.get('hidden_size', 256),
        clip_value_loss=config.get('clip_value_loss', True),
        **config.get('agent_kwargs', {})  # Additional agent-specific kwargs
    )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    agent.load(str(checkpoint_path))
    agent.actor_critic.eval()
    print("Checkpoint loaded successfully!")

    total_rewards = []
    total_lengths = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("="*60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        hidden_state = agent.actor_critic.init_hidden(batch_size=1, device=agent.device)
        
        episode_reward = 0
        episode_length = 0
        
        if render:
            env.display_interactive()
        
        for step in range(max_steps_per_episode):
            action, log_prob, value, hidden_state = agent.select_action(
                obs, hidden_state, deterministic=deterministic
            )
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.display_interactive()
                time.sleep(render_delay)
            
            if terminated or truncated:
                print(f"Episode {episode + 1:2d}/{num_episodes} | "
                      f"Steps: {episode_length:3d} | "
                      f"Reward: {episode_reward:6.2f}")
                break
            
            obs = next_obs
        
        if episode_length >= max_steps_per_episode:
            print(f"Episode {episode + 1:2d}/{num_episodes} | "
                  f"Steps: {episode_length:3d} (max) | "
                  f"Reward: {episode_reward:6.2f}")
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
    
    # Calculate and return results
    results = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
        'mean_length': np.mean(total_lengths),
        'std_length': np.std(total_lengths),
        'success_rate': sum(1 for r in total_rewards if r > 0) / len(total_rewards) * 100,
    }
    
    print("\n" + "="*60)
    print("Evaluation Summary:")
    print(f"  Episodes:     {results['num_episodes']}")
    print(f"  Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min/Max:      {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Mean Length:  {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print("="*60)
    
    return results
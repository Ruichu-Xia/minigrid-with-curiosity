"""
Render short GIFs of trained agents successfully solving each MiniGrid task.

Loads the best_model.pt for each environment used in the report and rolls out
the policy until it finds a successful episode (extrinsic reward > 0), then
saves the rendered frames as an animated GIF for the README.

Run from the src/ directory:  python make_gifs.py
"""
from pathlib import Path

import torch
from PIL import Image

import envs.env  # noqa: F401  (registers custom env ids: DoorKey-9x9, MultiRoom-N7-S8)
from envs.env import MiniGridEnvWrapper
from models.ppo_lstm import PPOLSTMActorCritic

# Each entry: display name -> (gym env id, checkpoint dir under checkpoints/)
TASKS = {
    "SimpleCrossing-S9N1": (
        "MiniGrid-SimpleCrossingS9N1-v0",
        "simplecrossing_ppolstm_naive_20251208_082519",
    ),
    "FourRooms": (
        "MiniGrid-FourRooms-v0",
        "fourrooms_ppolstm_naive_20251128_092826",
    ),
    "DoorKey-9x9": (
        "MiniGrid-DoorKey-9x9-v0",
        "doorkey_ppolstm_novelty_20251125_105044",
    ),
    "MultiRoom-N4-S5": (
        "MiniGrid-MultiRoom-N4-S5-v0",
        "multiroomN4S5_ppolstm_novelty_20251126_013918",
    ),
    "MultiRoom-N6": (
        "MiniGrid-MultiRoom-N6-v0",
        "multiroomN6_ppolstm_novelty_count_20251205_094046",
    ),
    "MultiRoom-N7-S8": (
        "MiniGrid-MultiRoom-N7-S8-v0",
        "multiroomN7S8_ppolstm_novelty_20251126_045456",
    ),
}

CHECKPOINT_ROOT = Path("checkpoints")
OUTPUT_DIR = Path("../assets/gifs")
MAX_STEPS = 400          # hard cap per episode attempt
MAX_ATTEMPTS = 200       # seeds to try before giving up on a success
PASSES_PER_GIF = 3       # number of successful episodes (distinct mazes) per GIF
FRAME_WIDTH = 320        # downscale rendered frames to keep GIFs small
FRAME_MS = 80            # ms per frame
HOLD_LAST_FRAMES = 12    # repeat the final (solved) frame to pause on success
GAP_FRAMES = 6           # hold the solved frame between passes as a separator
DEVICE = torch.device("cpu")


def load_policy(env, checkpoint_path):
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    net = PPOLSTMActorCritic(obs_shape, num_actions).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ckpt["actor_critic_state_dict"])
    net.eval()
    return net


def select_action(net, obs, hidden, deterministic):
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        logits, _, new_hidden = net(obs_t, hidden)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
    return action.item(), new_hidden


def resize_frame(frame):
    img = Image.fromarray(frame)
    w, h = img.size
    if w > FRAME_WIDTH:
        new_h = int(h * FRAME_WIDTH / w)
        img = img.resize((FRAME_WIDTH, new_h), Image.NEAREST)
    return img


def run_episode(net, env, seed, deterministic):
    """Roll out one episode from a given seed; return (frames, total_reward)."""
    obs, _ = env.reset(seed=seed)
    hidden = net.init_hidden(batch_size=1, device=DEVICE)
    frames = [resize_frame(env.render())]
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        action, hidden = select_action(net, obs, hidden, deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(resize_frame(env.render()))
        if terminated or truncated:
            break
    return frames, total_reward


def collect_successes(net, env, num_passes):
    """Find `num_passes` successful episodes, each from a distinct seed/maze."""
    passes = []
    seed = 0
    while len(passes) < num_passes and seed < MAX_ATTEMPTS:
        # Sampling explores; a few deterministic tries can also help.
        deterministic = seed % 5 == 0
        frames, reward = run_episode(net, env, seed, deterministic)
        if reward > 0:
            passes.append({
                "frames": frames,
                "reward": reward,
                "seed": seed,
                "deterministic": deterministic,
            })
        seed += 1
    return passes


def save_gif(passes, path):
    """Concatenate the frames of every pass into one looping GIF."""
    seq = []
    for i, p in enumerate(passes):
        seq.extend(p["frames"])
        # Hold the solved frame: short gap between passes, longer pause at the end.
        hold = HOLD_LAST_FRAMES if i == len(passes) - 1 else GAP_FRAMES
        seq.extend([p["frames"][-1]] * hold)
    seq[0].save(
        path,
        save_all=True,
        append_images=seq[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (env_id, ckpt_dir) in TASKS.items():
        ckpt_path = CHECKPOINT_ROOT / ckpt_dir / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue
        print(f"\n=== {name} ({env_id}) ===")
        env = MiniGridEnvWrapper(env_id, render_mode="rgb_array", fully_observed=False)
        net = load_policy(env, ckpt_path)
        passes = collect_successes(net, env, PASSES_PER_GIF)
        env.close()
        if len(passes) < PASSES_PER_GIF:
            print(
                f"[WARN] {name}: only {len(passes)}/{PASSES_PER_GIF} successes "
                f"within {MAX_ATTEMPTS} attempts"
            )
        if not passes:
            print(f"[FAIL] {name}: no success within {MAX_ATTEMPTS} attempts")
            results[name] = None
            continue
        out_path = OUTPUT_DIR / f"{name}.gif"
        save_gif(passes, out_path)
        kb = out_path.stat().st_size / 1024
        seeds = ", ".join(
            f"seed {p['seed']} (r={p['reward']:.2f})" for p in passes
        )
        print(f"[OK] {name}: {len(passes)} passes [{seeds}] -> {out_path} ({kb:.0f} KB)")
        results[name] = out_path
    print("\nDone. GIFs written to", OUTPUT_DIR.resolve())
    return results


if __name__ == "__main__":
    main()

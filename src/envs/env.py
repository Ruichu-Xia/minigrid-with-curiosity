import gymnasium as gym
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.wrappers import FullyObsWrapper
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt


class DoorKeyEnv9x9(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)

register(
    id='MiniGrid-DoorKey-9x9-v0',
    entry_point=__name__ + ':DoorKeyEnv9x9'
)


class EmptyEnv9x9(EmptyEnv):
    """
    An empty 9x9 grid environment.
    The agent just has to navigate to the goal.
    """
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)

register(
    id='MiniGrid-Empty-9x9-v0',
    entry_point=__name__ + ':EmptyEnv9x9'
)


class ImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

        self.fig = None
        self.ax = None
        self.im = None
        self.total_steps = 0

    def observation(self, obs):
        return obs["image"]
    
    def display_interactive(self):
        """
        Displays the current environment state in an interactive matplotlib window.
        Similar to ContinualEnv.display_interactive() but for single environments.
        """
        if self.render_mode != 'rgb_array':
            print(f"Interactive display only works with render_mode='rgb_array', but mode is '{self.render_mode}'.")
            return
        
        if self.fig is None:
            # Initialize the plot on the first call
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            img = self.render()
            self.im = self.ax.imshow(img)
            plt.axis('off')

        # Update the image data and title on every call
        new_img = self.render()
        self.im.set_data(new_img)
        self.ax.set_title(f"Environment: {self.unwrapped.spec.id}\nTotal Steps: {self.total_steps}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def step(self, action):
        """
        Override step to increment total_steps counter.
        """
        self.total_steps += 1
        return super().step(action)
    
    def reset(self, **kwargs):
        """
        Override reset to reset the total_steps counter.
        """
        self.total_steps = 0
        return super().reset(**kwargs)
    
    def close(self):
        """Closes the environment and the matplotlib figure."""
        super().close()
        if self.fig is not None:
            plt.close(self.fig)


class MiniGridEnvWrapper:
    """
    A standalone wrapper for a single task environment with optional rendering control.
    Similar to ContinualEnv but designed for a single task.
    """
    def __init__(self, env_id: str, render_mode: str = None, fully_observed: bool = False):
        """
        Initialize a single task environment wrapper.
        
        Args:
            env_id: The gymnasium environment ID (e.g., 'MiniGrid-Empty-9x9-v0')
            render_mode: The render mode for the environment ('rgb_array', 'human', or None)
            fully_observed: If True, use FullyObsWrapper to provide full grid observation
        """
        self.env_id = env_id
        self.render_mode = render_mode
        self.fully_observed = fully_observed
        
        base_env = gym.make(env_id, render_mode=self.render_mode)
        
        # Apply FullyObsWrapper if requested (gives agent full grid view)
        if fully_observed:
            base_env = FullyObsWrapper(base_env)
        
        self.env = ImgObsWrapper(base_env)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.fig = None
        self.ax = None
        self.im = None
        self.total_steps = 0
    
    def step(self, action: int):
        """
        Take a step in the environment.
        """
        self.total_steps += 1
        return self.env.step(action)
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        return self.env.reset(**kwargs)
    
    def render(self):
        """
        Render the environment.
        """
        return self.env.render()
    
    def display_interactive(self):
        """
        Displays the current environment state in an interactive matplotlib window.
        Only works if rendering is enabled and render_mode is 'rgb_array'.
        """
        if self.render_mode != 'rgb_array':
            print(f"Interactive display only works with render_mode='rgb_array', but mode is '{self.render_mode}'.")
            return
        
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            img = self.render()
            if img is None:
                print("Failed to render image.")
                return
            self.im = self.ax.imshow(img)
            plt.axis('off')

        new_img = self.render()
        if new_img is None:
            return
        self.im.set_data(new_img)
        self.ax.set_title(f"Environment: {self.env_id}\nTotal Steps: {self.total_steps}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """
        Close the environment and the matplotlib figure.
        """
        self.env.close()
        if self.fig is not None:
            plt.close(self.fig)
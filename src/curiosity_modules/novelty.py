import numpy as np
from collections import defaultdict


class NoveltyApproachReward:
    """
    Reward getting closer to objects that haven't been interacted with yet.
    
    FIXED VERSION:
    - Caches detections to avoid duplicate work
    - Tracks SAME objects across observations (not just nearest)
    - Prevents exploitation from turning back and forth
    - ✅ Rewards actual interactions (pickup, toggle)
    """
    
    def __init__(self, 
                 novelty_reward_scale: float = 0.3,
                 interaction_reward_scale: float = 1.0):  # ✅ NEW parameter
        self.detector = SymbolicDistinctivenessDetector()
        self.novelty_tracker = UninteractedObjectTracker(novelty_reward_scale)
        self.novelty_reward_scale = novelty_reward_scale
        self.interaction_reward_scale = interaction_reward_scale  # ✅ NEW
        # Agent position is constant in first-person view
        self.agent_pos = None  # Will be set on first observation
    
    def compute_reward(self, obs: np.ndarray, next_obs: np.ndarray, 
                  action: int,
                  terminated: bool = False,
                  truncated: bool = False,
                  env_reward: float = 0.0) -> tuple[float, dict]:  # ✅ NEW parameter
        """
        Compute curiosity reward for approaching AND interacting with objects.
        Goal-reaching requires: terminated=True, truncated=False, AND reward > 0.
        """
        if self.agent_pos is None:
            H, W = obs.shape[:2]
            self.agent_pos = (H // 2, W - 1)
        
        # Detect objects ONCE and cache
        detection_before = self.detector.detect_distinctive_objects(obs)
        detection_after = self.detector.detect_distinctive_objects(next_obs)
        
        # ✅ Pass ALL info including reward
        interacted_objects = self.novelty_tracker.detect_interaction_cached(
            obs, next_obs, action,
            detection_before=detection_before,
            detection_after=detection_after,
            terminated=terminated,
            truncated=truncated,
            env_reward=env_reward  # ✅ NEW: Verify positive reward
        )
        
        # Compute approach reward
        approach_reward = self._compute_approach_reward(
            detection_before, detection_after
        )
        
        # Compute interaction reward (includes goal if conditions met)
        interaction_reward = self._compute_interaction_reward(interacted_objects)
        
        # Total reward
        total_reward = approach_reward + interaction_reward
        
        # Enhanced info
        goal_reached = any(obj.get('interaction_type') == 'goal_reached' 
                        for obj in interacted_objects)
        
        info = {
            'approach_reward': approach_reward,
            'interaction_reward': interaction_reward,
            'goal_reached': goal_reached,
            'interacted_objects': [obj['object_name'] for obj in interacted_objects],
            'num_interactions': len(interacted_objects),
            'total_reward': total_reward
        }
        
        return total_reward, info
    
    def _compute_approach_reward(self, detection_before: dict, detection_after: dict) -> float:
        """
        Reward if novel objects appear closer in the next observation.
        
        FIXED: Now tracks the SAME objects across observations by signature.
        This prevents exploitation from just turning to see different objects.
        """
        # Build maps of objects by signature (type, color)
        # This allows us to track the SAME object across observations
        objects_before_by_sig = {
            (obj['object_type'], obj['color']): obj 
            for obj in detection_before['objects']
        }
        objects_after_by_sig = {
            (obj['object_type'], obj['color']): obj 
            for obj in detection_after['objects']
        }
        
        total_reward = 0.0
        
        # Only reward for objects that appear in BOTH observations
        common_signatures = (
            set(objects_before_by_sig.keys()) & 
            set(objects_after_by_sig.keys())
        )
        
        for sig in common_signatures:
            obj_before = objects_before_by_sig[sig]
            obj_after = objects_after_by_sig[sig]
            
            # Check if object still has novelty value
            novelty = self.novelty_tracker.compute_novelty_value(obj_after)
            if novelty < 0.1:
                continue  # Skip well-explored objects
            
            # Calculate distance change for THIS specific object
            dist_before = np.linalg.norm(
                np.array(obj_before['position']) - np.array(self.agent_pos)
            )
            dist_after = np.linalg.norm(
                np.array(obj_after['position']) - np.array(self.agent_pos)
            )
            
            distance_delta = dist_before - dist_after
            
            if distance_delta > 0:  # Object got closer
                # Weight by novelty (more novel = more reward)
                reward = self.novelty_reward_scale * distance_delta * novelty
                total_reward += reward
        
        return total_reward
    
    def _compute_interaction_reward(self, interacted_objects: list[dict]) -> float:
        """
        ✅ NEW: Reward for actually interacting with objects.
        
        This gives a bonus when the agent successfully:
        - Picks up an object (e.g., key)
        - Toggles an object (e.g., opens a door)
        
        The reward is scaled by the novelty of the object:
        - First time interacting: Full reward
        - Repeated interactions: Diminishing returns
        
        Args:
            interacted_objects: List of objects that were interacted with
            
        Returns:
            Total interaction reward
        """
        total_reward = 0.0
        
        for obj in interacted_objects:
            # Get novelty value for this object
            # Note: novelty is computed AFTER the interaction was recorded,
            # so we need to get the novelty from BEFORE the interaction
            # The interaction was already recorded in detect_interaction_cached,
            # so we need to account for that
            
            # Get the object signature
            signature = self.novelty_tracker.get_object_signature(obj)
            
            # Get current interaction count (already incremented by detect_interaction)
            interaction_count = self.novelty_tracker.interaction_counts[signature]
            
            # Calculate novelty as if this was BEFORE the interaction
            # We subtract 1 from the count to get pre-interaction novelty
            pre_interaction_count = interaction_count - 1
            novelty = np.exp(-0.01 * pre_interaction_count)
            
            # Give reward scaled by pre-interaction novelty
            interaction_reward = self.interaction_reward_scale * novelty
            total_reward += interaction_reward
        
        return total_reward
    
    def reset_episode(self):
        """Reset episode state (currently no-op since tracker maintains history)."""
        pass


class UninteractedObjectTracker:
    """
    Track which distinctive objects the agent has NOT yet interacted with.
    Encourage approaching novel/uninteracted objects.
    """
    
    def __init__(self, novelty_reward_scale: float = 0.3):
        """
        Args:
            novelty_reward_scale: Reward scale for approaching novel objects
        """
        self.novelty_reward_scale = novelty_reward_scale
        
        # Track which object signatures we've seen disappear (picked up) or change state (toggled)
        # Key: (object_type, color) tuple
        self.interacted_objects = set()
        
        # Count how many times we've interacted with each type
        self.interaction_counts = defaultdict(int)

        self.episode_interacted_objects = set()
        
        # OPTIMIZATION: Cache detector instances (don't create new ones each time)
        self.detector = SymbolicDistinctivenessDetector()
        
    def get_object_signature(self, obj: dict) -> tuple[int, int]:
        """
        Get signature for an object (object_type, color).
        Ignoring state for generalization across similar objects.
        """
        return (obj['object_type'], obj['color'])
    
    def is_novel(self, obj: dict) -> bool:
        """
        Check if we have NOT interacted with this object type yet.
        """
        signature = self.get_object_signature(obj)
        return signature not in self.interacted_objects
    
    def compute_novelty_value(self, obj: dict) -> float:
        """
        Compute curiosity value for this object.
        - Never interacted: value = 1.0
        - Previously interacted: decayed value based on count
        """
        signature = self.get_object_signature(obj)
        interaction_count = self.interaction_counts[signature]
        
        # Exponential decay: fewer interactions = higher curiosity value
        novelty = np.exp(-0.01 * interaction_count)
        
        return novelty
    
    def detect_interaction(self, obs_before: np.ndarray, obs_after: np.ndarray, 
                      action: int) -> list[dict]:
        """
        Optimized interaction detection with early exits and caching.
        Detects interactions by comparing observations for ANY state changes.
        More general than action-specific checks.
        
        Interactions detected:
        - Object disappeared (pickup action succeeded)
        - Object state changed (toggle action succeeded)
        
        Args:
            obs_before: Observation before action
            obs_after: Observation after action  
            action: Action taken (for logging/debugging, not required for detection)
            
        Returns:
            List of objects that were interacted with
        """
        interacted = []
        
        # OPTIMIZATION 1: Early exit for clearly non-interactive actions
        # Turn left/right (0, 1) and move forward (2) never cause state changes
        # Only pickup (3), drop (4), and toggle (5) can cause interactions
        if action in [0, 1, 2]:  # turn left, turn right, move forward
            return interacted
        
        # OPTIMIZATION 2: Quick check - if observations are identical, no interaction
        if np.array_equal(obs_before, obs_after):
            return interacted
        
        # Detect distinctive objects before and after
        detection_before = self.detector.detect_distinctive_objects(obs_before)
        detection_after = self.detector.detect_distinctive_objects(obs_after)
        
        # OPTIMIZATION 3: Early exit if no distinctive objects
        if detection_before['count'] == 0 and detection_after['count'] == 0:
            return interacted
        
        # Handle edge cases
        if detection_before['count'] == 0:
            # Only new objects appeared - this is unusual, likely not an interaction
            return interacted
        
        if detection_after['count'] == 0:
            # All objects disappeared
            for obj in detection_before['objects']:
                interacted.append(obj)
                signature = self.get_object_signature(obj)
                self.interacted_objects.add(signature)
                self.interaction_counts[signature] += 1
            return interacted
        
        # Build maps for efficient lookup
        objects_before_map = {obj['position']: obj for obj in detection_before['objects']}
        objects_after_map = {obj['position']: obj for obj in detection_after['objects']}
        
        # 1. Detect disappeared objects (pickup, consumption, etc.)
        # Only count if action was pickup (3) - otherwise might just be out of view
        disappeared_positions = set(objects_before_map.keys()) - set(objects_after_map.keys())
        if action == 3:  # pickup
            for pos in disappeared_positions:
                obj = objects_before_map[pos]
                signature = self.get_object_signature(obj)

                in_episode_set = signature in self.episode_interacted_objects
                count = self.interaction_counts[signature]
                print(f"  Pickup: {obj['object_name']} | InEpisode:{in_episode_set} | Count:{count}", end="")

                if signature not in self.episode_interacted_objects:
                    interacted.append(obj)
                    self.interacted_objects.add(signature)
                    self.interaction_counts[signature] += 1
                    self.episode_interacted_objects.add(signature)
                    print(f" → ✅ REWARDED")
                else:
                    print(f" → ❌ NOT REWARDED")
        
        # 2. Detect state changes at same position (toggle, activation, etc.)
        # Only count if action was toggle (5) - otherwise might be view change
        if action == 5:  # toggle
            common_positions = set(objects_before_map.keys()) & set(objects_after_map.keys())
            for pos in common_positions:
                obj_before = objects_before_map[pos]
                obj_after = objects_after_map[pos]
                
                # Check if same object type but different state
                if (obj_before['object_type'] == obj_after['object_type'] and
                    obj_before['signature'] != obj_after['signature']):
                    # State changed (e.g., door opened, object activated)
                    interacted.append(obj_before)
                    signature = self.get_object_signature(obj_before)
                    self.interacted_objects.add(signature)
                    self.interaction_counts[signature] += 1
        
        return interacted

    def detect_interaction_cached(self, obs_before: np.ndarray, obs_after: np.ndarray, 
                             action: int,
                             detection_before: dict | None = None,
                             detection_after: dict | None = None,
                             terminated: bool = False,
                             truncated: bool = False,
                             env_reward: float = 0.0) -> list[dict]:  # ✅ NEW: Check reward
        """
        Detect interactions including:
        - Pickup (action 3)
        - Toggle (action 5)  
        - Goal reaching (terminated=True AND reward > 0) ✅ Enhanced
        """
        interacted = []
        
        # OPTIMIZATION 1: Early exit for clearly non-interactive actions
        if action in [0, 1] and not (terminated and not truncated and env_reward > 0):
            return interacted
        
        # OPTIMIZATION 2: Quick check
        if np.array_equal(obs_before, obs_after) and not (terminated and env_reward > 0):
            return interacted
        
        # Use cached detections if provided
        if detection_before is None:
            detection_before = self.detector.detect_distinctive_objects(obs_before)
        if detection_after is None:
            detection_after = self.detector.detect_distinctive_objects(obs_after)
        
        # OPTIMIZATION 3: Early exit if no distinctive objects and no goal
        if detection_before['count'] == 0 and detection_after['count'] == 0:
            # ✅ Check for goal even if no objects detected
            if terminated and not truncated and env_reward > 0:
                # Goal was reached but not visible in current view
                # Create a synthetic goal interaction
                goal_obj = {
                    'position': (-1, -1),  # Unknown position
                    'object_type': 8,
                    'object_name': 'goal',
                    'color': 1,  # Green (typical goal color)
                    'color_name': 'green',
                    'state': 0,
                    'signature': (8, 1, 0),
                    'interaction_type': 'goal_reached'
                }
                interacted.append(goal_obj)
                signature = self.get_object_signature(goal_obj)
                self.interacted_objects.add(signature)
                self.interaction_counts[signature] += 1
            return interacted
        
        # Handle edge cases
        if detection_before['count'] == 0:
            # ✅ Still check for goal
            if terminated and not truncated and env_reward > 0:
                goal_obj = {
                    'position': (-1, -1),
                    'object_type': 8,
                    'object_name': 'goal',
                    'color': 1,
                    'color_name': 'green',
                    'state': 0,
                    'signature': (8, 1, 0),
                    'interaction_type': 'goal_reached'
                }
                interacted.append(goal_obj)
                signature = self.get_object_signature(goal_obj)
                self.interacted_objects.add(signature)
                self.interaction_counts[signature] += 1
            return interacted
        
        # Build maps for efficient lookup
        objects_before_map = {obj['position']: obj for obj in detection_before['objects']}
        objects_after_map = {obj['position']: obj for obj in detection_after['objects']}
        
        # 1. Detect disappeared objects (pickup)
        disappeared_positions = set(objects_before_map.keys()) - set(objects_after_map.keys())
        if action == 3:  # pickup
            for pos in disappeared_positions:
                obj = objects_before_map[pos]
                signature = self.get_object_signature(obj)

                in_episode_set = signature in self.episode_interacted_objects
                count = self.interaction_counts[signature]
                print(f"  Pickup: {obj['object_name']} | InEpisode:{in_episode_set} | Count:{count}", end="")

                if signature not in self.episode_interacted_objects:
                    interacted.append(obj)
                    self.interacted_objects.add(signature)
                    self.interaction_counts[signature] += 1
                    self.episode_interacted_objects.add(signature)
                    print(f" → ✅ REWARDED")
                else:
                    print(f" → ❌ NOT REWARDED")
        
        # 2. Detect state changes (toggle)
        if action == 5:  # toggle
            common_positions = set(objects_before_map.keys()) & set(objects_after_map.keys())
            for pos in common_positions:
                obj_before = objects_before_map[pos]
                obj_after = objects_after_map[pos]
                
                if (obj_before['object_type'] == obj_after['object_type'] and
                    obj_before['state'] > obj_after['state']):
                    interacted.append(obj_before)
                    signature = self.get_object_signature(obj_before)
                    self.interacted_objects.add(signature)
                    self.interaction_counts[signature] += 1
        
        # ✅ 3. Detect goal reaching (THREE CONDITIONS)
        if terminated and not truncated and env_reward > 0:
            # Episode ended successfully with positive reward = reached goal
            
            # Look for goal object in observation
            goal_found = False
            for obj in detection_before['objects']:
                if obj['object_type'] == 8:  # Goal object type
                    goal_interaction = obj.copy()
                    goal_interaction['interaction_type'] = 'goal_reached'
                    interacted.append(goal_interaction)
                    
                    signature = self.get_object_signature(obj)
                    self.interacted_objects.add(signature)
                    self.interaction_counts[signature] += 1
                    goal_found = True
                    break
            
            # If goal not visible but we got reward, still count it
            if not goal_found:
                goal_obj = {
                    'position': (-1, -1),
                    'object_type': 8,
                    'object_name': 'goal',
                    'color': 1,
                    'color_name': 'green',
                    'state': 0,
                    'signature': (8, 1, 0),
                    'interaction_type': 'goal_reached'
                }
                interacted.append(goal_obj)
                signature = self.get_object_signature(goal_obj)
                self.interacted_objects.add(signature)
                self.interaction_counts[signature] += 1
        
        return interacted
    
    def reset_episode(self):
        """
        Reset episode state.
        Keep interaction history across episodes for learning.
        """
        self.episode_interacted_objects.clear()


class SymbolicDistinctivenessDetector:
    """
    Detect distinctive objects directly from MiniGrid's symbolic encoding.
    Fixed to exclude the agent itself.
    """
    
    def __init__(self, 
                 interesting_objects=None,
                 background_objects=None):
        """
        Args:
            interesting_objects: Set of object IDs to track (None = auto-detect)
            background_objects: Set of object IDs to ignore (walls, floors, etc.)
        """
        # Default: ignore common background elements AND the agent itself
        if background_objects is None:
            self.background_objects = {
                0,  # unseen
                1,  # empty
                2,  # wall
                3,  # floor
                10, # agent (ADDED - don't treat agent as an object to interact with)
            }
        else:
            self.background_objects = set(background_objects)
        
        # Rest of the code stays the same...
        if interesting_objects is None:
            self.interesting_objects = None
        else:
            self.interesting_objects = set(interesting_objects)
        
        self.object_names = {
            0: 'unseen', 1: 'empty', 2: 'wall', 3: 'floor',
            4: 'door', 5: 'key', 6: 'ball', 7: 'box',
            8: 'goal', 9: 'lava', 10: 'agent'
        }
        
        self.color_names = {
            0: 'red', 1: 'green', 2: 'blue', 3: 'purple',
            4: 'yellow', 5: 'grey', 6: 'white'
        }
    
    def detect_distinctive_objects(self, obs: np.ndarray) -> dict:
        """
        Detect distinctive objects from symbolic observation.
        
        Args:
            obs: (H, W, 3) symbolic observation [object_type, color, state]
            
        Returns:
            Dictionary with detected objects
        """
        H, W, _ = obs.shape
        
        distinctive_objects = []
        distinctive_mask = np.zeros((H, W), dtype=bool)
        
        # Extract object types (channel 0)
        object_types = obs[:, :, 0]
        colors = obs[:, :, 1]
        states = obs[:, :, 2]
        
        for i in range(H):
            for j in range(W):
                obj_type = object_types[i, j]
                color = colors[i, j]
                state = states[i, j]
                
                # Check if this is a distinctive object
                is_distinctive = self._is_distinctive(obj_type)
                
                if is_distinctive:
                    distinctive_mask[i, j] = True
                    
                    distinctive_objects.append({
                        'position': (i, j),
                        'object_type': int(obj_type),
                        'object_name': self.object_names.get(int(obj_type), f'unknown_{obj_type}'),
                        'color': int(color),
                        'color_name': self.color_names.get(int(color), f'color_{color}'),
                        'state': int(state),
                        'signature': (int(obj_type), int(color), int(state))
                    })
        
        return {
            'mask': distinctive_mask,
            'objects': distinctive_objects,
            'count': len(distinctive_objects),
            'object_types_present': list(set(obj['object_type'] for obj in distinctive_objects))
        }
    
    def _is_distinctive(self, obj_type: int) -> bool:
        """
        Determine if an object type is distinctive.
        """
        obj_type = int(obj_type)
        
        # If we have explicit interesting objects, check membership
        if self.interesting_objects is not None:
            return obj_type in self.interesting_objects
        
        # Otherwise, anything not in background is distinctive
        return obj_type not in self.background_objects
    
    def get_object_by_type(self, detection_result: dict, object_type: int) -> list[dict]:
        """
        Get all detected objects of a specific type.
        """
        return [obj for obj in detection_result['objects'] 
                if obj['object_type'] == object_type]
    
    def get_nearest_object(self, detection_result: dict, agent_pos: tuple[int, int],
                          object_type: int | None = None) -> dict | None:
        """
        Get the nearest distinctive object to the agent.
        
        Args:
            detection_result: Result from detect_distinctive_objects
            agent_pos: Agent position (i, j)
            object_type: If specified, only consider this object type
        """
        objects = detection_result['objects']
        
        if object_type is not None:
            objects = [obj for obj in objects if obj['object_type'] == object_type]
        
        if not objects:
            return None
        
        # Compute distances
        agent_array = np.array(agent_pos)
        min_dist = float('inf')
        nearest = None
        
        for obj in objects:
            pos_array = np.array(obj['position'])
            dist = np.linalg.norm(agent_array - pos_array)
            
            if dist < min_dist:
                min_dist = dist
                nearest = obj
        
        if nearest is not None:
            nearest['distance_to_agent'] = min_dist
        
        return nearest


def get_agent_position_from_obs(obs: np.ndarray) -> tuple[int, int]:
    """
    Get agent position from observation.
    In MiniGrid's first-person view, the agent is always at:
    - Rightmost column (W-1)
    - Middle row (H//2)
    
    For 7x7 obs: row 3, column 6 (0-indexed)
    
    Args:
        obs: (H, W, 3) symbolic observation
        
    Returns:
        (row, col) position as (H//2, W-1)
    """
    H, W = obs.shape[:2]
    # Agent is at rightmost column, middle row
    return (H // 2, W - 1)
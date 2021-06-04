from .engine.emb_game import Game
from .engine.graphics.gui_visualizer import GuiVisualizer
import time

DEFAULT_CONFIG = {
    "game": {
        "num_obsts": 16,
        "num_preds": 3,
        "num_preys": 7,
        "x_limit": 12,
        "y_limit": 12,
        "obstacle_radius_bounds": [0.8, 2.0],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/100,
        "frameskip": 5
    },
    "environment": {
        "frameskip": 5,
        "time_limit": 1000
    }
}


class PredatorsAndPreysEnv:
    def __init__(self, config=DEFAULT_CONFIG, render=False):
        self.game = Game(config["game"])
        self.time_limit = config["environment"]["time_limit"]
        self.frame_skip = config["environment"]["frameskip"]
        self.predator_action_size = config["game"]["num_preds"]
        self.prey_action_size = config["game"]["num_preys"]
        self.time_left = -1
        self.world_timestep = config["game"]["world_timestep"]
        if render:
            self.visualizer = GuiVisualizer(self.game)
        else:
            self.visualizer = None
            
    def seed(self, n):
        self.game.seed(n)
        
    def step(self, predator_actions, prey_actions):
        if self.time_left < 0:
            return self.game.get_state_dict(), True

        action_dict = {
            "preys": prey_actions,
            "predators": predator_actions
        }
        
        for _ in range(self.frame_skip):
            self.game.step(action_dict)
            if self.visualizer is not None:
                to_break = self.visualizer.update()
                time.sleep(self.world_timestep * 2)
                if to_break:
                    self.time_left = -1
        
        self.time_left -= 1
        state = self.game.get_state_dict()
        is_done = True
        for prey in state["preys"]:
            is_done = is_done and not prey["is_alive"]
        is_done = is_done or self.time_left < 0
        
        reward = self.game.get_reward()
        return state, reward, is_done
    
    def reset(self):
        self.game.reset()
        self.time_left = self.time_limit
        if self.visualizer is not None:
            self.visualizer.update()
        return self.game.get_state_dict()

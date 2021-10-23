import os
import sys
import shutil
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from PIL import Image
from mapgen.env import Dungeon
from utils.random_utils import set_seed

from wrapper import Wrapper
from utils.evaluation import generate_gif
import wandb


class PPO:
    def __init__(self, env_config):
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        tune.register_env("Dungeon", lambda config: Wrapper(**config))

        self.checkpoint_root = "tmp/ppo/dungeon"
        shutil.rmtree(self.checkpoint_root, ignore_errors=True, onerror=None)

        self.ray_results = os.getenv("HOME") + "/ray_results1/"
        shutil.rmtree(self.ray_results, ignore_errors=True, onerror=None)

        self.env_config = env_config

        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = 1
        config["log_level"] = "INFO"
        config["framework"] = "torch"
        config["env"] = "Dungeon"
        config["env_config"] = self.env_config

        config["model"] = {
            "conv_filters": [
                [16, (3, 3), 2],
                [32, (3, 3), 2],
                [32, (3, 3), 1],
            ],
            "post_fcnet_hiddens": [32],
            "post_fcnet_activation": "relu",
            "vf_share_layers": False,
        }

        config["rollout_fragment_length"] = 100
        config["entropy_coeff"] = 0.1
        config["lambda"] = 0.95
        config["vf_loss_coeff"] = 1.0

        self.agent = ppo.PPOTrainer(config)

    def train(self, epochs=500, saverate=None, seed=42):
        set_seed(42)
        saverate = saverate if saverate is not None else epochs // 100

        for epoch in range(epochs):
            result = self.agent.train()

            filename = self.agent.save(self.checkpoint_root)

            print(
                f"Epoch {epoch + 1}: {result['episode_reward_min']} | {result['episode_reward_mean']} | {result['episode_reward_max']} | {result['episode_len_mean']}"
            )

            wandb.log(
                {
                    "reward_min": result["episode_reward_min"],
                    "reward_mean": result["episode_reward_mean"],
                    "reward_max": result["episode_reward_max"],
                }
            )

            if (epoch + 1) % saverate == 0:
                env = Dungeon(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
                obs = env.reset()
                Image.fromarray(env._map.render(env._agent)).convert("RGB").resize((500, 500), Image.NEAREST).save(
                    "tmp.png"
                )

                frames = []

                for _ in range(500):
                    action = self.agent.compute_single_action(obs)

                    frame = (
                        Image.fromarray(env._map.render(env._agent))
                        .convert("RGB")
                        .resize((500, 500), Image.NEAREST)
                        .quantize()
                    )
                    frames.append(frame)

                    # frame.save('tmp1.png')
                    obs, _, done, _ = env.step(action)
                    if done:
                        break

                frames[0].save(f"out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)

                wandb.log({"video": wandb.Video("out.gif", fps=30, format="gif")})

        return self

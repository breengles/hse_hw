from mapgen import Dungeon


class Wrapper(Dungeon):
    def __init__(self, *args, timepenalty=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.timepenalty = timepenalty

    def reset(self):
        observation = super().reset()
        return observation[:, :, :-1].transpose(2, 0, 1)  # as torch conv wotks on (C, H, W)

    def step(self, action: int):
        observation, metric, done, info = super().step(action)
        observation = observation[:, :, :-1]

        reward = 0
        if info["is_new"]:
            reward += 1

        if self.timepenalty:
            reward -= 0.1

        if info["collided"]:
            reward -= 0.5

        # if not info["moved"]:
        #     reward -= 1

        return observation.transpose(2, 0, 1), reward, done, metric, info

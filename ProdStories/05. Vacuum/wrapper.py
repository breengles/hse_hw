from mapgen import Dungeon


class Wrapper(Dungeon):
    def __init__(self, *args, timepenalty=False, transpose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.timepenalty = timepenalty
        self.transpose = transpose

    def reset(self):
        observation = super().reset()
        return observation.transpose(2, 0, 1) if self.transpose else observation
        # return observation[:, :, :-1].transpose(2, 0, 1)  # as torch conv wotks on (C, H, W)

    def step(self, action: int):
        observation, reward, done, info = super().step(action)

        if self.transpose:
            observation = observation.transpose(2, 0, 1)
            # observation = observation[:, :, :-1].transpose(2, 0, 1)

        if info["is_new"]:
            reward += 1

        if self.timepenalty:
            reward -= 0.1

        # if info["collided"]:
        #     reward -= 0.5

        # if not info["moved"]:
        #     reward -= 1

        return observation, reward, done, info

from mapgen import Dungeon


class Wrapper(Dungeon):
    def __init__(self, *args, timepenalty=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.timepenalty = timepenalty

    def step(self, action: int):
        observation, reward, done, info = super().step(action)

        if self.timepenalty:
            reward -= self._step / self._max_steps

        return observation, reward, done, info

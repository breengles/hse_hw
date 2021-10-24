from mapgen import Dungeon
import gym
import os
import uuid
import imageio


class Wrapper(Dungeon):
    def __init__(self, *args, timepenalty=None, transpose=False, **kwargs):
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

        if self.timepenalty is not None:
            reward -= self.timepenalty

        # if info["collided"]:
        #     reward -= 0.5

        # if not info["moved"]:
        #     reward -= 1

        return observation, reward, done, info


class VideoRecorder(gym.Wrapper):
    def __init__(self, env, video_path, size=512, fps=60, extension="mp4"):
        super().__init__(env)
        assert extension in {"mp4", "gif"}, "wrong video extension, supported only mp4 or gif"
        self.fps = fps
        self.size = size
        self.extension = extension

        os.makedirs(video_path, exist_ok=True)
        self.video_path = video_path
        self._frames = None

    def _save(self):
        assert self._frames is not None
        filename = os.path.join(self.video_path, str(uuid.uuid4()) + f".{self.extension}")
        self.filename = filename
        imageio.mimsave(filename, self._frames, fps=self.fps)

    def reset(self):
        state = self.env.reset()
        self._frames = [self.env.render(mode="rgb_array", size=self.size)]
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._frames.append(self.env.render(mode="rgb_array", size=self.size))
        if done:
            self._save()
        return state, reward, done, info

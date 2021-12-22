from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2

# https://github.com/OSLL/aido-auto-feedback/blob/815d82eaf7f36afb67ca09d2d42776935f6493a7df33da5457f54c46/dont_crush_duckie/misc/record.mp4
class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

        self.lower_bound = np.array([0, 10, 85])
        self.upper_bound = np.array([60, 255, 255])

    def solve(self):
        env = self.generated_task["env"]
        img, _, _, _ = env.step([0, 0])

        condition = True
        while condition:
            img, *_ = env.step([1, 0])
            # img in RGB
            # add here some image processing
            condition = self.get_yellow_amount(img) < 4e6
            env.render()

        env.step([0, 0])

    def get_yellow_amount(self, img):
        return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), self.lower_bound, self.upper_bound).sum()

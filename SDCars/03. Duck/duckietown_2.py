from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2


class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

        self.lower_bound = np.array([0, 10, 85])
        self.upper_bound = np.array([60, 255, 255])

        self.turn_left = 1
        self.env = self.generated_task["env"]

    def to_turn(self, img):
        return self.get_yellow_amount(img[img.shape[0] // 2 :]) > 0.1

    def get_yellow_amount(self, img):
        return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), self.lower_bound, self.upper_bound).mean() / 255.0

    def process_road(self, img):
        road = img[img.shape[0] // 2 :]
        road_gray = cv2.cvtColor(road, cv2.COLOR_RGB2GRAY)
        _, road_thresh = cv2.threshold(road_gray, 100, 1, cv2.THRESH_BINARY)
        return np.mean(road_thresh)

    def solve(self):
        img, *_ = self.env.step([0, 0])

        condition = True
        while condition:
            img, *_ = self.env.step([1, 0])

            if self.to_turn(img):
                self.turn(15, self.turn_left)
                self.move(20, 0.5)
                self.turn(30, -self.turn_left)
                self.move(20, 0.5)
                self.turn(15, self.turn_left)

                condition = False

            self.env.render()

        self.move(30, 10)
        self.env.step([0, 0])

    def move(self, steps, amount):
        for _ in range(steps):
            self.env.render()
            _ = self.env.step(np.array([amount, 0]))

    def turn(self, steps, amount):
        for _ in range(steps):
            self.env.render()
            _ = self.env.step(np.array([0, amount]))

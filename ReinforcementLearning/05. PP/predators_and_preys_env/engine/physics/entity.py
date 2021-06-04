import numpy as np


class Entity:
    def __init__(self, radius, speed, position_x, position_y):
        self.position = np.array([position_x, position_y])
        self.radius = radius
        self.speed = speed

    def move(self, angle, timestep):
        # pi --> 2 pi
        # a [-1, 1] --> [0, 1]
        self.position = self.position + np.array([self.speed * np.cos(2 * np.pi * angle) * timestep,
                                                  self.speed * np.sin(2 * np.pi * angle) * timestep])

    def center_distance(self, other):
        return np.linalg.norm(self.position - other.position)

    def real_distance(self, other):
        return np.linalg.norm(self.position - other.position) - (self.radius + other.radius)

    def is_intersect(self, other):
        return self.center_distance(other) <= (self.radius + other.radius)

    def force_not_intersect(self, other):
        if self.is_intersect(other):
            v = (self.position - other.position) / (self.center_distance(other) + 1e-8)
            v *= (other.radius + self.radius) * (1 + 1e-2)
            self.position = other.position + v
            return True
        else:
            return False

    def force_clip_positon(self, min_x, min_y, max_x, max_y):
        self.position = np.clip(self.position, [self.radius + min_x, self.radius + min_y],
                                [max_x - self.radius, max_y - self.radius])
import numpy as np
from models.point2d import Point2D

class Dataset:
    def __init__(self, points):
        self.points = points

    @classmethod
    def generate_sample(cls):
        class_a = [
            Point2D(1.0, 1.0, -1), Point2D(1.6, 1.4, -1), Point2D(1.2, 2.1, -1),
            Point2D(2.0, 1.7, -1), Point2D(2.4, 2.2, -1), Point2D(2.7, 1.2, -1),
            Point2D(3.0, 2.7, -1)
        ]
        class_b = [
            Point2D(5.6, 5.5, 1), Point2D(6.1, 5.1, 1), Point2D(6.3, 6.0, 1),
            Point2D(7.0, 5.7, 1), Point2D(7.2, 6.4, 1), Point2D(5.5, 6.5, 1),
            Point2D(6.7, 7.1, 1)
        ]
        return cls(class_a + class_b)

    def to_numpy(self):
        x = np.array([p.as_vector() for p in self.points], dtype=float)
        y = np.array([p.label for p in self.points], dtype=float)
        return x, y

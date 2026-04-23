from dataclasses import dataclass

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    label: int

    def as_vector(self):
        return [self.x, self.y]

from enum import IntEnum
from shapely.geometry import Polygon

from typing import Tuple

class ConfidenceLevel(IntEnum):
    """Represents a confidence level for fire data.

    Usually nominal or high confidence data are useful.
    """
    LOW = 0
    NOMINAL = 1
    HIGH = 2

    @classmethod
    def from_char(cls, c):
        if c in 'lL':
            return cls.LOW
        if c in 'nN':
            return cls.NOMINAL
        if c in 'hH':
            return cls.HIGH

        raise ValueError(f'unknown character for fire confidence: {c}')


def extents_to_polygon(extents: Tuple[float, float, float, float]) -> Polygon:
    x1, x2, y1, y2 = extents
    return Polygon([
        (x1, y1),
        (x1, y2),
        (x2, y2),
        (x2, y1),
        (x1, y1)
    ])

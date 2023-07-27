import numpy as np

from .get_spectrum_amplitude import get_spectrum_amplitude
from .readwav import readwav
from .estimate import estimate, positioning_2d
from .create_db import create_db


class SoundDB:
    def __init__(self, sample_dir, interval=0.100, dim=3):
        self.dimension = dim
        self.interval = interval
        self.db = create_db(sample_dir, interval=interval, dimension=dim)

    def positioning(self, file, output="rect"):
        if self.dimension == 2:
            return positioning_2d(self.db[0], self.db[1], file, self.interval, output)
        return estimate(self.db[0], self.db[1], file, self.interval, output)

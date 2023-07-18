import numpy as np

from .get_spectrum_amplitude import get_spectrum_amplitude
from .readwav import readwav
from .estimate import estimate
from .create_db import create_db


class SoundDB:
    def __init__(self, sample_dir):
        self.db = create_db(sample_dir)

    def positioning(self, file, output="rect"):
        return estimate(self.db[0], self.db[1], file, output)

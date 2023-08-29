import numpy as np

from .get_spectrum_amplitude import get_spectrum_amplitude
from .readwav import readwav
from .estimate import (
    estimate,
    positioning_2d,
    positioning_mic_revision,
    positioning_ampli_revision,
)
from .create_db import create_db, create_mic_revision_db, create_ampli_revision_db


class SoundDB:
    def __init__(self, sample_dir, interval=0.100, dim=3):
        self.dimension = dim
        self.interval = interval
        self.db = create_db(sample_dir, interval=interval, dimension=dim)

    def positioning(self, file, output="rect"):
        if self.dimension == 2:
            return positioning_2d(self.db[0], self.db[1], file, self.interval, output)
        return estimate(self.db[0], self.db[1], file, self.interval, output)


class MicRevisionDB:
    """マイクの位置補正を行うためのデータベースを作成するクラス

    Attributes
    ----------
    db : tuple
        作成した方位角、マイクの角度ごとのスペクトル,アンプリチュードの参照データベース

    Methods
    -------
    positioning(file, output="rect")
        測位を行う
    """

    def __init__(self, speaker_dir, mic_dir, interval=0.2):
        self.interval = interval
        self.db = create_mic_revision_db(speaker_dir, mic_dir, interval=interval)

    def positioning(self, file, output="rect"):
        return positioning_mic_revision(
            self.db[0], self.db[1], file, self.interval, output
        )


class AmpliRevisionDB:
    """マイクの角度によるアンプリチュード補正を行うためのデータベースを作成するクラス

    Attributes
    ----------
    db : tuple
        作成した方位角、マイクの角度ごとのスペクトル,アンプリチュードの参照データベース

    Methods
    -------
    positioning(file, output="rect")
        測位を行う
    """

    def __init__(self, speaker_dir, mic_dir, interval=0.2):
        self.interval = interval
        self.db = create_ampli_revision_db(speaker_dir, mic_dir, interval=interval)

    def positioning(self, file, output="rect"):
        return positioning_ampli_revision(
            self.db[0], self.db[1], file, self.interval, output
        )

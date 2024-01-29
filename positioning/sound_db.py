import numpy as np

from .get_spectrum_amplitude import (
    get_spectrum_amplitude,
    get_tukey_spectrum_amplitude,
    get_tukey_spectrum,
    get_reflect_ceiling_tdoa,
)
from .readwav import readwav
from .estimate import (
    estimate,
    estimate_distance,
    positioning_2d,
    positioning_tukey,
    positioning_mic_revision,
    positioning_ampli_revision,
    positioning_reflect_ceiling,
    estimate_direction_3d,
    positioning_direction_reflect_tdoa,
)
from .create_db import (
    create_db,
    create_tukey_db,
    create_mic_revision_db,
    create_ampli_revision_db,
    create_reflect_ceiling_db,
    create_3d_spectrum_db,
)


class SoundDB:
    def __init__(
        self,
        sample_dir,
        first_freq: int = 4000,
        last_freq: int = 13000,
        interval=0.100,
        dim=3,
    ):
        self.dimension = dim
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.interval = interval
        self.db = create_db(
            sample_dir,
            first_freq=first_freq,
            last_freq=last_freq,
            interval=interval,
            dimension=dim,
        )

    def positioning(self, file, output="rect"):
        if self.dimension == 2:
            return positioning_2d(
                self.db[0],
                self.db[1],
                file,
                first_freq=self.first_freq,
                last_freq=self.last_freq,
                interval=self.interval,
                output=output,
            )
        return estimate(self.db[0], self.db[1], file, self.interval, output)

    def estimate_distance(self, file, direction):
        test_spec, test_ampli = get_spectrum_amplitude(
            file,
            first_freq=self.first_freq,
            last_freq=self.last_freq,
            interval_length=self.interval,
        )
        return estimate_distance(self.db[1], test_ampli, direction)


class TukeyDB:
    def __init__(
        self,
        sample_dir,
        first_freq: int = 15000,
        last_freq: int = 22000,
        interval=0.2,
    ):
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.interval = interval
        self.db = create_tukey_db(
            sample_dir,
            first_freq=first_freq,
            last_freq=last_freq,
            interval=interval,
        )

    def positioning(self, file, output="rect"):
        return positioning_tukey(
            self.db[0],
            self.db[1],
            file,
            first_freq=self.first_freq,
            last_freq=self.last_freq,
            interval=self.interval,
            output=output,
        )

    def estimate_distance(self, file, direction):
        test_spec, test_ampli = get_tukey_spectrum_amplitude(
            file,
            first_freq=self.first_freq,
            last_freq=self.last_freq,
            interval_length=self.interval,
        )
        return estimate_distance(self.db[1], test_ampli, direction)


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

    def __init__(
        self, speaker_dir, mic_dir, first_freq=4000, last_freq=13000, interval=0.2
    ):
        self.interval = interval
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.db = create_mic_revision_db(
            speaker_dir,
            mic_dir,
            first_freq=first_freq,
            last_freq=last_freq,
            interval=interval,
        )

    def positioning(self, file, output="rect"):
        return positioning_mic_revision(
            self.db[0],
            self.db[1],
            file,
            self.first_freq,
            self.last_freq,
            self.interval,
            output,
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


class ReflectCeilingDB:
    def __init__(
        self,
        sample_dir,
        first_freq: int = 15000,
        last_freq: int = 22000,
        interval=0.2,
        signal_length=0.003,
    ):
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.interval = interval
        self.signal_length = signal_length
        self.db = create_reflect_ceiling_db(
            sample_dir,
            first_freq=first_freq,
            last_freq=last_freq,
            interval=interval,
            signal_length=signal_length,
        )

    def positioning(self, file):
        return positioning_reflect_ceiling(
            self.db,
            file,
            first_freq=self.first_freq,
            last_freq=self.last_freq,
            interval=self.interval,
            signal_length=self.signal_length,
        )


class CeilingTDoADB:
    """天井反射のTDoAを使った測位を行うためのデータベースを作成するクラス"""

    def __init__(
        self,
        sample_dir,
        first_freq: int = 1000,
        last_freq: int = 24000,
        interval=0.1,
        signal_length=0.001,
    ):
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.interval = interval
        self.signal_length = signal_length
        self.db = create_3d_spectrum_db(
            sample_dir,
            first_freq=first_freq,
            last_freq=last_freq,
            interval=interval,
            signal_length=signal_length,
        )

    def positioning(self, file):
        direction = estimate_direction_3d(
            self.db,
            file,
            first_freq=self.first_freq,
            last_freq=self.last_freq,
            interval=self.interval,
            signal_length=self.signal_length,
        )
        tdoa = get_reflect_ceiling_tdoa(
            file,
            first_freq=self.first_freq,
            last_freq=self.last_freq,
            interval_length=self.interval,
            signal_length=self.signal_length,
        )
        position = positioning_direction_reflect_tdoa(*direction, tdoa)
        return np.concatenate([position, direction, [tdoa]])

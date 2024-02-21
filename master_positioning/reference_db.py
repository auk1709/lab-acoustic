import numpy as np
from .create_db import create_3d_spectrum_db
from .estimate import (
    estimate_direction_3d,
    get_reflect_ceiling_tdoa,
    positioning_direction_reflect_tdoa,
)


class CeilingTDoADB:
    """天井反射のTDoAを使った測位を行うためのデータベースを作成するクラス

    Parameters
    ----------
    sample_dir : string
        計測データファイルが入ってるディレクトリの場所
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval : float
        チャープのバンド間の間隔(s)
    signal_length : float
        チャープ一発の信号長

    Attributes
    ----------
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval : float
        チャープのバンド間の間隔(s)
    signal_length : float
        チャープ一発の信号長
    db : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース

    Methods
    -------
    positioning(file)
        測位を行う
    """

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
        """測位を行う

        Parameters
        ----------
        file : string
            測位を行う音声ファイルの場所

        Returns
        -------
        NDArray
            推定した位置, 方向, TDoA
        """
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

import numpy as np
import pandas as pd
from .readwav import readwav
from .get_spectrum import get_tukey_spectrum


def create_3d_spectrum_db(
    sample_dir,
    first_freq: int = 1000,
    last_freq: int = 24000,
    interval: float = 0.1,
    signal_length: float = 0.001,
):
    """スピーカーの方位特性のデータベースとなる配列を生成する
    Tukey窓をかけたチャープを用いる
    3次元で方位のみ、距離方向は床面反射から推定
    elevationは20~50°

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

    Returns
    -------
    NDArray[]
        方位角、仰角ごとのスペクトル
    """

    sampling_rate = 48000  # マイクのサンプリングレート(Hz)
    len_chirp_sample = int(sampling_rate * signal_length)  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    count_azimuth_sample = 81  # 方位角方向のサンプル数
    count_elevation_sample = 31  # 仰角方向のサンプル数
    azimuth_degs = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    elevation_degs = np.arange(20, 51, 10)
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    # numpy配列の初期化
    # 各方向ごとのスペクトルを対応するインデックスに格納する配列, nanで初期化
    # pandasのinterpolateを使用するためにnanで初期化
    # あまり美しくないが、一旦これで
    spectrum_db = np.full(
        (
            count_azimuth_sample,
            count_elevation_sample,
            band_freq_index_range * len(band_freqs),
        ),
        np.nan,
    )
    for azi in azimuth_degs:
        for ele in elevation_degs:
            sample = readwav(f"{sample_dir}/a{str(azi)}e{str(ele)}.wav")
            if sample.ndim > 1:
                sample = sample[:, 1]  # ステレオの場合は1chのみを使用
            spectrum_db[
                azi - azimuth_degs[0], ele - elevation_degs[0], :
            ] = get_tukey_spectrum(
                sample[:240000],
                first_freq=first_freq,
                last_freq=last_freq,
                interval_length=interval,
                signal_length=signal_length,
            )  # 各方向のスペクトルを取得
    # 補間処理
    # pandasのinterpolateを使用, 正直ふさわしい使い方じゃないが、一旦これで
    # 方位角方向の補間
    for ele in elevation_degs:
        spectrum_db[:, ele - elevation_degs[0], :] = (
            pd.DataFrame(spectrum_db[:, ele - elevation_degs[0], :])
            .interpolate("akima")
            .to_numpy()
        )
    # 仰角方向の補間
    for i in range(count_azimuth_sample):
        spectrum_db[i, :, :] = (
            pd.DataFrame(spectrum_db[i, :, :]).interpolate("akima").to_numpy()
        )
    return spectrum_db

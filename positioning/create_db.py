import numpy as np
import pandas as pd
from .readwav import readwav
from .get_spectrum_amplitude import get_spectrum_amplitude


def create_db(sample_dir, interval: float = 0.100, dimension: int = 3):
    """スピーカーの方位、距離特性のデータベースとなる配列を生成する

    Parameters
    ----------
    sample_dir : string
        計測データファイルが入ってるディレクトリの場所
    interval : float
        チャープのバンド間の間隔(s)
    dimension : int [2 or 3]
        測位の次元、2次元か3次元か

    Returns
    -------
    NDArray[]
        方位角、仰角ごとのスペクトル
    NDArray
        方位角、仰角ごとの振幅
    """

    # 方位角-40~40°, 仰角0~50°, スペクトル6個×10発
    sampling_rate = 48000  # マイクのサンプリングレート(Hz)
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    count_azimuth_sample = 81  # 方位角方向のサンプル数
    count_elevation_sample = 51  # 仰角方向のサンプル数
    speaker_heights = [0, 10, 20, 30, 40, 50]
    mic_degs = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    band_freqs = np.arange(4000, 13000, chirp_width)  # 送信する周波数のバンド
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    if dimension == 2:
        sound_db = np.full(
            (count_azimuth_sample, band_freq_index_range * len(band_freqs)), np.nan
        )
        ampli_db = np.full((count_azimuth_sample), np.nan)
        for mic_deg in mic_degs:
            sample = readwav(f"{sample_dir}/a{str(mic_deg)}.wav")
            sound_db[mic_deg + 40, :], ampli_db[mic_deg + 40] = get_spectrum_amplitude(
                sample, interval_length=interval
            )
        sound_db_complete = pd.DataFrame(sound_db).interpolate("akima").to_numpy()
        ampli_db_complete = pd.DataFrame(ampli_db).interpolate("akima").to_numpy()
        return sound_db_complete, ampli_db_complete
    else:
        # 角度ごとのスペクトルのデータベース
        sound_db = np.full(
            (
                count_azimuth_sample,
                count_elevation_sample,
                band_freq_index_range * len(band_freqs),
            ),
            np.nan,
        )
        # 角度ごとの振幅のデータベース
        ampli_db = np.full(
            (
                count_azimuth_sample,
                count_elevation_sample,
            ),
            np.nan,
        )
        for mic_deg in mic_degs:
            for speaker_height in speaker_heights:
                # wav読み込み
                sound_sample = readwav(
                    f"{sample_dir}/s{str(speaker_height)}m{str(mic_deg)}.wav"
                )[480000:]
                (
                    sound_db[mic_deg + 40, speaker_height, :],
                    ampli_db[mic_deg + 40, speaker_height],
                ) = get_spectrum_amplitude(sound_sample, interval_length=interval)

        # ここから秋間補間，マイク方位で補間した後スピーカ高さで補間
        Akima = np.full(
            (
                count_azimuth_sample,
                count_elevation_sample,
                band_freq_index_range * len(band_freqs),
            ),
            np.nan,
        )
        # マイク方位で補間, 水平方向, 今ある仰角の計測点分やる
        for speaker_height in speaker_heights:
            sound_db_d = pd.DataFrame(sound_db[:, speaker_height, :])
            sound_db_d.astype("float64")
            Akima[:, speaker_height, :] = sound_db_d.interpolate("akima")

        # スピーカ高さで補間, 高さ方向, 全方位角に対して
        for i in range(0, count_azimuth_sample):
            sound_db_d = pd.DataFrame(Akima[i, :, :])
            sound_db_d.astype("float64")
            Akima[i, :, :] = sound_db_d.interpolate("akima")

        # 振幅の補間
        df_ampli = pd.DataFrame(ampli_db)  # 補間のためにpandasのデータフレームに変換
        df_ampli_azimuth = df_ampli.interpolate("akima", axis=1)  # 秋間補間, 水平方向
        df_ampli_complete = df_ampli_azimuth.interpolate("akima", axis=0)  # 秋間補間, 高さ方向

        return Akima, df_ampli_complete.values

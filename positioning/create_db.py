import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from .readwav import readwav
from .get_spectrum_amplitude import (
    get_spectrum_amplitude,
    get_tukey_spectrum_amplitude,
    get_spec_ampli_noise,
)


def create_db(
    sample_dir,
    first_freq: int = 4000,
    last_freq: int = 13000,
    interval: float = 0.100,
    dimension: int = 3,
):
    """スピーカーの方位、距離特性のデータベースとなる配列を生成する

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
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    if dimension == 2:
        sound_db = np.full(
            (count_azimuth_sample, band_freq_index_range * len(band_freqs)), np.nan
        )
        ampli_db = np.full(count_azimuth_sample, np.nan)
        for mic_deg in mic_degs:
            sample = readwav(f"{sample_dir}/a{str(mic_deg)}.wav")
            sound_db[mic_deg + 40, :], ampli_db[mic_deg + 40] = get_spectrum_amplitude(
                sample,
                first_freq=first_freq,
                last_freq=last_freq,
                interval_length=interval,
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


def create_tukey_db(
    sample_dir,
    first_freq: int = 15000,
    last_freq: int = 22000,
    interval: float = 0.2,
):
    """スピーカーの方位、距離特性のデータベースとなる配列を生成する
    Tukey窓をかけたチャープを用いる

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

    Returns
    -------
    NDArray[]
        方位角、仰角ごとのスペクトル
    NDArray
        方位角、仰角ごとの振幅
    """

    sampling_rate = 48000  # マイクのサンプリングレート(Hz)
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    count_azimuth_sample = 81  # 方位角方向のサンプル数
    mic_degs = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    sound_db = np.full(
        (count_azimuth_sample, band_freq_index_range * len(band_freqs)), np.nan
    )
    ampli_db = np.full(count_azimuth_sample, np.nan)
    for mic_deg in mic_degs:
        sample = readwav(f"{sample_dir}/a{str(mic_deg)}.wav")
        if sample.ndim > 1:
            sample = sample[:, 1]
        (
            sound_db[mic_deg + 40, :],
            ampli_db[mic_deg + 40],
        ) = get_tukey_spectrum_amplitude(
            sample,
            first_freq=first_freq,
            last_freq=last_freq,
            interval_length=interval,
            ampli_band="all",
        )
    sound_db_complete = pd.DataFrame(sound_db).interpolate("akima").to_numpy()
    ampli_db_complete = pd.DataFrame(ampli_db).interpolate("akima").to_numpy()
    return sound_db_complete, ampli_db_complete


def create_mic_revision_db(
    speaker_dir, mic_dir, first_freq: int = 4000, last_freq: int = 13000, interval=0.2
):
    """マイクの角度の補正をした2次元測位のデータベースを作成する
    スピーカー、マイクのスペクトルは周波数領域で掛け算

    Parameters
    ----------
    speaker_dir : string
        スピーカの角度ごとの音声データが入ってるディレクトリの場所
    mic_dir : string
        マイクの角度ごとの音声データが入ってるディレクトリの場所
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval : float
        チャープのバンド間の間隔(s)
    """

    sampling_rate = 48000  # マイクのサンプリングレート(Hz)
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    count_azimuth_sample = 81  # 方位角方向のサンプル数
    azimuth_points = np.arange(-40, 50, 10)
    mic_angles = np.arange(0, 91, 10)
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    # スピーカーの角度ごとのスペクトルと振幅を取得
    speaker_spec = np.empty((0, len(band_freqs), len_chirp_sample))
    speaker_ampli = np.array([])
    for azimuth in azimuth_points:
        signal = readwav(f"{speaker_dir}/a{str(azimuth)}.wav")
        spec, ampli = get_spectrum_amplitude(
            signal,
            first_freq=first_freq,
            last_freq=last_freq,
            interval_length=interval,
            ret_spec="all",
        )  # 各バンドごとのスペクトルと振幅を取得
        speaker_spec = np.vstack((speaker_spec, [spec]))
        speaker_ampli = np.append(speaker_ampli, ampli)

    # マイクの角度ごとのスペクトルと振幅を取得
    mic_spec = np.empty((0, len(band_freqs), len_chirp_sample))
    mic_ampli = np.array([])
    for angle in range(0, 91, 10):
        signal = readwav(f"{mic_dir}/angle{str(angle)}.wav")
        spec, ampli = get_spectrum_amplitude(
            signal,
            first_freq=first_freq,
            last_freq=last_freq,
            interval_length=interval,
            ret_spec="all",
        )
        mic_spec = np.vstack((mic_spec, [spec]))
        mic_ampli = np.append(mic_ampli, ampli)

    # マイクのスペクトルを0°基準で割り算（マイクの特性だけを取り出す）
    mic_spec = mic_spec / speaker_spec[4, :, :]

    # スピーカーとマイクのスペクトルを掛け算
    mix_spec = np.empty((0, len(mic_spec), len(band_freqs), len_chirp_sample))
    for s_spec in speaker_spec:
        mix = np.empty((0, len(band_freqs), len_chirp_sample))
        for m_spec in mic_spec:
            x_spec = np.multiply(s_spec, m_spec)
            mix = np.vstack((mix, [x_spec]))
        mix_spec = np.vstack((mix_spec, [mix]))

    # かけあわせたものからスペクトルのパターンを抽出
    pattern_spec = np.empty(
        (0, len(mic_angles), len(band_freqs) * band_freq_index_range)
    )
    for azimuth_spec in mix_spec:
        azimuth_pattern = np.empty((0, len(band_freqs) * band_freq_index_range))
        for spec in azimuth_spec:
            pattern = np.array([])
            for band_spec, freq in zip(spec, band_freqs):
                band_index = int(freq // fft_freq_rate)
                pattern = np.append(
                    pattern,
                    np.abs(band_spec)[band_index : band_index + band_freq_index_range],
                )
            azimuth_pattern = np.vstack([azimuth_pattern, (pattern / np.max(pattern))])
        pattern_spec = np.vstack((pattern_spec, [azimuth_pattern]))

    # 検証用、スピーカー側のみ、スペクトルのパターンを抽出
    # azimuth_pattern = np.empty((0, len(band_freqs) * band_freq_index_range))
    # for spec in speaker_spec:
    #     pattern = np.array([])
    #     for band_spec, freq in zip(spec, band_freqs):
    #         band_index = int(freq // fft_freq_rate)
    #         pattern = np.append(
    #             pattern,
    #             np.abs(band_spec)[band_index : band_index + band_freq_index_range],
    #         )
    #     azimuth_pattern = np.vstack([azimuth_pattern, (pattern / np.max(pattern))])

    # amplitudeの計算
    normalized_spk_ampli = speaker_ampli / speaker_ampli[4]
    mix_ampli = np.empty((0, len(mic_ampli)))
    for s_ampli in normalized_spk_ampli:
        mix = np.array([])
        for m_ampli in mic_ampli:
            x_ampli = s_ampli * m_ampli
            mix = np.append(mix, x_ampli)
        mix_ampli = np.vstack([mix_ampli, mix])

    # 補間
    all_azimuth = np.arange(-40, 41)
    all_mic_angles = np.arange(0, 91)
    mic_interp_spec = np.empty(
        (0, len(all_mic_angles), len(band_freqs) * band_freq_index_range)
    )
    for azimuth_spec in pattern_spec:
        pattern = np.empty((len(all_mic_angles), 0))
        for i in range(len(azimuth_spec[0, :])):
            akima = interpolate.Akima1DInterpolator(mic_angles, azimuth_spec[:, i])
            tmp = akima(all_mic_angles).reshape(-1, 1)
            pattern = np.hstack([pattern, tmp])
        mic_interp_spec = np.vstack((mic_interp_spec, [pattern]))
    perfect_spec = np.empty(
        (len(all_azimuth), 0, len(band_freqs) * band_freq_index_range)
    )
    for i in range(len(all_mic_angles)):
        pattern = np.empty((len(all_azimuth), 0))
        for j in range(len(mic_interp_spec[0, 0, :])):
            akima = interpolate.Akima1DInterpolator(
                azimuth_points, mic_interp_spec[:, i, j]
            )
            tmp = akima(all_azimuth).reshape(-1, 1)
            pattern = np.hstack([pattern, tmp])
        pattern = pattern.reshape(len(all_azimuth), 1, 36)
        perfect_spec = np.hstack((perfect_spec, pattern))

    mic_interp_ampli = np.empty((0, len(all_mic_angles)))
    for azimuth_ampli in mix_ampli:
        akima = interpolate.Akima1DInterpolator(mic_angles, azimuth_ampli)
        tmp = akima(all_mic_angles)
        mic_interp_ampli = np.vstack([mic_interp_ampli, tmp])
    perfect_ampli = np.empty((len(all_azimuth), 0))
    for i in range(len(all_mic_angles)):
        akima = interpolate.Akima1DInterpolator(azimuth_points, mic_interp_ampli[:, i])
        tmp = akima(all_azimuth).reshape(-1, 1)
        perfect_ampli = np.hstack([perfect_ampli, tmp])

    # 検証用、スピーカー側のみ、補間
    # all_azimuth = np.arange(-40, 41)
    # perfect_spec = np.empty((len(all_azimuth), 0))
    # for i in range(len(azimuth_pattern[0, :])):
    #     akima = interpolate.Akima1DInterpolator(azimuth_points, azimuth_pattern[:, i])
    #     tmp = akima(all_azimuth).reshape(-1, 1)
    #     perfect_spec = np.hstack([perfect_spec, tmp])
    # ampli_akima = interpolate.Akima1DInterpolator(azimuth_points, speaker_ampli)
    # perfect_ampli = ampli_akima(all_azimuth)

    return perfect_spec, perfect_ampli


def create_ampli_revision_db(speaker_dir, mic_dir, interval=0.2):
    """マイクの角度からアンプリチュードのみ補正をした2次元測位のデータベースを作成する
    マイクは常にまっすぐと想定

    Parameters
    ----------
    speaker_dir : string
        スピーカの角度ごとの音声データが入ってるディレクトリの場所
    mic_dir : string
        マイクの角度ごとの音声データが入ってるディレクトリの場所
    interval : float
        チャープのバンド間の間隔(s)
    """

    sampling_rate = 48000  # マイクのサンプリングレート(Hz)
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    count_azimuth_sample = 81  # 方位角方向のサンプル数
    azimuth_points = np.arange(-40, 50, 10)
    mic_angles = np.arange(0, 91, 10)
    band_freqs = np.arange(4000, 13000, chirp_width)  # 送信する周波数のバンド
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    # スピーカーの角度ごとのスペクトルと振幅を取得
    speaker_spec = np.empty((0, len(band_freqs), len_chirp_sample))
    speaker_ampli = np.array([])
    for azimuth in azimuth_points:
        signal = readwav(f"{speaker_dir}/a{str(azimuth)}.wav")
        spec, ampli = get_spectrum_amplitude(
            signal, interval_length=interval, ret_spec="all"
        )  # 各バンドごとのスペクトルと振幅を取得
        speaker_spec = np.vstack((speaker_spec, [spec]))
        speaker_ampli = np.append(speaker_ampli, ampli)

    # マイクの角度ごとのスペクトルと振幅を取得
    mic_spec = np.empty((0, len(band_freqs), len_chirp_sample))
    mic_ampli = np.array([])
    for angle in range(0, 91, 10):
        signal = readwav(f"{mic_dir}/angle{str(angle)}.wav")
        spec, ampli = get_spectrum_amplitude(
            signal, interval_length=interval, ret_spec="all"
        )
        mic_spec = np.vstack((mic_spec, [spec]))
        mic_ampli = np.append(mic_ampli, ampli)

    # スピーカー側のみ、スペクトルのパターンを抽出
    azimuth_pattern = np.empty((0, len(band_freqs) * band_freq_index_range))
    for spec in speaker_spec:
        pattern = np.array([])
        for band_spec, freq in zip(spec, band_freqs):
            band_index = int(freq // fft_freq_rate)
            pattern = np.append(
                pattern,
                np.abs(band_spec)[band_index : band_index + band_freq_index_range],
            )
        azimuth_pattern = np.vstack([azimuth_pattern, (pattern / np.max(pattern))])

    # スペクトル補間
    all_azimuth = np.arange(-40, 41)
    all_mic_angles = np.arange(0, 91)

    perfect_spec = np.empty((len(all_azimuth), 0))
    for i in range(len(azimuth_pattern[0, :])):
        akima = interpolate.Akima1DInterpolator(azimuth_points, azimuth_pattern[:, i])
        tmp = akima(all_azimuth).reshape(-1, 1)
        perfect_spec = np.hstack([perfect_spec, tmp])

    # アンプリチュード補間
    azimuth_akima = interpolate.Akima1DInterpolator(azimuth_points, speaker_ampli)
    azimuth_ampli = azimuth_akima(all_azimuth)
    mic_angle_akima = interpolate.Akima1DInterpolator(mic_angles, mic_ampli)
    mic_angle_ampli = mic_angle_akima(all_mic_angles)

    # アンプリチュード補正
    normalized_mic_ampli = mic_angle_ampli / mic_angle_ampli[0]
    mix_ampli = np.empty((0, len(normalized_mic_ampli)))
    for s_ampli in azimuth_ampli:
        mix = np.array([])
        for m_ampli in normalized_mic_ampli:
            x_ampli = s_ampli * m_ampli
            mix = np.append(mix, x_ampli)
        mix_ampli = np.vstack([mix_ampli, mix])

    return perfect_spec, mix_ampli

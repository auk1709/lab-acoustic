import numpy as np
import scipy.signal as sg
from .make_wave import chirp_exp
import matplotlib.pyplot as plt
import seaborn as sns
from .make_wave import reference_transmit_signal, reference_transmit_tukey


def extract_signal_start(
    res_signal: np.ndarray,
    first_freq=4000,
    last_freq=13000,
    interval_length: float = 0.100,
):
    """音声信号から1波形分の信号を抽出する最初のインデックスを求める
    マッチドフィルターによって1波形のみを抽出する

    Parameters
    ----------
    res_signal : NDArray
        受信信号
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval_length : float
        チャープのバンド間の間隔(s)

    Returns
    -------
    index_f : int
        1波形分の信号を抽出する最初のインデックス
    """

    sampling_rate = 48000  # マイクのサンプリングレート
    sample_frame_length = int(interval_length * 20 * sampling_rate)  # 信号約2個分のフレーム数
    chirp = reference_transmit_signal(
        first_freq=first_freq, last_freq=last_freq, interval_length=interval_length
    )  # 参照信号の生成
    corr = sg.correlate(res_signal[:sample_frame_length], chirp, mode="valid")  # 相互相関
    corr_lags = sg.correlation_lags(
        len(res_signal[:sample_frame_length]), len(chirp), mode="valid"
    )
    # 最大値のインデックス見つける
    index_f = corr_lags[np.abs(corr).argmax()]
    return index_f


def get_spectrum_amplitude(
    res_signal: np.ndarray,
    first_freq: int = 4000,
    last_freq: int = 13000,
    interval_length: float = 0.100,
    ampli_band="first",
    ret_spec="pattern",
    plot=False,
):
    """音声信号のスペクトル振幅を求める
    帯域ごとに区切られたチャープ信号が連続で送られてくる受信信号から、
    該当部分を切り出し送信されたチャープ信号のスペクトル振幅を求める

    Parameters
    ---------
    res_signal : NDArray
        受信信号
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval_length : float
        チャープのバンド間の間隔(s)
    ampli_band : string
        受信強度を出すときに相互相関をとる帯域, 'first' or 'all'
    ret_spec : string
        返り値のスペクトルの形式, 'pattern' or 'all'
    plot : bool
        プロットするかどうか

    Returns
    -------
    spec_ampli : NDArray
        スペクトル振幅
    max_corr : float
        相互相関の最大値, 参照信号の振幅となる
    """

    # 方位角-40~40°, 仰角0~50°, スペクトル6個×10発
    sampling_rate = 48000  # マイクのサンプリングレート
    signal_length = 0.003  # チャープ一発の信号長
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    sampling_buffer = 48  # データ切り出しの前後のゆとりN_c (1ms)
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    # マッチドフィルター
    sample_frame_length = int(interval_length * 20 * sampling_rate)  # 0.1秒分のサンプル数
    chirp = reference_transmit_signal(
        first_freq=first_freq, last_freq=last_freq, interval_length=interval_length
    )  # 参照信号の生成
    corr = sg.correlate(res_signal[:sample_frame_length], chirp, mode="valid")  # 相互相関
    if ampli_band == "all":
        max_corr = np.abs(corr).max()
    corr_lags = sg.correlation_lags(
        len(res_signal[:sample_frame_length]), len(chirp), mode="valid"
    )
    index_f = corr_lags[np.abs(corr).argmax()]  # 最大値のインデックス見つける

    # 検証用のプロット
    if plot:
        print(index_f)
        sns.set()
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes[0][0].plot(res_signal[:96000])
        axes[0][1].plot(chirp)
        axes[1][0].plot(corr_lags, corr)
        axes[1][1].plot(res_signal[index_f : index_f + 50000])
        plt.show()

    spec_ampli = np.array([])  # 各バンドから抽出したスペクトルパターンを格納する配列
    bands_spec = np.empty((0, len_chirp_sample))  # 各バンドのスペクトル振幅を格納する配列
    for i, band_freq in enumerate(band_freqs):
        start_i = index_f + (i * (len_chirp_sample + interval_sample_length))
        current_sample = res_signal[
            start_i - sampling_buffer : start_i + len_chirp_sample + sampling_buffer
        ]
        # チャープ信号の生成
        chirp = chirp_exp(
            band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi
        )
        # 相互相関
        corr = sg.correlate(current_sample, chirp, mode="valid")
        corr_lags = sg.correlation_lags(len(current_sample), len(chirp), mode="valid")
        # 最大値のインデックス見つける
        index = corr_lags[np.abs(corr).argmax()]
        # 最初のバンドの相互相関の最大値を参照信号の振幅とする
        if ampli_band == "first" and i == 0:
            max_corr = np.abs(corr).max()
        # 1つ分の波の抽出
        X_1sec = current_sample[index : index + len_chirp_sample]
        # 計測点のスペクトル算出
        spectrum = np.fft.fft(X_1sec)
        fft_freq = np.fft.fftfreq(len(X_1sec), 1 / sampling_rate)
        bands_spec = np.vstack([bands_spec, spectrum])
        ampli_spec = np.abs(spectrum)
        band_i = int(band_freq // fft_freq_rate)
        spec_ampli = np.append(
            spec_ampli, ampli_spec[band_i : band_i + band_freq_index_range]
        )
        # 検証用のプロット
        if plot:
            fig, axis = plt.subplots(3, 2, figsize=(16, 10))
            plt.subplots_adjust(hspace=0.5)
            axis[0][0].plot(current_sample)
            axis[0][1].plot(chirp)
            axis[1][0].plot(corr_lags, corr)
            axis[1][1].plot(X_1sec)
            axis[2][0].plot(fft_freq, ampli_spec)
            axis[2][1].plot(fft_freq, np.abs(np.fft.fft(chirp)))
            plt.show()
    if ret_spec == "all":
        return bands_spec, max_corr
    # 正規化
    nomalized_spec = spec_ampli / np.max(spec_ampli)
    return nomalized_spec, max_corr


def get_tukey_spectrum_amplitude(
    res_signal: np.ndarray,
    first_freq: int = 15000,
    last_freq: int = 24000,
    interval_length: float = 0.2,
    ampli_band="first",
    ret_spec="pattern",
    plot=False,
):
    """音声信号のスペクトル振幅を求める
    tukey窓をかけた帯域ごとに区切られたチャープ信号が連続で送られてくる受信信号から、
    該当部分を切り出し送信されたチャープ信号のスペクトル振幅を求める

    Parameters
    ---------
    res_signal : NDArray
        受信信号
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval_length : float
        チャープのバンド間の間隔(s)
    ampli_band : string
        受信強度を出すときに相互相関をとる帯域, 'first' or 'all'
    ret_spec : string
        返り値のスペクトルの形式, 'pattern' or 'all'
    plot : bool
        プロットするかどうか

    Returns
    -------
    spec_ampli : NDArray
        スペクトル振幅
    max_corr : float
        相互相関の最大値, 参照信号の振幅となる
    """

    # 方位角-40~40°, 仰角0~50°, スペクトル6個×10発
    sampling_rate = 48000  # マイクのサンプリングレート
    signal_length = 0.003  # チャープ一発の信号長
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    sampling_buffer = 48  # データ切り出しの前後のゆとりN_c (1ms)
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲
    tukey = sg.windows.tukey(int(sampling_rate * 0.003))
    # マッチドフィルター
    sample_frame_length = int(interval_length * 20 * sampling_rate)  # 0.1秒分のサンプル数
    chirp = reference_transmit_tukey(
        first_freq=first_freq, last_freq=last_freq, interval_length=interval_length
    )  # 参照信号の生成
    corr = sg.correlate(res_signal[:sample_frame_length], chirp, mode="valid")  # 相互相関
    if ampli_band == "all":
        max_corr = np.abs(corr).max()
    corr_lags = sg.correlation_lags(
        len(res_signal[:sample_frame_length]), len(chirp), mode="valid"
    )
    index_f = corr_lags[np.abs(corr).argmax()]  # 最大値のインデックス見つける

    spec_ampli = np.array([])  # 各バンドから抽出したスペクトルパターンを格納する配列
    bands_spec = np.empty((0, len_chirp_sample))  # 各バンドのスペクトル振幅を格納する配列
    for i, band_freq in enumerate(band_freqs):
        start_i = index_f + (i * (len_chirp_sample + interval_sample_length))
        current_sample = res_signal[
            start_i - sampling_buffer : start_i + len_chirp_sample + sampling_buffer
        ]
        # チャープ信号の生成
        chirp = (
            chirp_exp(band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi)
            * tukey
        )
        # 相互相関
        corr = sg.correlate(current_sample, chirp, mode="valid")
        corr_lags = sg.correlation_lags(len(current_sample), len(chirp), mode="valid")
        # 最大値のインデックス見つける
        index = corr_lags[np.abs(corr).argmax()]
        # 最初のバンドの相互相関の最大値を参照信号の振幅とする
        if ampli_band == "first" and i == 0:
            max_corr = np.abs(corr).max()
        # 1つ分の波の抽出
        X_1sec = current_sample[index : index + len_chirp_sample]
        # 計測点のスペクトル算出
        spectrum = np.fft.fft(X_1sec)
        fft_freq = np.fft.fftfreq(len(X_1sec), 1 / sampling_rate)
        bands_spec = np.vstack([bands_spec, spectrum])
        ampli_spec = np.abs(spectrum)
        band_i = int(band_freq // fft_freq_rate)
        spec_ampli = np.append(
            spec_ampli, ampli_spec[band_i : band_i + band_freq_index_range]
        )
    if ret_spec == "all":
        return bands_spec, max_corr
    # 正規化
    nomalized_spec = spec_ampli / np.max(spec_ampli)
    return nomalized_spec, max_corr


def get_sn_amplitude(
    res_signal: np.ndarray,
    first_freq: int = 15000,
    last_freq: int = 22000,
    interval_length: float = 0.2,
):
    """音声信号の信号とノイズの振幅を求める
    tukey窓をかけた帯域ごとに区切られたチャープ信号が連続で送られてくる受信信号から、
    各周波数帯ごとの信号とノイズの振幅を求める
    これを100回分やって後でSN比を求めるのに使う

    Parameters
    ---------
    res_signal : NDArray
        受信信号
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval_length : float
        チャープのバンド間の間隔(s)

    Returns
    -------
    NDArray
        各周波数帯ごとの信号とノイズの振幅
    """

    # 方位角-40~40°, 仰角0~50°, スペクトル6個×10発
    sampling_rate = 48000  # マイクのサンプリングレート
    signal_length = 0.003  # チャープ一発の信号長
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    sampling_buffer = 48  # データ切り出しの前後のゆとりN_c (1ms)
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲
    tukey = sg.windows.tukey(int(sampling_rate * 0.003))
    # マッチドフィルター
    sample_frame_length = int(interval_length * 20 * sampling_rate)  # 0.1秒分のサンプル数
    chirp = reference_transmit_tukey(
        first_freq=first_freq, last_freq=last_freq, interval_length=interval_length
    )  # 参照信号の生成
    corr = sg.correlate(res_signal[:sample_frame_length], chirp, mode="valid")  # 相互相関
    corr_lags = sg.correlation_lags(
        len(res_signal[:sample_frame_length]), len(chirp), mode="valid"
    )
    index_f = corr_lags[np.abs(corr).argmax()]  # 最大値のインデックス見つける

    signal_noise = []
    for i, band_freq in enumerate(band_freqs):
        start_i = index_f + (i * (len_chirp_sample + interval_sample_length))
        current_sample = res_signal[
            start_i - sampling_buffer : start_i + len_chirp_sample + sampling_buffer
        ]
        chirp = (
            chirp_exp(band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi)
            * tukey
        )  # 参照信号
        corr = sg.correlate(current_sample, chirp, mode="valid")  # 相互相関
        corr_lags = sg.correlation_lags(len(current_sample), len(chirp), mode="valid")
        ampli_signal = np.abs(corr).max()  # 相互相関の最大値を参照信号の振幅とする
        index = corr_lags[np.abs(corr).argmax()]  # 最大値のインデックス
        noise_i = start_i - sampling_buffer + index - (3 * len_chirp_sample)
        noise_sample = res_signal[noise_i : noise_i + 2 * len_chirp_sample]
        n_corr = sg.correlate(noise_sample, chirp, mode="valid")
        ampli_noise = np.abs(n_corr).max()
        signal_noise.append([ampli_signal, ampli_noise])
    return np.array(signal_noise)


def get_spec_ampli_noise(
    res_signal: np.ndarray, interval_length: float = 0.2, ret_spec="pattern"
):
    """スペクトル、振幅、無音時のノイズの平均を取得

    Parameters
    ----------
    res_signal : NDArray
        受信信号
    interval_length : float
        チャープのバンド間の間隔(s)
    ret_spec : string
        返り値のスペクトルの形式, 'pattern' or 'all'

    """

    # 方位角-40~40°, 仰角0~50°, スペクトル6個×10発
    sampling_rate = 48000  # マイクのサンプリングレート
    signal_length = 0.003  # チャープ一発の信号長
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    len_chirp_sample = 144  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    band_freqs = np.arange(4000, 13000, chirp_width)  # 送信する周波数のバンド
    sampling_buffer = 48  # データ切り出しの前後のゆとりN_c (1ms)
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    spec_ampli = np.array([])  # 各バンドから抽出したスペクトルパターンを格納する配列
    bands_spec = np.empty((0, len_chirp_sample))  # 各バンドのスペクトル振幅を格納する配列
    # マッチドフィルター
    index_f = extract_signal_start(res_signal, interval_length=interval_length)

    for i, band_freq in enumerate(band_freqs):
        start_i = index_f + (i * (len_chirp_sample + interval_sample_length))
        current_sample = res_signal[
            start_i - sampling_buffer : start_i + len_chirp_sample + sampling_buffer
        ]
        chirp = chirp_exp(
            band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi
        )  # チャープ信号の生成
        corr = sg.correlate(current_sample, chirp, mode="valid")  # 相互相関
        corr_lags = sg.correlation_lags(len(current_sample), len(chirp), mode="valid")
        index = corr_lags[np.abs(corr).argmax()]  # 最大値のインデックス見つける
        if i == 0:
            max_corr = np.abs(corr).max()  # 最初のバンドの相互相関の最大値を参照信号の振幅とする
        X_1sec = current_sample[index : index + len_chirp_sample]  # 1つ分の波の抽出
        spectrum = np.fft.fft(X_1sec)  # 計測点のスペクトル算出
        bands_spec = np.vstack([bands_spec, spectrum])
        ampli_spec = np.abs(spectrum)
        band_i = int(band_freq // fft_freq_rate)
        spec_ampli = np.append(
            spec_ampli, ampli_spec[band_i : band_i + band_freq_index_range]
        )

    noise_i = index_f + ((len_chirp_sample + interval_sample_length) * len(band_freqs))
    noise_sample = res_signal[
        noise_i
        - int(interval_sample_length / 2) : noise_i
        + int(interval_sample_length / 2)
    ]
    noise_avg = np.mean(noise_sample**2)

    if ret_spec == "all":
        return bands_spec, max_corr, noise_avg
    nomalized_spec = spec_ampli / np.max(spec_ampli)  # スペクトルパターンの正規化
    return nomalized_spec, max_corr, noise_avg

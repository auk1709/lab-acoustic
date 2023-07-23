import numpy as np
import scipy.signal as sg
from .chirp_exp import chirp_exp
import matplotlib.pyplot as plt
import seaborn as sns


def get_spectrum_amplitude(
    res_signal: np.ndarray, interval_length: float = 0.100, plot=False
):
    """音声信号のスペクトル振幅を求める
    帯域ごとに区切られたチャープ信号が連続で送られてくる受信信号から、
    該当部分を切り出し送信されたチャープ信号のスペクトル振幅を求める

    Parameters
    ---------
    res_signal : NDArray
        受信信号
    interval_length : float
        チャープのバンド間の間隔(s)
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
    band_freqs = np.arange(4000, 13000, chirp_width)  # 送信する周波数のバンド
    sampling_buffer = 48  # データ切り出しの前後のゆとりN_c (1ms)
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲

    spec_ampli = np.full((band_freq_index_range * len(band_freqs)), np.nan)
    # チャープ信号の生成
    chirp = chirp_exp(
        band_freqs[0], band_freqs[0] + chirp_width, signal_length, 0.5 * np.pi
    )
    for band_freq in band_freqs[1:]:
        interval = np.zeros(interval_sample_length)
        chirp = np.concatenate([chirp, interval])
        tmp_chirp = chirp_exp(
            band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi
        )
        chirp = np.concatenate([chirp, tmp_chirp])

    # 相互相関
    corr = sg.correlate(res_signal[:96000], chirp, mode="valid")
    corr_lags = sg.correlation_lags(len(res_signal[:96000]), len(chirp), mode="valid")
    # 最大値のインデックス見つける
    index_f = corr_lags[np.abs(corr).argmax()]

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
        if i == 0:
            max_corr = np.abs(corr).max()
        # 1つ分の波の抽出
        X_1sec = current_sample[index : index + len_chirp_sample]
        # データベース構築, 計測点のスペクトル算出
        spectrum = np.fft.fft(X_1sec)
        ampli_spec = np.abs(spectrum)
        fft_freq = np.fft.fftfreq(len(X_1sec), 1 / sampling_rate)
        spec_ampli[
            i * band_freq_index_range : (i + 1) * band_freq_index_range,
        ] = ampli_spec[
            int(band_freq // fft_freq_rate) : int(band_freq // fft_freq_rate)
            + band_freq_index_range
        ]
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
    # 正規化
    nomalized_spec = spec_ampli / np.max(spec_ampli)
    return nomalized_spec, max_corr

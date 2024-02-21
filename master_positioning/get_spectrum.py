import numpy as np
from scipy import signal
from .make_wave import chirp_exp, reference_transmit_tukey


def get_tukey_spectrum(
    res_signal: np.ndarray,
    first_freq: int = 1000,
    last_freq: int = 24000,
    interval_length: float = 0.1,
    signal_length=0.001,
):
    """音声信号のスペクトル振幅を求める
    tukey窓をかけた帯域ごとに区切られたチャープ信号が連続で送られてくる受信信号から、
    該当部分を切り出し送信されたチャープ信号のスペクトル振幅を求める
    スペクトルのパターンのみを返す

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
    signal_length : float
        チャープの信号長(s)

    Returns
    -------
    spec_ampli : NDArray
        スペクトル振幅
    """

    sampling_rate = 48000  # マイクのサンプリングレート
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    len_chirp_sample = int(sampling_rate * signal_length)  # 受信したチャープのサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅(Hz)
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    sampling_buffer = 48  # データ切り出しの前後のゆとりN_c (1ms)
    fft_freq_rate = sampling_rate / len_chirp_sample  # FFTの周波数分解能
    band_freq_index_range = int(chirp_width / fft_freq_rate + 1)  # 1つの帯域の周波数インデックスの範囲
    tukey = signal.windows.tukey(len_chirp_sample)  # tukey窓

    # マッチドフィルター
    chirp = reference_transmit_tukey(
        first_freq=first_freq,
        last_freq=last_freq,
        interval_length=interval_length,
        signal_length=signal_length,
    )  # 参照信号の生成
    corr = signal.correlate(res_signal, chirp, mode="valid")  # 相互相関
    corr_lags = signal.correlation_lags(len(res_signal), len(chirp), mode="valid")
    signal_start_i = corr_lags[np.abs(corr).argmax()]  # 信号の開始位置検出

    spectrum = np.array([])  # 各バンドから抽出したスペクトルパターンを格納する配列
    for i, band_freq in enumerate(band_freqs):
        cur_start_i = signal_start_i + (i * (len_chirp_sample + interval_sample_length))
        current_sample = res_signal[
            cur_start_i
            - sampling_buffer : cur_start_i
            + len_chirp_sample
            + sampling_buffer
        ]  # 前後ゆとりを持たせたデータを切り出す
        # マッチドフィルター
        chirp = (
            chirp_exp(band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi)
            * tukey
        )  # チャープ信号の生成
        corr = signal.correlate(current_sample, chirp, mode="valid")  # 相互相関
        corr_lags = signal.correlation_lags(
            len(current_sample), len(chirp), mode="valid"
        )
        matched_i = corr_lags[np.abs(corr).argmax()]  # 切り出したデータから正確な位置を見つける
        spec = np.abs(
            np.fft.fft(current_sample[matched_i : matched_i + len_chirp_sample])
        )  # 計測点のスペクトル算出
        band_i = int(band_freq // fft_freq_rate)
        spectrum = np.append(
            spectrum, spec[band_i : band_i + band_freq_index_range]
        )  # 必要な帯域のスペクトルを抽出
    nomalized_spec = spectrum / np.max(spectrum)  # 正規化
    return nomalized_spec

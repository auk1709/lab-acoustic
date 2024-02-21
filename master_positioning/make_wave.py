import numpy as np
from scipy.signal import chirp, windows
from scipy.io.wavfile import write
import soundfile as sf


def make_transmit_tukey(
    first_freq: int = 1000,
    last_freq: int = 24000,
    interval_time: float = 0.1,
    signal_length: float = 0.001,
):
    """tukey窓をかけた送信信号となる音声ファイルを生成する
    1kHzごとの周波数帯を連続的に送信するチャープ信号のwaveファイルを生成する

    Parameters
    ----------
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval_time : float
        送信する周波数帯の間隔(秒)
    """

    sampling_rate = 48000  # サンプリング周波数
    transmit_bands = np.arange(first_freq, last_freq, 1000)  # 送信する周波数のバンド
    signal = np.array([])
    tukey = windows.tukey(int(sampling_rate * signal_length))
    for band in transmit_bands:
        tmp_signal = (
            chirp(
                t=np.arange(0, signal_length, 1 / sampling_rate),
                f0=band,
                f1=band + 1000,
                t1=signal_length,
            )
            * tukey
        )
        signal = np.concatenate([signal, tmp_signal])
        interval = np.zeros(int(interval_time * sampling_rate))
        signal = np.concatenate([signal, interval])
    ret_time = ((len(signal) // (sampling_rate / 2)) + 1) / 2  # 送信信号の長さ,切りの良い時間にする(秒)
    pad_len = int(ret_time * sampling_rate - len(signal))
    pad = np.zeros(pad_len)  # 全体が0.5秒単位になるように0で埋める
    signal = np.concatenate([signal, pad])
    f_name = f"transmit_tukey_{int(first_freq // 1000)}k-{int(last_freq // 1000)}k_i{int(interval_time * 1000)}.wav"
    write(f_name, sampling_rate, signal)


def chirp_exp(
    start: int,
    stop: int,
    signal_length: float,
    phase: float = 0,
    sampling_rate: int = 48000,
):
    """複素のチャープの生成関数
    チャープ信号の正の周波数成分のみを返す

    Parameters
    ----------
    start : int
        開始周波数[Hz]
    stop : int
        終了周波数[Hz]
    signal_length : float
        信号長[秒]
    phase : float
        初期位相[rad]
    samplingRate : int
        サンプリングレート

    Returns
    -------
    NDArray[float]
        チャープ信号の正の周波数成分の配列
    """

    num_samples = int(signal_length * sampling_rate)  # サンプリング数
    time = np.linspace(0, signal_length, num_samples)  # 時間
    k = (stop - start) / signal_length  # 周波数の変化
    sweep_freqs = (start + k / 2.0 * time) * time  # チャープ信号の周波数
    chirp = np.array(np.exp(phase * 1j - 2 * np.pi * 1j * sweep_freqs))
    return chirp


def reference_transmit_tukey(
    first_freq: int = 1000,
    last_freq: int = 24000,
    interval_length: float = 0.1,
    signal_length: float = 0.001,
):
    """受信信号抽出のための参照信号を生成する
    マッチドフィルターによって1波形のみを抽出するための参照信号を生成する

    Parameters
    ----------
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval_length : float
        送信する周波数帯の間隔(秒)
    signal_length : float
        チャープ一発の信号長

    Returns
    -------
    NDArray
        参照信号
    """

    sampling_rate = 48000  # マイクのサンプリングレート
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    tukey = windows.tukey(int(sampling_rate * signal_length))  # tukey窓
    chirp = (
        chirp_exp(
            band_freqs[0], band_freqs[0] + chirp_width, signal_length, 0.5 * np.pi
        )
        * tukey
    )  # 最初の周波数帯のチャープ
    for band_freq in band_freqs[1:]:  # 残りの周波数帯のチャープ,間隔を挟む
        interval = np.zeros(interval_sample_length)
        chirp = np.concatenate([chirp, interval])
        tmp_chirp = (
            chirp_exp(band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi)
            * tukey
        )
        chirp = np.concatenate([chirp, tmp_chirp])

    return chirp

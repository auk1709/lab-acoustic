import numpy as np
from scipy.signal import chirp, windows
from scipy.io.wavfile import write
import soundfile as sf


def make_transmit_signal(
    first_freq: int = 4000, last_freq: int = 13000, interval_time: float = 0.1
):
    """送信信号となる音声ファイルを生成する
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
    for band in transmit_bands:
        tmp_signal = chirp(
            t=np.arange(0, 0.003, 1 / sampling_rate), f0=band, f1=band + 1000, t1=0.003
        )
        signal = np.concatenate([signal, tmp_signal])
        interval = np.zeros(int(interval_time * sampling_rate))
        signal = np.concatenate([signal, interval])
    ret_time = (len(signal) // sampling_rate) + 1  # 送信信号の長さ,切りの良い時間にする(秒)
    pad_len = ret_time * sampling_rate - len(signal)
    pad = np.zeros(pad_len)  # 全体が1秒になるように0で埋める
    signal = np.concatenate([signal, pad])
    f_name = f"transmit_{int(first_freq // 1000)}k-{int(last_freq // 1000)}k_i{int(interval_time * 1000)}.wav"
    write(f_name, sampling_rate, signal)


def make_transmit_tukey(
    first_freq: int = 15000, last_freq: int = 24000, interval_time: float = 0.2
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
    tukey = windows.tukey(int(sampling_rate * 0.003))
    for band in transmit_bands:
        tmp_signal = (
            chirp(
                t=np.arange(0, 0.003, 1 / sampling_rate),
                f0=band,
                f1=band + 1000,
                t1=0.003,
            )
            * tukey
        )
        signal = np.concatenate([signal, tmp_signal])
        interval = np.zeros(int(interval_time * sampling_rate))
        signal = np.concatenate([signal, interval])
    ret_time = (len(signal) // sampling_rate) + 1  # 送信信号の長さ,切りの良い時間にする(秒)
    pad_len = ret_time * sampling_rate - len(signal)
    pad = np.zeros(pad_len)  # 全体が1秒になるように0で埋める
    signal = np.concatenate([signal, pad])
    f_name = f"transmit_tukey_{int(first_freq // 1000)}k-{int(last_freq // 1000)}k_i{int(interval_time * 1000)}.wav"
    write(f_name, sampling_rate, signal)


def chirp_exp(
    start_Hz: int,
    stop_Hz: int,
    chirpLen_s: float,
    phase_rad: float = 0,
    samplingRate: int = 48000,
):
    """チャープの生成関数,chirpaは生成したチャープ信号の配列を返す
    チャープ信号の正の周波数成分のみを返す
    Parameters
    ----------
    start_Hz : int
        開始周波数
    stop_Hz : int
        終了周波数
    chirpLen_s : float
        チャープ信号の時間
    phase_rad : float
        チャープ信号の位相
    samplingRate : int
        サンプリングレート

    Returns
    -------
    NDArray[float]
        チャープ信号の正の周波数成分の配列
    """
    numSamples = int(chirpLen_s * samplingRate)  # Number of samples.
    times_s = np.linspace(0, chirpLen_s, numSamples)  # Chirp times.
    k = (stop_Hz - start_Hz) / chirpLen_s  # Chirp rate.
    sweepFreqs_Hz = (start_Hz + k / 2.0 * times_s) * times_s  # 時間に対する周波数の変化
    chirpa = np.array(np.exp(phase_rad * 1j - 2 * np.pi * 1j * sweepFreqs_Hz))
    return chirpa


def reference_transmit_signal(
    first_freq: int = 4000, last_freq: int = 13000, interval_length: float = 0.1
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

    Returns
    -------
    NDArray
        参照信号
    """

    sampling_rate = 48000  # マイクのサンプリングレート
    signal_length = 0.003  # チャープ一発の信号長
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド

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

    return chirp


def reference_transmit_tukey(
    first_freq: int = 15000, last_freq: int = 24000, interval_length: float = 0.2
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

    Returns
    -------
    NDArray
        参照信号
    """

    sampling_rate = 48000  # マイクのサンプリングレート
    signal_length = 0.003  # チャープ一発の信号長
    interval_sample_length = int(interval_length * sampling_rate)  # チャープのバンド間の間隔のサンプル数
    chirp_width = 1000  # チャープ一発の周波数帯域の幅
    band_freqs = np.arange(first_freq, last_freq, chirp_width)  # 送信する周波数のバンド
    tukey = windows.tukey(int(sampling_rate * 0.003))
    chirp = (
        chirp_exp(
            band_freqs[0], band_freqs[0] + chirp_width, signal_length, 0.5 * np.pi
        )
        * tukey
    )
    for band_freq in band_freqs[1:]:
        interval = np.zeros(interval_sample_length)
        chirp = np.concatenate([chirp, interval])
        tmp_chirp = (
            chirp_exp(band_freq, band_freq + chirp_width, signal_length, 0.5 * np.pi)
            * tukey
        )
        chirp = np.concatenate([chirp, tmp_chirp])

    return chirp


def make_floor_reflect_wav():
    """床面反射で測距するためのwavファイルを生成する
    スマホが出す, とりあえずただのチャープ信号
    一応tukey窓はかけておく
    100回+α回繰り返す
    """

    sampling_rate = 48000  # サンプリング周波数
    first_freq = 15000
    last_freq = 22000
    signal_length = 0.05
    interval_length = 0.1
    tukey = windows.tukey(int(sampling_rate * signal_length))
    signal = (
        chirp(
            t=np.arange(0, signal_length, 1 / sampling_rate),
            f0=first_freq,
            f1=last_freq,
            t1=signal_length,
        )
        * tukey
    )
    interval = np.zeros(int(interval_length * sampling_rate))
    set_signal = np.concatenate([signal, interval])
    repeat_signal = np.tile(set_signal, 535)
    f_name = f"reflect_floor.wav"
    sf.write(f_name, repeat_signal, sampling_rate)
    # write(f_name, sampling_rate, repeat_signal)


if __name__ == "__main__":
    make_floor_reflect_wav()

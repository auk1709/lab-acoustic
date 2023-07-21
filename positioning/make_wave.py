import numpy as np
from scipy.signal import chirp
from scipy.io.wavfile import write


def make_transmit_signal(
    first_freq: int = 4000, last_freq: int = 13000, interval_time: float = 0.1
):
    """送信信号となる音声ファイルを生成する
    1kHzごとの周波数帯を連続的に送信するチャープ信号を生成する

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
    pad_len = 48000 - len(signal)
    pad = np.zeros(pad_len)  # 全体が1秒になるように0で埋める
    signal = np.concatenate([signal, pad])
    write("transmit_4k-13k.wav", sampling_rate, signal)


if __name__ == "__main__":
    make_transmit_signal()

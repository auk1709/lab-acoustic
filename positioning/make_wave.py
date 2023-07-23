import numpy as np
from scipy.signal import chirp
from scipy.io.wavfile import write


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


if __name__ == "__main__":
    make_transmit_signal()

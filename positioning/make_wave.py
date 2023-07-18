import numpy as np
from scipy.signal import chirp
from scipy.io.wavfile import write


def make_transmit_signal():
    """送信信号となる音声ファイルを生成する
    1kHzごとの周波数帯を連続的に送信するチャープ信号を生成する
    """

    sampling_rate = 48000  # サンプリング周波数
    transmit_bands = np.arange(4000, 13000, 1000)  # 送信する周波数のバンド
    signal = np.array([])
    for band in transmit_bands:
        tmp_signal = chirp(
            t=np.arange(0, 0.003, 1 / sampling_rate), f0=band, f1=band + 1000, t1=0.003
        )
        signal = np.concatenate([signal, tmp_signal])
        interval = np.zeros(int(0.100 * sampling_rate))
        signal = np.concatenate([signal, interval])
    pad_len = 48000 - len(signal)
    pad = np.zeros(pad_len)
    signal = np.concatenate([signal, pad])
    write("transmit_4k-13k.wav", sampling_rate, signal)


if __name__ == "__main__":
    make_transmit_signal()

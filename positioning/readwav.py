import wave
import numpy as np


def readwav(file1):
    """waveファイルを読み込む

    Parameters
    ----------
    file1: _File
        対象のwaveファイル

    Returns
    -------
    NDArray[int16]
        信号データの配列
    """

    wr = wave.open(file1, "r")
    data = wr.readframes(wr.getnframes())
    wr.close()
    X = np.frombuffer(data, dtype=np.int16)
    return X

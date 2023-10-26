import soundfile as sf


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

    data, rate = sf.read(file1)
    return data

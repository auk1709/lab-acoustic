import numpy as np


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

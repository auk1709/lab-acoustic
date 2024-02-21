import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar
from .make_wave import reference_transmit_tukey
from .get_spectrum import get_tukey_spectrum


def estimate_direction_3d(
    reference_spec,
    recieved_signal,
    first_freq: int = 1000,
    last_freq: int = 24000,
    interval=0.1,
    signal_length: float = 0.001,
):
    """3次元の方位推定を行う

    Parameters
    ----------
    reference_spec : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース
    recieved_signal : NDArray
        読み込んだ検証用の音響信号データ
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval : float
        送信する周波数帯の間隔(秒)

    Returns
    -------
    NDArray
        推定した方位角、仰角
    """

    test_spec = get_tukey_spectrum(
        recieved_signal,
        first_freq=first_freq,
        last_freq=last_freq,
        interval_length=interval,
        signal_length=signal_length,
    )  # テストデータのスペクトルと振幅を取得

    # 全角度のスペクトルとの誤差の総和を記録
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=2)
    # 記録した誤差が最小となるインデックスを取得（角度決定）
    est_direction = np.unravel_index(np.argmin(rss_db), rss_db.shape)
    est_azimuth = est_direction[0] - 40
    est_elevation = est_direction[1] + 20

    return np.array([est_azimuth, est_elevation])


def get_reflect_ceiling_tdoa(
    res_signal,
    first_freq=1000,
    last_freq=24000,
    interval_length=0.1,
    signal_length=0.001,
):
    """天井反射音の到来時間差を求める関数

    Parameters
    ----------
    res_signal : np.ndarray
        受信信号
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    interval_length : float
        チャープのバンド間の間隔
    signal_length : float
        チャープ一発の信号長

    Returns
    -------
    int
        天井反射音の到来時間差のサンプル数
    """

    ref_transmit = reference_transmit_tukey(
        first_freq=first_freq,
        last_freq=last_freq,
        interval_length=interval_length,
        signal_length=signal_length,
    )  # 参照信号の生成
    corr = np.abs(signal.correlate(res_signal, ref_transmit, mode="valid"))  # 相互相関
    corr_lags = signal.correlation_lags(
        len(res_signal), len(ref_transmit), mode="valid"
    )
    index_f = corr_lags[corr.argmax()]  # 最大値のインデックス見つける
    next_peak = np.argmax(corr[index_f + 20 : index_f + 100]) + index_f + 20
    diff = next_peak - index_f
    return diff


def positioning_direction_reflect_tdoa(azimuth, elevation, tdoa_sample):
    """方位角、仰角、天井反射の到来時間差から測位する関数
    スピーカーが天井から30cmの位置にある設定

    Parameters
    ----------
    azimuth : float
        方位角 [deg]
    elevation : float
        仰角 [deg]
    tdoa_sample : int
        天井反射の到来時間差 [sample]

    Returns
    -------
    NDArray
        推定位置(x, y, z)[m]
    """

    sampling_rate = 48000  # マイクのサンプリングレート
    speaker_height = 2.2  # スピーカーの高さ
    speaker_ceiling_distance = 0.3  # スピーカーと天井の距離
    diff_distance = tdoa_sample / sampling_rate * 340  # 天井反射の到来時間差から距離差を計算

    def func(r):
        return (
            (
                (r * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))) ** 2
                + (r * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))) ** 2
                + (
                    r * np.sin(np.radians(elevation))
                    - (speaker_height + (speaker_ceiling_distance * 2))
                )
                ** 2
            )
            ** 0.5
            - (
                (r * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))) ** 2
                + (r * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))) ** 2
                + ((r * np.sin(np.radians(elevation))) - speaker_height) ** 2
            )
            ** 0.5
            - diff_distance
        ) ** 2

    res = minimize_scalar(func, method="bounded", bounds=(0, 5))
    x = res.x * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
    y = res.x * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
    z = -res.x * np.sin(np.radians(elevation)) + speaker_height

    return np.array([x, y, z])

import numpy as np
from scipy.signal import chirp, windows, correlate, correlation_lags
from scipy.optimize import minimize_scalar
import sympy
from .get_spectrum_amplitude import (
    get_spectrum_amplitude,
    get_tukey_spectrum_amplitude,
    get_spec_ampli_noise,
    get_tukey_spectrum,
)
from .make_wave import chirp_exp


def estimate(reference_spec, reference_ampli, file, interval=0.1, output="rect"):
    """測位を行う

    Parameters
    ----------
    reference_spec : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース
    reference_ampli : NDArray
        作成した方位角、仰角ごとの振幅の参照データベース
    file : NDArray
        読み込んだ検証用の音響信号データ
    output : string
        出力形式, 'rect' or 'polar'
    """

    test_spec, test_ampli = get_spectrum_amplitude(
        file, interval_length=interval
    )  # テストデータのスペクトルと振幅を取得

    # 角度推定
    # 全角度のスペクトルとの誤差の総和を記録
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=2)
    # 記録した残差平方和が最小となるインデックスを取得（角度決定）
    est_direction = np.unravel_index(np.argmin(rss_db), rss_db.shape)
    est_deg = np.array([est_direction[0] - 40, est_direction[1]])
    # 方位をradで, est_direction[0]-40
    est_azimuth = np.radians(est_direction[0] - 40)
    est_elevation = np.radians(est_direction[1])

    # 距離推定
    est_distance = reference_ampli[est_direction[0], est_direction[1]] / test_ampli
    if output == "polar":
        return np.append(est_deg, est_distance)

    # 測位点の座標を計算
    x_ans = est_distance * np.cos(est_elevation) * np.sin(est_azimuth)
    y_ans = est_distance * np.cos(est_elevation) * np.cos(est_azimuth)
    z_ans = est_distance * np.sin(est_elevation) + 1
    return np.array([x_ans, y_ans, z_ans])


def estimate_distance(reference_ampli, target_ampli, direction):
    """距離推定を行う

    Parameters
    ----------
    reference_ampli : NDArray
        作成した方位角、仰角ごとの振幅の参照データベース
    target_ampli : float
        推定対象の点の振幅
    direction : int
        推定対象の方向

    Returns
    -------
    float
        推定した距離
    """

    est_distance = reference_ampli[direction] / target_ampli
    return est_distance


def positioning_2d(
    reference_spec,
    reference_ampli,
    recieved_signal,
    first_freq: int = 4000,
    last_freq: int = 13000,
    interval=0.1,
    output="rect",
):
    """測位を行う

    Parameters
    ----------
    reference_spec : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース
    reference_ampli : NDArray
        作成した方位角、仰角ごとの振幅の参照データベース
    recieved_signal : NDArray
        読み込んだ検証用の音響信号データ
    output : string
        出力形式, 'rect' or 'polar', 直交座標系か極座標系か
    """

    test_spec, test_ampli = get_spectrum_amplitude(
        recieved_signal,
        first_freq=first_freq,
        last_freq=last_freq,
        interval_length=interval,
    )  # テストデータのスペクトルと振幅を取得

    # 角度推定
    # 全角度のスペクトルとの誤差の総和を記録
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=1)
    # 記録した残差平方和が最小となるインデックスを取得（角度決定）
    est_direction = np.argmin(rss_db)
    est_deg = est_direction - 40
    # 方位をradで, est_direction[0]-40
    est_azimuth = np.radians(est_direction - 40)

    # 距離推定
    est_distance = reference_ampli[est_direction] / test_ampli
    if output == "polar":
        return np.append(est_deg, est_distance)

    # 測位点の座標を計算
    x_ans = est_distance * np.sin(est_azimuth)
    y_ans = est_distance * np.cos(est_azimuth)
    return np.array([x_ans, y_ans])


def positioning_tukey(
    reference_spec,
    reference_ampli,
    recieved_signal,
    first_freq: int = 15000,
    last_freq: int = 22000,
    interval=0.2,
    output="rect",
):
    """測位を行う

    Parameters
    ----------
    reference_spec : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース
    reference_ampli : NDArray
        作成した方位角、仰角ごとの振幅の参照データベース
    recieved_signal : NDArray
        読み込んだ検証用の音響信号データ
    output : string
        出力形式, 'rect' or 'polar', 直交座標系か極座標系か
    """

    test_spec, test_ampli = get_tukey_spectrum_amplitude(
        recieved_signal,
        first_freq=first_freq,
        last_freq=last_freq,
        interval_length=interval,
        ampli_band="all",
    )  # テストデータのスペクトルと振幅を取得

    # 角度推定
    # 全角度のスペクトルとの誤差の総和を記録
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=1)
    # 記録した残差平方和が最小となるインデックスを取得（角度決定）
    est_direction = np.argmin(rss_db)
    est_deg = est_direction - 40
    # 方位をradで, est_direction[0]-40
    est_azimuth = np.radians(est_direction - 40)

    # 距離推定
    est_distance = reference_ampli[est_direction] / test_ampli
    if output == "polar":
        return np.append(est_deg, est_distance)

    # 測位点の座標を計算
    x_ans = est_distance * np.sin(est_azimuth)
    y_ans = est_distance * np.cos(est_azimuth)
    return np.array([x_ans, y_ans])


def positioning_mic_revision(
    reference_spec,
    reference_ampli,
    recieved_signal,
    first_freq: int = 4000,
    last_freq: int = 13000,
    interval=0.2,
    output="rect",
):
    """測位を行う

    Parameters
    ----------
    reference_spec : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース
    reference_ampli : NDArray
        作成した方位角、仰角ごとの振幅の参照データベース
    recieved_signal : NDArray
        読み込んだ検証用の音響信号データ
    first_freq : int
        送信する最初の周波数
    last_freq : int
        送信する最後の周波数
    output : string
        出力形式, 'rect' or 'polar', 直交座標系か極座標系か
    """

    test_spec, test_ampli = get_spectrum_amplitude(
        recieved_signal,
        first_freq=first_freq,
        last_freq=last_freq,
        interval_length=interval,
    )  # テストデータのスペクトルと振幅を取得

    # 角度推定
    # 全角度のスペクトルとの誤差の総和を記録
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=2)
    # 記録した残差平方和が最小となるインデックスを取得（角度決定）
    est_direction = np.unravel_index(np.argmin(rss_db), rss_db.shape)
    est_azimuth_deg = est_direction[0] - 40
    # 方位をradで
    est_azimuth = np.radians(est_azimuth_deg)
    est_mic_angle = np.radians(est_direction[1])

    # 距離推定
    est_distance = reference_ampli[est_direction[0], est_direction[1]] / test_ampli

    if output == "polar":
        return np.array([est_azimuth_deg, est_distance, est_direction[1]])

    # 測位点の座標を計算
    x_ans = est_distance * np.sin(est_azimuth)
    y_ans = est_distance * np.cos(est_azimuth)
    return np.array([x_ans, y_ans])

    # 検証用、スピーカーのみ
    # rss_db = np.sum(np.abs(reference_spec - test_spec), axis=1)
    # est_direction = np.argmin(rss_db)
    # est_azimuth_deg = est_direction - 40
    # est_azimuth = np.radians(est_azimuth_deg)
    # est_distance = reference_ampli[est_direction] / test_ampli
    # if output == "polar":
    #     return np.array([est_azimuth_deg, est_distance])
    # x_ans = est_distance * np.sin(est_azimuth)
    # y_ans = est_distance * np.cos(est_azimuth)
    # return np.array([x_ans, y_ans])


def positioning_ampli_revision(
    reference_spec, reference_ampli, recieved_signal, interval=0.2, output="rect"
):
    """測位を行う

    Parameters
    ----------
    reference_spec : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース
    reference_ampli : NDArray
        作成した方位角、仰角ごとの振幅の参照データベース
    recieved_signal : NDArray
        読み込んだ検証用の音響信号データ
    output : string
        出力形式, 'rect' or 'polar', 直交座標系か極座標系か
    """

    test_spec, test_ampli = get_spectrum_amplitude(
        recieved_signal, interval_length=interval
    )  # テストデータのスペクトルと振幅を取得

    # 角度推定
    # 全角度のスペクトルとの誤差の総和を記録
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=1)
    # 記録した残差平方和が最小となるインデックスを取得（角度決定）
    est_direction = np.argmin(rss_db)
    est_azimuth_deg = est_direction - 40
    # 方位をradで
    est_azimuth = np.radians(est_azimuth_deg)
    est_mic_angle_deg = np.abs(est_azimuth_deg)
    est_mic_angle = np.radians(est_mic_angle_deg)

    # 距離推定
    est_distance = reference_ampli[est_direction, est_mic_angle_deg] / test_ampli

    if output == "polar":
        return np.array([est_azimuth_deg, est_distance, est_mic_angle_deg])

    # 測位点の座標を計算
    x_ans = est_distance * np.sin(est_azimuth)
    y_ans = est_distance * np.cos(est_azimuth)
    return np.array([x_ans, y_ans])


def positioning_phone(
    reference_spec,
    reference_ampli,
    reference_noise,
    recieved_signal,
    interval=0.2,
    output="rect",
):
    """測位を行う

    Parameters
    ----------
    reference_spec : NDArray
        作成した方位角、仰角ごとのスペクトルの参照データベース
    reference_ampli : NDArray
        作成した方位角、仰角ごとの振幅の参照データベース
    reference_noise : NDArray
        作成したマイクの角度ごとのノイズの参照データベース
    recieved_signal : NDArray
        読み込んだ検証用の音響信号データ
    output : string
        出力形式, 'rect' or 'polar', 直交座標系か極座標系か
    """

    test_spec, test_ampli, test_noise = get_spec_ampli_noise(
        recieved_signal, interval_length=interval
    )  # テストデータのスペクトルと振幅を取得

    # 角度推定
    # 全角度のスペクトルとの誤差の総和を記録
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=2)
    # 記録した残差平方和が最小となるインデックスを取得（角度決定）
    est_direction = np.unravel_index(np.argmin(rss_db), rss_db.shape)
    est_azimuth_deg = est_direction[0] - 40
    # 方位をradで
    est_azimuth = np.radians(est_azimuth_deg)
    est_mic_angle = np.radians(est_direction[1])

    # 距離推定
    est_distance = (
        (reference_noise[np.abs(est_azimuth_deg)] / test_noise)
        * reference_ampli[est_direction[0], np.abs(est_azimuth_deg)]
        / test_ampli
    )

    if output == "polar":
        return np.array([est_azimuth_deg, est_distance, np.abs(est_azimuth_deg)])

    # 測位点の座標を計算
    x_ans = est_distance * np.sin(est_azimuth)
    y_ans = est_distance * np.cos(est_azimuth)
    return np.array([x_ans, y_ans])


def estimate_direction_3d(
    reference_spec,
    recieved_signal,
    first_freq: int = 15000,
    last_freq: int = 22000,
    interval=0.2,
    signal_length: float = 0.003,
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


def estimate_height(signal):
    """床面反射からの到達時間を用いて高さを推定する
    5発の平均をとっている
    現状、実際には天井からの距離になっている

    Parameters
    ----------
    signal : NDArray
        読み込んだ音響信号データ, 1s分

    Returns
    -------
    float
        推定した高さ
    """
    ref_chirp = chirp_exp(15000, 22000, 0.05, 0.5, 48000) * windows.tukey(
        int(48000 * 0.05)
    )
    corr = correlate(signal, ref_chirp)
    corr = corr / np.max(corr)
    first_max_i = np.argmax(corr[:10000])
    corr_oneset = np.array([corr[first_max_i - 100 : first_max_i + 3500]])
    for i in range(1, 5):
        current_i = first_max_i + (7200 * i)
        max_i_k = (
            np.argmax(corr[current_i - 3600 : current_i + 3600]) + current_i - 3600
        )
        corr_oneset = np.concatenate(
            [corr_oneset, [corr[max_i_k - 100 : max_i_k + 3500]]]
        )
    corr_avg = np.mean(corr_oneset, axis=0)
    corr_avg_max_i = np.argmax(corr_avg)
    corr_avg_second_i = (
        np.argmax(corr_avg[corr_avg_max_i + 100 :]) + corr_avg_max_i + 100
    )
    estimated_height = (340 * (corr_avg_second_i - corr_avg_max_i)) / (48000 * 2)
    return estimated_height


def calc_position(azimuth, elevation, height):
    """方位角、仰角、高さから座標を計算する"""
    x = (1.5 - height) * np.sin(np.radians(azimuth)) / np.sin(np.radians(elevation))
    y = (1.5 - height) * np.cos(np.radians(azimuth)) / np.sin(np.radians(elevation))
    z = height
    return x, y, z


def positioning_reflect_ceiling(
    reference_spec,
    recieved_signal,
    first_freq: int = 15000,
    last_freq: int = 22000,
    interval=0.2,
    signal_length: float = 0.003,
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
    rss_db = np.sum(np.abs(reference_spec - test_spec), axis=3)
    # 記録した誤差が最小となるインデックスを取得（角度決定）
    est_index = np.unravel_index(np.argmin(rss_db), rss_db.shape)

    return np.array(
        [
            est_index[0] / 100 - 0.25,
            est_index[1] / 100 + 1.25,
            est_index[2] / 100 + 0.75,
        ]
    )


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

    # r = sympy.Symbol("r")
    # x = sympy.Symbol("x")
    # y = sympy.Symbol("y")
    # z = sympy.Symbol("z")
    # expr1 = r * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)) - x
    # expr2 = r * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)) - y
    # expr3 = r * np.sin(np.radians(elevation)) - z
    # expr4 = (
    #     (x**2 + y**2 + (z - (speaker_height + (speaker_ceiling_distance * 2))) ** 2)
    #     ** 0.5
    #     - (x**2 + y**2 + (z - speaker_height) ** 2) ** 0.5
    #     - diff_distance
    # )
    # sol = sympy.solve([expr1, expr2, expr3, expr4], [x, y, z, r])
    # if sol[0][3] > 0:
    #     return np.array([sol[0][0], sol[0][1], sol[0][2]])
    # return np.array([sol[1][0], sol[1][1], sol[1][2]])

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

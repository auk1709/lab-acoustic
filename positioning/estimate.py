import numpy as np
from .get_spectrum_amplitude import get_spectrum_amplitude


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


def positioning_2d(
    reference_spec, reference_ampli, recieved_signal, interval=0.1, output="rect"
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

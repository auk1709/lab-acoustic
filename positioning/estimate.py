import numpy as np
from .get_spectrum_amplitude import get_spectrum_amplitude


def estimate(reference_spec, reference_ampli, file, output="rect"):
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

    test_spec, test_ampli = get_spectrum_amplitude(file)  # テストデータのスペクトルと振幅を取得

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

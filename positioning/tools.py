import numpy as np
import pandas as pd


def export_reference_position():
    """データベース作成時の座標を出力する
    角度の情報から、直交座標系での座標を計算してcsvファイルに出力する
    """
    azimuth_degs = np.arange(-40, 50, 10)
    distance = 1
    x = distance * np.sin(np.radians(azimuth_degs))
    y = distance * np.cos(np.radians(azimuth_degs))
    positions = np.stack([x, y], axis=1)
    positions_df = pd.DataFrame(positions, columns=["x", "y"])
    positions_df.to_csv("reference_position.csv", index=False)


def polar_to_rect(r, theta):
    """極座標から直交座標への変換
    y軸方向を0度として、時計回りに角度を増加させる
    通常の極座標とは異なるため注意

    Parameters
    ----------
    r : float
        極座標の半径
    theta : float
        極座標の角度(rad)
    Returns
    -------
    x : float
        直交座標のx座標
    y : float
        直交座標のy座標
    """
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y


def rect_to_polar(x, y):
    """直交座標から極座標への変換
    y軸方向を0度として、時計回りに角度を増加させる
    通常の極座標とは異なるため注意

    Parameters
    ----------
    x : float
        直交座標のx座標
    y : float
        直交座標のy座標

    Returns
    -------
    r : float
        極座標の半径
    theta : float
        極座標の角度(deg)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(x, y))
    return r, theta


def points_on_test18_line():
    """test18の線上にある点の座標を返す
    シンポのときのtest18の減衰を調べるための座標を返す
    """

    azimuth_deg = 22
    azimuth_rad = np.radians(azimuth_deg)
    distance_base = 1.62
    near_dists = np.arange(1.22, distance_base, 0.05)
    far_dists = np.arange(distance_base, 1.8, 0.05)
    dists = np.append(near_dists, far_dists)
    positions = np.array([polar_to_rect(dist, azimuth_rad) for dist in dists])
    positions_df = pd.DataFrame(positions, columns=["x", "y"])
    positions_df.to_csv("test18_line_positions.csv", index=False)


def export_reference_distance_position():
    """角度ごとに距離を変えたときの座標を出力する
    直交座標系での座標を計算してcsvファイルに出力する
    """
    distances = np.arange(1, 2.25, 0.25)
    azimuth_degs = np.arange(-40, 50, 10)
    dis, azi = np.meshgrid(distances, azimuth_degs)
    x, y = polar_to_rect(dis.flatten(), np.radians(azi.flatten()))
    pos_df = pd.DataFrame({"x": x, "y": y})
    pos_df.to_csv("reference_distance_position.csv")


if __name__ == "__main__":
    export_reference_distance_position()

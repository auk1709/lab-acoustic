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

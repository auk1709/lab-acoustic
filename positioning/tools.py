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


def polar_to_rect_3d(r, theta, phi):
    """極座標から直交座標への変換
    y軸方向を0度として、時計回りに角度を増加させる
    phiはz軸方向の角度、下に向かうと正
    通常の極座標とは異なるため注意

    Parameters
    ----------
    r : float
        極座標の半径
    theta : float
        極座標の角度(rad)
    phi : float
        極座標のz軸方向の角度(rad)
    Returns
    -------
    x : float
        直交座標のx座標
    y : float
        直交座標のy座標
    z : float
        直交座標のz座標
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.cos(phi)
    z = r * -np.sin(phi)
    return x, y, z


def rect_to_polar_3d(x, y, z):
    """直交座標から極座標への変換

    Parameters
    ----------
    x : float
        直交座標のx座標
    y : float
        直交座標のy座標
    z : float
        直交座標のz座標

    Returns
    -------
    r : float
        極座標の半径
    theta : float
        極座標の角度(deg)
    phi : float
        極座標のz軸方向の角度(deg)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arctan2(x, y))
    phi = np.degrees(np.arctan2(-z, y))
    return r, theta, phi


def export_reference_3d_position():
    """3次元のデータベース用の座標を出力する
    mcsでの(1,0.5,1.5)を原点とする
    """
    azimuth_degs = np.arange(-40, 50, 10)
    elevation_degs = np.arange(0, 60, 10)
    azazi, elele = np.meshgrid(azimuth_degs, elevation_degs)
    x, y, z = polar_to_rect_3d(
        1, np.radians(azazi.flatten()), np.radians(elele.flatten())
    )
    df_pos = pd.DataFrame(
        {"x": np.round(x, 3), "y": np.round(y, 3), "z": np.round(z, 3)}
    )
    df_pos["azimuth"] = azazi.flatten()
    df_pos["elevation"] = elele.flatten()
    df_pos["mcs_x"] = df_pos["x"] + 1
    df_pos["mcs_y"] = df_pos["y"] + 0.5
    df_pos["mcs_z"] = df_pos["z"] + 1.5
    x_t, y_t, z_t = polar_to_rect_3d(
        1.15, np.radians(azazi.flatten()), np.radians(elele.flatten())
    )
    df_tail = pd.DataFrame(
        {"x": np.round(x_t, 3), "y": np.round(y_t, 3), "z": np.round(z_t, 3)}
    )
    df_pos["mcs_tail_x"] = df_tail["x"] + 1
    df_pos["mcs_tail_y"] = df_tail["y"] + 0.5
    df_pos["mcs_tail_z"] = df_tail["z"] + 1.5
    df_pos.to_csv("reference_3d_position.csv", index=False)


def export_3d_test_position():
    """3次元測位の計測点の真値を出力する"""
    pos_polar = np.array(
        [
            [1.1, -20, 25],
            [1.1, 0, 25],
            [1.1, 20, 25],
            [1.4, -15, 20],
            [1.4, 0, 20],
            [1.4, 15, 20],
            [1.7, -10, 10],
            [1.7, 0, 10],
            [1.7, 10, 10],
        ]
    )
    df_polar = pd.DataFrame(pos_polar, columns=["distance", "azimuth", "elevation"])
    x, y, z = polar_to_rect_3d(
        pos_polar[:, 0], np.radians(pos_polar[:, 1]), np.radians(pos_polar[:, 2])
    )
    df_pos = pd.DataFrame(
        {"x": np.round(x, 3), "y": np.round(y, 3), "z": np.round(z, 3)}
    )
    df_pos["mcs_x"] = df_pos["x"] + 1
    df_pos["mcs_y"] = df_pos["y"] + 0.5
    df_pos["mcs_z"] = df_pos["z"] + 1.5
    df_pos["distance"] = df_polar["distance"]
    df_pos["azimuth"] = df_polar["azimuth"]
    df_pos["elevation"] = df_polar["elevation"]
    df_pos.to_csv("3d_test_position.csv", index=False)


def get_snr(signal, noise):
    """SN比を求める

    Parameters
    ----------
    signal : float
        信号の大きさ, 雑音込み
    noise : float
        雑音の大きさ

    Returns
    -------
    float
        SN比
    """
    snr = 20 * np.log10((signal - noise) / noise)
    return snr


def dir_height_to_position(azimuth, elevation, height, origin_h):
    """方位角、仰角、高さから座標を計算する

    Parameters
    ----------
    azimuth : float
        方位角(deg)
    elevation : float
        仰角(deg)
    height : float
        高さ
    origin_h : float
        原点の高さ
    """
    x = -(
        (height - origin_h)
        * np.tan(np.radians(azimuth))
        / np.tan(np.radians(elevation))
    )
    y = -((height - origin_h) / np.tan(np.radians(elevation)))
    z = height
    print(elevation)
    print(np.radians(elevation))
    print(np.tan(np.radians(elevation)))
    print(y)
    return x, y, z


def export_reference_3d_position_plane():
    """3次元のデータベース用の座標を出力する
    mcsでの(1,0.5,1.6)を原点とする
    スマホ側の高さを1mで平行移動だけで取れるようにする
    """
    azimuth_degs = np.arange(-40, 41, 5)
    elevation_degs = np.arange(20, 51, 5)
    azazi, elele = np.meshgrid(azimuth_degs, elevation_degs)
    x, y, z = dir_height_to_position(azazi.flatten(), elele.flatten(), 1, 1.6)
    df_pos = pd.DataFrame(
        {"x": np.round(x, 3), "y": np.round(y, 3), "z": np.round(z, 3)}
    )
    df_pos["azimuth"] = azazi.flatten()
    df_pos["elevation"] = elele.flatten()
    df_pos["mcs_x"] = np.round(df_pos["x"] + 1, 3)
    df_pos["mcs_y"] = np.round(df_pos["y"] + 0.5, 3)
    df_pos["mcs_z"] = np.round(df_pos["z"], 3)
    df_pos.to_csv("reference_3d_position.csv", index=False)


if __name__ == "__main__":
    export_reference_3d_position_plane()

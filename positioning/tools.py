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

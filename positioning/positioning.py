# 測位用の関数を集めたファイル
import numpy as np
from scipy import signal
import readwav
import create_db
import estimate

# 信号読み込み
# データベース生成
# 測位
# 変更点
# ・データベースの数
# ・周波数帯(14kHz-24kHz)
# ・np.zeros(144*10*81*76).reshape(81,76,144*10)
# ・speaker_heights[6] -> 0,17,34,50,64,75

# 相互相関を見直す
# 使用する帯域祖14~22kHzに変更


# normalized_x = normalzie(x, amin=-1, amax=1)


if __name__ == "__main__":
    # speaker_heights = [0,17,34,50,64,75]
    houi = -22
    gyouhu = 3
    cdf_array = np.zeros(100 * 3).reshape(100, 3)
    error_array = np.zeros(100)
    distance_error_array = np.zeros(100)
    deg_array = np.zeros(80 * 2).reshape(80, 2)
    # データベース作成
    db1 = create_db()
    # 計測信号読み込み
    test_number = 9
    # スピーカマイク間距離の真値db
    distance_db = [91, 83, 83, 135, 135, 140, 140, 91, 91]
    # -0.3759 0.3893 -0.7477
    dx = 0.0
    dy = 117.0
    dz = 100.0
    signal = readwav("vivesound_0817/test0826_" + str(test_number) + ".wav")
    sec = 150000
    for i in range(0, 80):
        # 残差平方和によって測位
        # cdf_array[i,:] = estimate(db1,signal)
        distance_error_array[i] = estimate(db1, signal) - distance_db[test_number - 1]
        sec += 48000 * 3
    # print(cdf_array[0,0])
    # print(cdf_array[0,1])
    # print(cdf_array)
    # print(np.array2string(deg_array,separator=','))
    # print(np.average(deg_array[:,0]))
    # print(np.average(deg_array[:,1]))
    # print(np.array2string(np.sort(cdf_array),separator=',',precision=2))
    print(np.array2string(distance_error_array, separator=","))
    print(np.average(distance_error_array))
    print(np.std(distance_error_array))

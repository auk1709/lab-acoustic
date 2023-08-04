# dopplerSin: 与えた位置系列の最初の位置で
# 受信を開始する状況を模擬
# マイク・スピーカ間の距離に応じた遅延もあり．

import numpy as np

def dopplerSin(tAry, posMat, vMat, spkPos, f, initPh,smplDr):

    sgEndTm = tAry[-1]
    
    phAry = np.zeros(len(tAry))
    phAry[:] = np.nan
    

    dtSum = 0

    # 各時刻の位相計算 
    # for文の中では初期位相0とする．
    for i in range(len(tAry)):
        # 各時刻でのドップラシフト後の周波数を計算する
        t = tAry[i]

        crntPos = posMat[:, i]
        crntV = vMat[:,i]

        # 法線ベクトル
        # 位相が増える方が正
        # つまり，離れる方が正
        nrmlVc = spkPos - crntPos
        nrmlVc = nrmlVc/np.linalg.norm(nrmlVc)

        # 法線方向の速度
        vNrm = np.dot(crntV,nrmlVc)

        #位置変化分を波動時間軸での時間変化分に変換
        dt = ((vNrm*smplDr)/340)

        dtSum = dtSum + dt

        # 波動時間軸での絶対的時刻
        dgt = t + dtSum

        if dgt < 0:
            continue
        

        if dgt > sgEndTm:
            # 時刻が信号長より長ければ
            # break（該当phAryはnanになる）
            break
        
        
        # 位相変化量
        dPh = f*dgt
        phAry[i] = dPh

    sAry = np.sin((2*np.pi*phAry) + initPh)

    sAry[np.isnan(sAry)] = 0

    # スピーカ・マイク間距離に応じて0をいれる
    initZrCnt = 1 + np.floor((np.linalg.norm(spkPos - posMat[:,0])/340)/smplDr)

    sAry = np.concatenate([np.zeros(int(initZrCnt)), sAry])

    # 瞬時的な周波数を計算
    phAry = phAry[~np.isnan(phAry)]

    dfPhAry = np.diff(phAry)
    frqAry = dfPhAry/smplDr

    return [sAry, frqAry]
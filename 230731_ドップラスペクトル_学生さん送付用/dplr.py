# dplr.py 移動しながら受信したときのスペクトル計算

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import dopplerSin
import calibPh
from scipy import signal
import time
import japanize_matplotlib

# 信号関係の設定・準備
fs = 48000*20
smplDr = 1/fs
lrgT = 0.004
f = 15000

smplN = lrgT/smplDr
tAry = smplDr*np.arange(0,smplN)
tCnt = len(tAry)

smplN2 = (lrgT/2)/smplDr
twAry = smplDr*np.arange(0,smplN2)

exp1 = np.exp(1j*2*np.pi*f*tAry)

# 速度ベクトルの設定
vNorm = 1.5
vCnstVc =  np.array([1,0])
vCnstVc = vNorm * (vCnstVc/np.linalg.norm(vCnstVc))
# vMat: サンプリング時の速度ベクトル（以下では等速を仮定）
vMat = numpy.matlib.repmat(vCnstVc, tCnt,1)
vMat = vMat.transpose()

# スピーカ位置
spkPos = np.array([0,0])

# マイクの初期位置の設定（複数箇所を設定）
xInitAry = np.arange(-1,1.2,0.2)

# マイクの初期位置の設定（1箇所だけの場合は以下のように設定）
# xInitAry = np.arange(-1,-0.7,1)

maxFAry = []

fg2 = plt.figure('Figure 2')
fg3 = plt.figure('Figure 3')
fg4 = plt.figure('Figure 4')

# 送信信号の初期位相
initPh = np.deg2rad(0)

# 初期位置毎に計算
for ix in xInitAry:

    # 各サンプリング時刻での位置
    posMat = np.zeros(vMat.shape)
    posMat[:,0] = [ix, 1]

    for i in np.arange(1,tCnt):
        posMat[:,i] = posMat[:,i-1] + vMat[:,i]*smplDr

    # 移動時の受信信号模擬
    [sAry,tmpFAry] = dopplerSin.dopplerSin(tAry, posMat, vMat, spkPos, f, initPh, smplDr)

    sAryZr = np.concatenate([sAry, np.zeros(int(np.round(0.1*fs)))])

    # Matched filter
    mf1 = np.convolve(sAryZr, np.flip(exp1), 'valid')
    amf = abs(mf1)

    # ToAのインデックス推定（位相値で補正）
    pmf = np.rad2deg(calibPh.calibPh(np.angle(mf1)))

    pkIndx = np.argmax(amf)
    pdAtPkIndx = pmf[pkIndx]

    phPerSmpl1 = np.rad2deg(calibPh.calibPh(2*np.pi*f*smplDr))
    pkIndx = np.round(pkIndx+((pdAtPkIndx-90+np.rad2deg(initPh))/phPerSmpl1))

    # 信号の切り出し
    clpInit1 = int(pkIndx)
    clpEnd1 = int(clpInit1 +  np.round(fs*lrgT)-1)
    clpRcv1 = sAryZr[clpInit1:clpEnd1+1]
    clpRcv1 = clpRcv1/np.max(np.abs(clpRcv1))

    # 切り出した受信信号の振幅スペクトル
    afc = np.abs(np.fft.fft(clpRcv1))

    # 送信周波数の正弦波の振幅スペクトル
    tmpSinNoClb = np.sin(2*np.pi*f*tAry)
    safc = np.abs(np.fft.fft(tmpSinNoClb))


    # 周波数推定（信号中心付近のみを利用）
    clpInit = int(pkIndx + np.round(fs*lrgT/2))
    clpEnd = int(clpInit + np.round(fs*lrgT/2)-1)
    clpRcv = sAryZr[clpInit:clpEnd+1]

    fAry = np.arange(f-100,f+100+0.01,0.01)
    fCnt = len(fAry)

    afAry = np.zeros(fCnt)
    afAry[:] = np.nan

    wndw = signal.hann(len(twAry))
    clpRcv = wndw * clpRcv

    for iaf in range(fCnt):
        tmpExp = np.exp(1j*2*np.pi*fAry[iaf]*twAry)
        tmpExp = wndw * tmpExp
        afAry[iaf] = abs(np.dot(tmpExp,clpRcv))

    maxIndx = np.argmax(afAry)

    maxF = fAry[maxIndx]
    print('maxF: %f\n' % maxF)

    maxFAry.append(maxF)

    # 推定周波数を用いてMatched filter
    tmpExpMaxF = np.exp(1j*2*np.pi*maxF*tAry)

    mf2 = (np.convolve(sAryZr, np.flip(tmpExpMaxF), 'valid'))

    amf2 = abs(mf2)
    pmf2 = np.rad2deg(calibPh.calibPh(np.angle(mf2)))

    tmpMax2 = np.max(amf2)
    pkIndx2 = np.argmax(amf2)
    
    pdAtPkIndx2 = pmf2[pkIndx2]

    # 位相値によるインデックスの補正  
    phPerSmpl = np.rad2deg(calibPh.calibPh(2*np.pi*maxF*smplDr))
    pkIndx2 = np.round(pkIndx2 + ((pdAtPkIndx2-90+np.rad2deg(initPh))/phPerSmpl))
    pkIndx2 = int(pkIndx2)

    # 推定周波数を用いたMF結果による信号の切り出し（念のため）と
    # その振幅スペクトル計算
    clpInit2 = int(pkIndx2)
    clpEnd2 = int(clpInit2 + np.round(fs*lrgT)-1)
    clpRcv2 = sAryZr[clpInit2:clpEnd2+1]
    clpRcv2 = clpRcv2/np.max(np.abs(clpRcv2))
    afc2 = np.abs(np.fft.fft(clpRcv2))

    # 推定周波数の正弦波の振幅スペクトル
    tmpSinClb = np.sin(2*np.pi*maxF*tAry)
    safc2 = np.abs(np.fft.fft(tmpSinClb))

    # 振幅スペクトルの描画
    fIndx = int((f/fs)*len(clpRcv1))
    ax2 = fg2.add_subplot(1,1,1)
    ax2.plot(afc,label='受信信号',marker='o',markersize=10)
    ax2.plot(safc,label='送信周波数の正弦波',marker='x',markersize=10)
    ax2.plot(safc2,label='推定周波数の正弦波',marker='d',markersize=10)
    ax2.plot(afc2, label='推定周波数MFで切り出した受信信号',marker='.',markersize=10)
    ax2.set_xlim(fIndx-20,fIndx+20)
    ax2.set_title('振幅スペクトル')
    ax2.legend()
    plt.show(block=False)

    # 周波数推定に関するプロット（debug用）
    fg3.clf()
    ax31 = fg3.add_subplot(3,1,1)
    # ax1.cla()
    ax31.plot(amf2,'.-')
    ax31.plot(pkIndx2, amf2[pkIndx2],'o')
    ax31.set_xlim([0,6000])
    ax31.set_title('推定fでのMF')

    ax32 = fg3.add_subplot(3,1,2)
    ax32.plot(amf2,'.-')
    ax32.plot(pkIndx2, amf2[pkIndx2],'o')
    ax32.set_xlim([pkIndx2-50,pkIndx2+50])

    ax33 = fg3.add_subplot(3,1,3)
    ax33.plot(pmf2,'.-')
    ax33.plot(pkIndx2, pmf2[pkIndx2],'o')
    ax33.set_title(pmf[pkIndx2])
    ax33.set_xlim([pkIndx2-50,pkIndx2+50])

    # print('pkIndx2:%d\n' %  pkIndx2)
    
    #「送信信号」と「受信信号」と「推定周波数の正弦波」のプロット（Debug用）
    fg4.clf()
    ax4 = fg4.add_subplot(1,1,1)
    ax4.plot(sAryZr[pkIndx2-20:int(pkIndx2+np.round(fs*lrgT)+200)], label='観測')
    ax4.plot(np.sin(2*np.pi*f*tAry), label='送信f正弦波')
    ax4.plot(np.sin(2*np.pi*maxF*tAry), label='推定f正弦波')
    ax4.set_xlim([0,400]) 
    ax4.set_title('「観測」と「推定fの正弦波」'+' 初期位置:' + str(ix))
    ax4.legend()

    plt.show(block=False)
    print('')

maxFAry = np.array(maxFAry)

plt.figure()
plt.plot(xInitAry, maxFAry/1000, 'x-')
plt.ylabel('周波数 [kHz]')
plt.xlabel('初期位置(x)')
plt.title('推定周波数')

plt.show(block=True)

print('')

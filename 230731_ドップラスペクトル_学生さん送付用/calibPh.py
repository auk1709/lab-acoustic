import numpy as np

def calibPh(ph):
    # -piより小さい，あるいはpiより大きい位相値を
    # -piからpiの範囲内の値に変更する関数
    ph = np.matrix(ph)
    rtn = np.zeros(ph.shape)
    
    rw = rtn.shape[0]
    clm = rtn.shape[1]

    for ilw in range(rw): 
        for iclm in range(clm):
            phElmnt = ph[ilw, iclm]
            
            if phElmnt >= 0:
                phElmnt = phElmnt % (2*np.pi)
            else:
                phElmnt = -1*(abs(phElmnt) % (2*np.pi))
            
            if phElmnt > np.pi:
                tmpVal = phElmnt - np.pi
                rtn[ilw, iclm] = - np.pi + tmpVal
                continue
                        
            if phElmnt <= (-1*np.pi):
                tmpVal = abs(phElmnt - (-np.pi))
                rtn[ilw, iclm] = np.pi - tmpVal
                continue
                        
            rtn[ilw, iclm] = phElmnt
            continue
    
    if rw == 1 and clm == 1:
    # rw==1, clm==1のとき数値に戻す
        rtn = rtn[0][0]
    elif rw == 1:  
    # rw==1のとき1次元のリストに戻す
        rtn = rtn[0]

    

    return rtn

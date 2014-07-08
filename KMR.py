from __future__ import division
import random
import matplotlib.pyplot as plt
import numpy as np
from discrete_rv import DiscreteRV
from mc_tools import mc_sample_path
p = 1/3
n = 10
t = 100000
epsilon = 0.1
payoff = [[[4,4],[0,3]],[[3,0],[2,2]]]
one_payoff =np.transpose(np.transpose(payoff)[0])
#遷移行列をつくる
X = np.zeros((n+1,n+1))
expay0 = np.empty(2) 
expay1 = np.empty(2) 
for k in range(1,n):
    #直前まで0だった人が選ばれた時の行動選択による期待利得
    expay0[0] = one_payoff[0][0]*(n-k-1)/(n-1)+one_payoff[0][1]*k/(n-1)
    expay0[1] = one_payoff[1][0]*(n-k-1)/(n-1)+one_payoff[1][1]*k/(n-1)
    #直前まで1だった人が選ばれた時の行動選択による期待利得
    expay1[0] = one_payoff[0][0]*(n-k)/(n-1)+one_payoff[0][1]*(k-1)/(n-1)
    expay1[1] = one_payoff[1][0]*(n-k)/(n-1)+one_payoff[1][1]*(k-1)/(n-1)
    if expay1[0]>expay1[1]:
        X[k][k-1]=(k/n)*(1-epsilon*0.5) #k人からk-1人になる確率
        X[k][k]=(k/n)*epsilon*0.5
    elif expay1[0]==expay1[1]:
        X[k][k-1]=(k/n)*0.5
        X[k][k]=(k/n)*0.5
    else:
        X[k][k-1]= (k/n)*epsilon*0.5
        X[k][k] = (k/n)*((1-epsilon)+epsilon*0.5)
    if expay0[1]>expay0[0]:
        X[k][k+1]=((n-k)/n)*((1-epsilon)+epsilon*0.5) #k人からk+1人になる確率
        X[k][k] += ((n-k)/n)*epsilon*0.5 #X[k][k]は上でも定めているので上書きでなく加えている
    elif expay0[1]==expay0[0]:
        X[k][k+1] = ((n-k)/n)*0.5
        X[k][k] += ((n-k)/n)*0.5
    else:
        X[k][k+1] = ((n-k)/n)*epsilon*0.5
        X[k][k] += ((n-k)/n)*((1-epsilon)+epsilon*0.5)
X[0][0] = (1-epsilon)+epsilon*0.5
X[0][1] = epsilon*0.5
X[n][n-1] = epsilon*0.5
X[n][n] = (1-epsilon)+epsilon*0.5
x_0 = np.random.binomial(n,p)  # determine X_0
xs=mc_sample_path(X,init=x_0,sample_size=t)

"""
確率分布を入れる場合は長さn+1のリストを入れなければいけないらしく
面倒なので各人が確率pで行動1を選択するという二項分布を作って整数値で代入
"""
plt.plot(xs, 'b-', label='X_t')
plt.legend()
plt.show()    

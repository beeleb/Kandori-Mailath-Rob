from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random

# 基本動作 oneKMRの名前でメソッドにする予定
payoff = [[[4,4],[0,3]],[[3,0],[2,2]]]
n = 100  # the number of players
T = 1000
upsilon = 0  # ε
x_0 = random.randint(0,n)  # determine X_0
one_payoff =np.transpose(np.transpose(payoff)[0])
current_x = x_0
x_list = [x_0]
for i in range(T):
    x_list.append(current_x)
    p = random.uniform(0,1)
    if p > current_x/n:
        former_act = 0  # former_actは変更機会持ちのプレイヤーの直前の行動
        act1_number = current_x  #act1_numberは変更機会のないプレイヤーのうち行動1をとる人数
    else:
        former_act = 1
        act1_number = current_x-1
    match1_probability = act1_number/(n-1)
    probabilities = [1-match1_probability, match1_probability]
    q = random.uniform(0,1)
    if q > upsilon:  #確率1-ε
        exp_pay = np.dot(one_payoff,probabilities)
        if exp_pay[0] == exp_pay[1]:
            chosen_act = random.randint(0,1)
        else:
            chosen_act = np.argmax(exp_pay)  # chosen_actは変更機会持ちのプレイヤーが実際に選んだ行動
    else:  # 確率ε
        chosen_act = random.randint(0,1)
    current_x = current_x -former_act + chosen_act  # 1→0なら1減り0→1なら1増える
# X_tの推移
"""
plt.plot(x_list, 'b-', label='X_t')
plt.legend()
plt.show()
"""

# X_tの平均の推移
"""
x_sums = np.cumsum(x_list)
ave_x_list =[]
for i in range(T):
    ave_x = x_sums[i]/(i+1)
    ave_x_list.append(ave_x)
plt.plot(ave_x_list, 'b-', label='X_t')
plt.legend()
plt.show()
"""
#X_(T-1)のヒストグラム(途中)
"""
last_x_list = []
for i in range(200):
    oneKMR(hogehoge)
    last_x_list.append(current_x)
plt.hist(last_x_list, bins=10)
plt.show()
"""
    
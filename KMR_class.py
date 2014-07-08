from __future__ import division
import random
import matplotlib.pyplot as plt
import numpy as np
from discrete_rv import DiscreteRV
from mc_tools import mc_sample_path
"""
p = 1/3
n = 10
t = 100000
epsilon = 0.1
payoff = [[[4,4],[0,3]],[[3,0],[2,2]]]
"""
def set_pay(payoff):
    if type(payoff[0][0]) == list or tuple and type(payoff[0][0][0]) == int:
        one_payoff = np.transpose(np.transpose(payoff)[0])
        print'OK.'
    elif type(payoff[0][0]) == int:
        one_payoff = payoff
        print'OK.'
    else:
        print 'Wrong Data Type ! '


class KMR:
    def __init__(self,n,p,epsilon):
        self.one_pay = one_payoff
        self.epsi = epsilon
        self.X = X
        self.n = n
        self.p = p
        self.xs = xs
        
    def det_X(self):#遷移行列をつくる
        self.X = np.zeros((self.n+1,self.n+1))
        expay0 = np.empty(2) 
        expay1 = np.empty(2) 
        for k in range(1,self.n):
            #直前まで0だった人が選ばれた時の行動選択による期待利得
            expay0[0] = self.one_pay[0][0]*(self.n-k-1)/(self.n-1)+self.one_pay[0][1]*k/(self.n-1)
            expay0[1] = self.one_pay[1][0]*(self.n-k-1)/(self.n-1)+self.one_pay[1][1]*k/(self.n-1)
            #直前まで1だった人が選ばれた時の行動選択による期待利得
            expay1[0] = self.one_pay[0][0]*(self.n-k)/(self.n-1)+self.one_pay[0][1]*(k-1)/(self.n-1)
            expay1[1] = self.one_pay[1][0]*(self.n-k)/(self.n-1)+self.one_pay[1][1]*(k-1)/(self.n-1)
            if expay1[0] > expay1[1]:
                self.X[k][k-1]=(k/self.n)*(1-self.epsi*0.5) #k人からk-1人になる確率
                self.X[k][k]=(k/self.n)*self.epsi*0.5
            elif expay1[0]==expay1[1]:
                self.X[k][k-1]=(k/self.n)*0.5
                self.X[k][k]=(k/self.n)*0.5
            else:
                self.X[k][k-1]= (k/self.n)*self.epsi*0.5
                self.X[k][k] = (k/self.n)*(1-self.epsi*0.5)                
            if expay0[1]>expay0[0]:
                self.X[k][k+1]=((self.n-k)/self.n)*(1-self.epsi*0.5) #k人からk+1人になる確率
                self.X[k][k]+=((self.n-k)/self.n)*self.epsi*0.5 #X[k][k]は上でも定めているので上書きでなく加えている
            elif expay0[1]==expay0[0]:
                self.X[k][k+1] = ((self.n-k)/self.n)*0.5
                self.X[k][k] += ((self.n-k)/self.n)*0.5
            else:
                self.X[k][k+1] = ((self.n-k)/self.n)*epsilon*0.5
                self.X[k][k] += ((self.n-k)/self.n)*(1-self.epsi*0.5)
        self.X[0][0] = 1-self.epsi*0.5
        self.X[0][1] = self.epsi*0.5
        self.X[self.n][self.n-1] = self.epsi*0.5
        self.X[self.n][self.n] = 1-self.epsi*0.5
    
    def sim(self, t):
        x_0 = np.random.binomial(self.n,self.p)  # determine X_0
        self.xs = mc_sample_path(self.X,init=x_0,sample_size=t)
        
    def simplot(self,t):
        self.sim(t)
        plt.plot(self.xs, 'b-', label='X_t')
        plt.legend()
        plt.show()
        



"""
入力の例
payoff = [[4,0],[3,2]]
set_pay(payoff)
f = KMR(10,1/3,0.1)  # (人数,二項分布の確率,ε)
f.det_X()
f.simplot(100000)
"""
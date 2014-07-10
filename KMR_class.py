from __future__ import division
import random
import matplotlib.pyplot as plt
import numpy as np
from discrete_rv import DiscreteRV
from mc_tools import mc_sample_path, mc_compute_stationary


def set_pay(pf):  # 2人分の利得表を入れると1人分に変形して返します
    global one_payoff
    WDT='Wrong Data Type ! '
    if type(pf) == int:
        print WDT
    elif type(pf[0]) ==int:
        print WDT
    else:
        if type(pf[0][0]) == list or type(pf[0][0]) ==tuple:
            one_payoff = np.transpose(np.transpose(pf)[0])
            print'OK. one_payoff is'
            print str(one_payoff)
        elif type(pf[0][0]) == int:
            one_payoff = pf
            print'OK. one_payoff is'
            print str(one_payoff)
        else:
            print WDT
#set_payを使わない場合はone_payoffに二重のlistかtupleでプレイヤー1人の利得を入れてください


class KMR:
    def __init__(self,n,p,epsilon):
        self.one_pay = one_payoff
        self.epsi = epsilon
        self.n = n
        self.X = 0
        self.p = p
        self.xs = 0
        self.x_0 = 0
        self.x_ts = []
        
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
                self.X[k][k] += ((self.n-k)/self.n)*self.epsi*0.5 #X[k][k]は上でも定めているので上書きでなく加えている
            elif expay0[1]==expay0[0]:
                self.X[k][k+1] = ((self.n-k)/self.n)*0.5
                self.X[k][k] += ((self.n-k)/self.n)*0.5
            else:
                self.X[k][k+1] = ((self.n-k)/self.n)*self.epsi*0.5
                self.X[k][k] += ((self.n-k)/self.n)*(1-self.epsi*0.5)
        self.X[0][0] = 1-self.epsi*0.5
        self.X[0][1] = self.epsi*0.5
        self.X[self.n][self.n-1] = self.epsi*0.5
        self.X[self.n][self.n] = 1-self.epsi*0.5
    
    def set_x0(self):
        self.x_0 = np.random.binomial(self.n,self.p)  # determine X_0
    
    def sim(self, t):
        self.det_X()
        self.set_x0()
        self.xs = mc_sample_path(self.X,init=self.x_0,sample_size=t)
        
    def simplot(self,t):
        self.sim(t)
        plt.plot(self.xs, 'b-', label='X_t')
        tit = str(self.n)+' people,  '+'p = '+str(round(self.p,2))+'\n'+'epsilon = '+str(self.epsi)+',  time length = '+str(t)
        plt.title(tit)
        plt.legend()
        plt.show()
        
    def hist(self,t,times): 
        self.x_ts = []
        self.det_X()
        self.set_x0()
        for i in range(times):
            self.sim(t)
            self.x_ts.append(self.xs[-1])
        ax = plt.subplot(111)
        ax.hist(self.x_ts, alpha=0.6, bins=10)     
        tit = str(self.n)+' people,  '+'p = '+str(round(self.p,2))+',  '+'epsilon = '+str(self.epsi)+'\n'+'time length = '+str(t)+',  '+str(times)+'times'
        ax.set_title(tit)
            
    def histplot(self,t,times):
        self.hist(t,times)
        plt.show()
        
    def equilibrium(self):
        self.det_X()
        Y = mc_compute_stationary(self.X)
        tit = str(self.n)+' people,  '+'epsilon = '+str(self.epsi)
        plt.bar(range(self.n+1), Y, align='center')
        plt.xlim([-0.5, self.n+0.5])
        plt.ylim([0,1])
        plt.title(tit)        
        plt.show()


"""
#入力の例
payoff = [[[4,4],[0,3]],[[3,0],[2,2]]]
set_pay(payoff)
f = KMR(10,1/3,0.01)  # (人数,二項分布の確率,ε)
f.simplot(100000) # (時間の長さ)
#f.histplot(10000,1000)  # (時間の長さ、回数)
"""

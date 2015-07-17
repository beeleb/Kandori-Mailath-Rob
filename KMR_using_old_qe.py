# -*- coding: utf-8 -*-
"""
Author:
KMR (Kandori-Mailath-Rob) Model
"""
from __future__ import division
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt



def kmr_markov_matrix(p, N, epsilon, mode='sequential'):
    # mode requires 'sequential' or 'simultaneous'
    """
    Generate the transition probability matrix for the KMR dynamics with
    two acitons.
    """
    # KMR の遷移確率行列を返す関数を書く
    P = np.zeros((N+1, N+1))  # たとえばこんな感じで始めて P の要素を埋めていく
    
    if mode == 'sequential':  # 逐次改訂
        expay = np.empty((2,2))  
        for k in range(1,N):
            #直前まで0だった人が選ばれた時の行動選択による期待利得
            expay[0][0] = p[0][0]*(N-k-1)/(N-1)+p[0][1]*k/(N-1)
            expay[0][1] = p[1][0]*(N-k-1)/(N-1)+p[1][1]*k/(N-1)
            #直前まで1だった人が選ばれた時の行動選択による期待利得
            expay[1][0] = p[0][0]*(N-k)/(N-1)+p[0][1]*(k-1)/(N-1)
            expay[1][1] = p[1][0]*(N-k)/(N-1)+p[1][1]*(k-1)/(N-1)
            if expay[1][0] > expay[1][1]:
                P[k][k-1]=(k/N)*(1-epsilon*0.5) #k人からk-1人になる確率
                P[k][k]=(k/N)*epsilon*0.5
            elif expay[1][0] == expay[1][1]:
                P[k][k-1] = (k/N)*0.5
                P[k][k] = (k/N)*0.5
            else:
                P[k][k-1]= (k/N)*epsilon*0.5
                P[k][k] = (k/N)*(1-epsilon*0.5)                
            if expay[0][1]>expay[0][0]:
                P[k][k+1]=((N-k)/N)*(1-epsilon*0.5) #k人からk+1人になる確率
                P[k][k] += ((N-k)/N)*epsilon*0.5 #X[k][k]は上でも定めているので上書きでなく加えている
            elif expay[0][1]==expay[0][0]:
                P[k][k+1] = ((N-k)/N)*0.5
                P[k][k] += ((N-k)/N)*0.5
            else:
                P[k][k+1] = ((N-k)/N)*epsilon*0.5
                P[k][k] += ((N-k)/N)*(1-epsilon*0.5)
        P[0][0] = 1-epsilon*0.5
        P[0][1] = epsilon*0.5
        P[N][N-1] = epsilon*0.5
        P[N][N] = 1-epsilon*0.5
            
    elif mode == 'simultaneous':  # 同時改訂
        list=[]
        for i in range(N+1):
            list.append(i)
        expay = np.empty(2)  
        for k in range(0,N+1):
            #各人の行動選択による期待利得
            expay[0] = pay[0][0]*(N-k)/(N)+pay[0][1]*k/(N)
            expay[1] = pay[1][0]*(N-k)/(N)+pay[1][1]*k/(N)
            if expay[0] > expay[1]:
                P[k] = binom.pmf(list,N,epsilon)
            elif expay[0] == expay[1]:
                P[k] = binom.pmf(list,N,0.5)
            else:
                P[k] = binom.pmf(list,N,1-epsilon)
    else:
        print 'The mode '+ mode +'is Unknown. Check your input.'
        sys.exit()        

    return P


class KMR(object):
    """
    Class representing the KMR dynamics with two actions.
    """
    def __init__(self, p, N, epsilon,mode='sequential'):
        
        self.num = N
        self.epsi = epsilon
        self.profits = p  
        self.mode = mode
        self.matrix = kmr_markov_matrix(p, N, epsilon,mode)
        self.mc = qe.MarkovChain(self.matrix)

        

    def simulate(self, ts_length, init=None, num_reps=None):
        
        if init is None:
            if num_reps is None:
                init = np.random.randint(0, self.num, 1)
            else:
                init = np.random.randint(0, self.num, num_reps)
        
        
        elif isinstance(init, int):
            if isinstance(num_reps, int):
                init = [init] * num_reps
                
        return self.mc.simulate(sample_size=ts_length)

        """
        Simulate the dynamics.
        Parameters
        ----------
        ts_length : scalar(int)
            Length of each simulation.
        init : scalar(int) or array_like(int, ndim=1),
               optional(default=None)
            Initial state(s). If None, the initial state is randomly
            drawn.
        num_reps : scalar(int), optional(default=None)
            Number of simulations. Relevant only when init is a scalar
            or None.
        Returns
        -------
        X : ndarray(int, ndim=1 or 2)
            Array containing the sample path(s), of shape (ts_length,)
            if init is a scalar (integer) or None and num_reps is None;
            of shape (k, ts_length) otherwise, where k = len(init) if
            init is an array_like, otherwise k = num_reps.
        """
    def plot(self, ts_length, init=None, num_reps=None):
        sim = self.simulate(ts_length, init, num_reps)
        plt.plot(sim, 'b-', label='X_t')
        tit = str(self.num)+' people,  '+self.mode+' mode, '+'p = '+str(self.profits)+'\n'+'epsilon = '+str(self.epsi)+',  time length = '+str(ts_length)
        plt.title(tit)
        plt.ylim([0,self.num])  
        plt.legend()
        plt.show()

    def hist(self, ts_length, init=None, num_reps=None): 
        sim = self.simulate(ts_length, init, num_reps)
        plt.hist(sim, bins=10)
        plt.ylim([0,ts_length])
        tit = str(self.num)+' people,  '+self.mode+' mode, '+'p = '+str(round(self.profits,2))+',  '+'\n'+'epsilon = '+str(self.epsi)+',  time length = '+str(ts_length)
        plt.title(tit)
        plt.show()
        
    def equilibrium(self):
        Y = self.compute_stationary_distribution()
        #Y = mc_compute_stationary(self.X)
        tit = str(self.num)+' people,  '+self.mode+' mode, '+'epsilon = '+str(self.epsi)
        plt.bar(range(self.num+1), Y, align='center')
        plt.xlim([-0.5, self.num+0.5])
        plt.ylim([0,1])
        plt.title(tit)        
        plt.show()


    def compute_stationary_distribution(self):
        # mc.stationary_distributions の戻り値は2次元配列．
        # 各行に定常分布が入っている (一般には複数)．
        # epsilon > 0 のときは唯一，epsilon == 0 のときは複数ありえる．
        # espilon > 0 のみを想定して唯一と決め打ちするか，
        # 0か正かで分岐するかは自分で決める．
        return self.mc.stationary_distributions[0]  # これは唯一と決め打ちの場合
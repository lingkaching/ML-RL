#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:12:55 2017

@author: KACHING
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def bellman(s,act,V,gamma,p_head):

    s_next1=s+act
    s_next2=s-act
    if s_next1==100:
        value=p_head*(0+gamma*V[s_next1])+\
                      (1-p_head)*(0+gamma*V[s_next2])
    else:
        value=p_head*(0+gamma*V[s_next1])+\
                      (1-p_head)*(0+gamma*V[s_next2])
    return value
        
    
    
    
def Value_Es():
    gamma=1
    n_non_term_states=99
    n_states=n_non_term_states+2
    #initialise state value function
    #state value funciton: the prob to reach the goal
    V=np.zeros((n_states))
    V[0]=0
    V[-1]=1
    #set parameters
    p_head=0.4
    p_tail=1-p_head
    Theta=1e-8
    Max_Inter_N=1000
    Delta=np.inf
    iterCnt=0
    PlotIters=[1,2,3,10]
    Value_Estimates=np.zeros((4,n_non_term_states))
    while Delta>Theta and iterCnt<Max_Inter_N:
        iterCnt+=1
        Delta=0
        for s in range(1,n_non_term_states+1):
            tmp=V[s]
            acts=range(1,(min(s,n_non_term_states+1-s)+1))
            Q=[]
            for act in acts:
                Q.append(bellman(s,act,V,gamma,p_head))
            V[s]=max(Q)
            Delta=max(Delta,np.abs(tmp-V[s]))
        if iterCnt==1:
            Value_Estimates[0,:]=V[1:100]
        elif iterCnt==2:
            Value_Estimates[1,:]=V[1:100]
        elif iterCnt==3:
            Value_Estimates[2,:]=V[1:100]
        elif iterCnt==10:
            Value_Estimates[3,:]=V[1:100]
        else:
            pass
    print(Value_Estimates)
    plt.figure()
    plt.plot(range(1,100),Value_Estimates[0,:])
    plt.plot(range(1,100),Value_Estimates[1,:])
    plt.plot(range(1,100),Value_Estimates[2,:])
    plt.plot(range(1,100),Value_Estimates[3,:])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.show()

def Greedy_Poly():
    gamma=1
    n_non_term_states=99
    n_states=n_non_term_states+2
    #initialise state value function
    #state value funciton: the prob to reach the goal
    V=np.zeros((n_states))
    V[0]=0
    V[-1]=1
    #set parameters
    p_head=0.4
    Theta=1e-6
    Max_Inter_N=100000
    Delta=np.inf
    iterCnt=0
    Value_Estimates=np.zeros((4,n_non_term_states))
    Best=np.ones((n_states))*(-np.inf)
    BestAct=np.ones((n_states))*(-np.inf)
    
    while Delta>Theta:
        iterCnt+=1
        Delta=0
        Best=np.ones((n_states))*(-np.inf)
        BestAct=np.ones((n_states))*(-np.inf)

        for s in range(1,n_non_term_states+1):
            tmp=V[s]
            acts=range(1,(min(s,n_non_term_states+1-s)+1))
            
            for act in acts:
                Q=bellman(s,act,V,gamma,p_head)
                if Q>Best[s]:
                    Best[s]=Q
                    BestAct[s]=act
            V[s]=Best[s]
            Delta=max(Delta,np.abs(tmp-V[s]))
    
    print(BestAct)
    plt.figure()
    plt.scatter(range(1,100),BestAct[1:100])
    plt.xlabel('Capital')
    plt.ylabel('Final Policy')
    plt.show()

def main():
 #   Value_Es()
    Greedy_Poly()
    
if __name__=='__main__':
    main()

            
            
                
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:28:04 2017

@author: KACHING
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def bellman(s,act,V,gamma,p_head):

    s_next1=int(s+act)
    s_next2=int(s-act)
    if s_next1==100:
        value=p_head*(0+gamma*V[s_next1])+\
                      (1-p_head)*(0+gamma*V[s_next2])
    else:
        value=p_head*(0+gamma*V[s_next1])+\
                      (1-p_head)*(0+gamma*V[s_next2])
    return value
        
    
    
    
def Poly_Eva(V,pol_pi,gamma):    
    #set parameters
    n_non_term_states=99
    p_head=0.4
    Theta=1e-8
    Max_Inter_N=100
    Delta=np.inf
    iterCnt=0
    while Delta>Theta and iterCnt<Max_Inter_N:
        iterCnt+=1
        Delta=0
        for s in range(1,n_non_term_states+1):
            tmp=V[s]
            V[s]=bellman(s,pol_pi[s],V,gamma,p_head)
            Delta=max(Delta,np.abs(tmp-V[s]))
    return V
            
def Poly_Imp(V,pol_pi,gamma):
    #set parameters
    n_non_term_states=99
    n_states=n_non_term_states+2
    policystable=1
    p_head=0.4
    BestValue=np.ones((n_states))*(-np.inf)
    BestAct=np.ones((n_states))*(-np.inf)
    for s in range(1,n_non_term_states+1):
        acts=range(1,(min(s,n_non_term_states+1-s)+1))
        for act in acts:
            Q=bellman(s,act,V,gamma,p_head)
            if Q>BestValue[s]:
                BestValue[s]=Q
                BestAct[s]=act
        if pol_pi[s]!=BestAct[s]:
            policystable=0
            pol_pi[s]=BestAct[s]
    return pol_pi,policystable

def main():
    gamma=1
    n_non_term_states=99
    n_states=n_non_term_states+2
    #initialise state value function
    #state value funciton: the prob to reach the goal
    V=np.zeros((n_states))
    V[0]=0
    V[-1]=1
    #initialise pol_pi
    pol_pi=np.ones((n_states))
    policystable=0
    cnt=0
    while not policystable and cnt<1000:
        cnt+=1
        print(cnt)
        V=Poly_Eva(V,pol_pi,gamma)
#        pdb.set_trace()
        pol_pi,policystable=Poly_Imp(V,pol_pi,gamma)
    print(pol_pi)
    plt.figure()
    plt.scatter(range(1,100),pol_pi[1:100])
    
        
        
    
if __name__=='__main__':
    main()


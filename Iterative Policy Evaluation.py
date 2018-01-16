#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:44:35 2017

@author: KACHING
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pdb

#==============================================================================
#Performs iterative policy evaluation on 
#the state-value function for the grid world example.
#where the policy is uniform random steps in either direction
#==============================================================================
gamma=1 #no discount
sideL=6 #actually sideL=4, 5 is more convinent to callate
nGrids=sideL**2
V=np.zeros((sideL,sideL))
# parameters
Max_N_Inters=100
iterCnt=0
Theta=1e-6
Delta=1e10
#a uniform policy to be evaluated
pol_pi=0.25
while Delta>Theta and iterCnt<Max_N_Inters:
    Delta=0
    for i in range(1,5):
        for j in range(1,5):
            if (i==1 and j==1) or (i==4 and j==4):
                continue
            Tmp=V[i,j]
            V[i,j]=pol_pi*((-1+gamma*V[i-1,j])+(-1+gamma*V[i+1,j])+\
                    (-1+gamma*V[i,j-1])+(-1+gamma*V[i,j+1]))
            Delta=max(Delta,np.abs(Tmp-V[i,j]))
    V[0,:]=V[1,:]
    V[-1,:]=V[-2,:]
    V[:,0]=V[:,1]
    V[:,-1]=V[:,-2]
    iterCnt+=1
V[1:5,1:5]
            


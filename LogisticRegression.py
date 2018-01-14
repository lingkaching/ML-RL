#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:27:49 2018

@author: KACHING
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def loaddata():
    data=pd.read_csv('testdata.csv')
    x=np.asarray(data.iloc[:,0:2])
    y=np.asarray(data.iloc[:,2])
    y=y.reshape(y.shape[0],-1)
    return x,y

def sigmoid(z):
    h=1/(1+np.exp(-z))
    return h
    
def optimisepara(x,y):
    #optimise the likihood function using gradient ascent method
    #L(theta)=\Sigma(y_i(theta'x_i))-\Sigma(log(1+exp(theta'x_i))
    m,n=np.shape(x)
    x=np.column_stack((np.ones((m,1)),x))    
    theta=np.ones((n+1,1))
    alpha=0.01
    i=0
    while i<500:
        temp1=np.dot(x.T,y)
        temp2=np.dot(x,theta)
        temp2=np.exp(temp2)/(1+np.exp(temp2))            
        gra=temp1-np.dot(x.T,temp2)
        theta=theta+alpha*gra
        i+=1
    return theta
        
def scatt(x,y,theta):
    n=len(y)
    j,k=0,0
    temp1=np.zeros((1,2))
    temp2=np.zeros((1,2))
    for i in range(0,n):
        if y[i]==1:
            temp1=np.row_stack((temp1,x[i,:]))
        else:
            temp2=np.row_stack((temp2,x[i,:]))
    temp1=np.delete(temp1,0,0)
    temp2=np.delete(temp2,0,0)
    
    X=np.linspace(-4,4,num=100)
    Y=-1/theta[2]*(theta[0]+theta[1]*X)
    
    plt.figure()
    plt.scatter(temp1[:,0],temp1[:,1],color='b')
    plt.scatter(temp2[:,0],temp2[:,1],color='r')
    plt.plot(X,Y)
    plt.show()

def main():
    x,y=loaddata()
    
    theta=optimisepara(x,y)
    
    scatt(x,y,theta)
    print(theta)
    
if __name__=='__main__':
    main()
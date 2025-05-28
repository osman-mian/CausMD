import numpy as np


#This function does NOT discretize data. It converts non numerical values to  distinct integers so the methods could use it.
#Arg1: Data
#Arg2: Variables indexes to treat as discrete
#Arg3: (Disabled for now) adding noise to discretized values 
def non_int_to_disc(X,idx,var=0.1):
    n = X.shape[0]
    for i in idx:
        col= X[:,i].tolist()
        distinct = list(set(col))
        distinct.sort()
        mapping = {distinct[value]:value for value in range(len(distinct))}
        transformed = np.array([mapping[j] for j in col],dtype=int)
        noise = var * np.random.randn(n)
        X[:,i] = transformed #+ noise
                   
    return np.array(X,dtype=int)

def standardize(X):
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std
    
              
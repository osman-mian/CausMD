
import pandas as pd

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

from utils import *
from CDmethods import GESX, PCX, GLOBEX,GRASPX

import pickle
import gc        #garbage collector

def main():
    # default parameters
    dirpath ="./"
    fname = "sachs.tsv"
    
    #load data
    X = pd.read_csv(dirpath+fname, sep='\t',dtype=object)
    headers = X.columns.to_list()
    data = X.to_numpy()
    
    #clean up
    dt = non_int_to_disc(data.copy(),[] ,0)  #take non integer discrete columns and replace those values with integer indexes
    n,d =dt.shape
    

    #define arguments for methods
    ges_bic_args  = {'score_func':"local_score_BIC"}
    ges_bdeu_args = {'score_func':"local_score_BDeu"}
    ges_gen_args  = {'score_func':"local_score_CV_general"}

    pc_args_kci   = {'alpha':0.05 , 'indep_test':'kci', 'kernelZ':'Polynomial'}
    pc_args_csq   = {'alpha':0.05 , 'indep_test':'chisq'}

    globe_disc_args = {'alpha':3,'score_func': 'ACID'}
    globe_cont_args = {'alpha':3,'score_func': 'GLOBE'}
    
    boss_args = {'score_func':'local_score_CV_general'}
    
    
    #initialize the methods
    methods = [GRASPX(**ges_bdeu_args),GESX(**ges_bdeu_args), GESX(**ges_bic_args), GLOBEX(**globe_disc_args),PCX(**pc_args_csq),PCX(**pc_args_kci),GRASPX(**ges_gen_args),GESX(**ges_gen_args)]
    cut_off = np.floor(len(methods) *0.5) #threshold of edge selection currently set to observing an edge in half of the algorithms
    
    print("Data: ",n ," x " ,d)
    print("To Run: ",[m.name_ for m in methods])
    print("Qualification cut-off: ",cut_off)
    

    #statistical book keeping
    totals = np.eye(d)*0
    cache = {}
    
    #run the methods
    for m in methods:
        gc.collect()
        if m.name_ not in cache:
            cache[m.name_] = []
        print("Running: ",m.name_,".....",str(m.kwargs))    
        network = m.run(dt,headers)
        cache[m.name_].append( (m.kwargs,network) )
        for i in range(d):
            for j in range(i+1,d):
                if "GLOBE" in m.name_:
                    if   network[i,j] == 1                     : totals[i,j]+=1
                    elif network[j,i] == 1                     : totals[j,i]+=1
                    elif network[j,i] == 1 and network[i,j]==1 : totals[i,j]+=1; totals[j,i]+=1 #unlikely this happens but still, putting this here
                else: #this is how MEC graphs are returned from causal-learn, check out the documentation to make sense of this :)
                    if  ( network[i,j] ==-1 or  network[i,j] == 0) and network[j,i]==1 : totals[i,j]+=1
                    elif  network[i,j] == 1 and (network[j,i] ==-1 or network[j,i]==0) : totals[j,i]+=1
                    elif  network[i,j] ==-1 and network[j,i] ==-1                      : totals[i,j]+=1; totals[j,i]+=1
                    elif  network[i,j] == 1 and network[j,i] == 1                      : totals[i,j]+=1; totals[j,i]+=1


    #print results
    pre_select = totals*0;
    
    print("Stats summary...")
    for i in range(d):
        for j in range(d):
            if totals[i,j]!=0: print(headers[i]," -> ", headers[j]," = ",totals[i,j])
            if (totals[i,j]+totals[j,i])>cut_off: pre_select[i,j]=1; pre_select[j,i]=1;
            

    print("Final Cut...")
    for i in range(d):
        for j in range(i+1,d):
            if pre_select[i,j]!=0 or pre_select[j,i]!=0: print(headers[i]," ---- ", headers[j], "(",totals[i,j]+totals[j,i],")")
            
            

    print("Running GLOBE...")
    globeX = GLOBEX(**globe_disc_args)
    network = globeX.run(dt,headers,pre_select)
    
    for j in range(d):
        p_set=[]
        p_idx=[]
        for i in range(d):
            if network[i,j]==1: p_set.append(headers[i]); p_idx.append(i)
        if len(p_set)>0:
            print(p_set," ----> ",headers[j],": ",globeX.get_confidence(dt,p_idx,j))
        else:
            print("[] ----> ",headers[j])
            
            
    print("Done...")
    
    #store stats
    store=True
    if store:
        with open('results_'+fname+".pkl", 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
main()
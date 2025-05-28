import numpy as np

''' This appears to be removed from an updated version of causallearn
from causallearn.search.PermutationBased.BOSS import boss
class BOSSX:
    def __init__(self,**kw_args):
        self.kwargs = kw_args
        self.name_ = "BOSS_" + self.kwargs['score_func']
        
    def run(self,data,headers):
        n,d   = data.shape
        graph = np.eye(d)*0 #+ np.inf
        try:
            G = boss(data,**self.kwargs)
            graph  = G.graph
        except Exception as e:
            print("Error running "+self.name_+": ")
            print(e)

        return graph
#'''
from causallearn.search.PermutationBased.GRaSP import grasp
class GRASPX:
    def __init__(self,**kw_args):
        self.kwargs = kw_args
        self.name_ = "GRASP_" + self.kwargs['score_func']
        
    def run(self,data,headers):
        n,d   = data.shape
        graph = np.eye(d)*0 #+ np.inf
        try:
            G      = grasp(data,**self.kwargs)
            graph  = G.graph
        except Exception as e:
            print("Error running "+self.name_+": ")
            print(e)

        return graph
    
    
    
from causallearn.search.ScoreBased.GES import ges
class GESX:
    def __init__(self,**kw_args):
        self.kwargs = kw_args
        self.name_ = "GES_" + self.kwargs['score_func']
        
    def run(self,data,headers):
        n,d   = data.shape
        graph = np.eye(d)*0 #+ np.inf
        try:
            Record = ges(data,**self.kwargs)
            graph  = Record['G'].graph
        except Exception as e:
            print("Error running "+self.name_+": ")
            print(e)

        return graph

    
from causallearn.search.ConstraintBased.PC import pc
class PCX:
    def __init__(self,**kw_args):
        self.kwargs = kw_args
        self.name_ = "PC_"+ self.kwargs['indep_test']
        
    def run(self,data,headers):
        n,d   = data.shape
        graph = np.eye(d)*0 #+ np.inf
        try:
            Record = pc(data,show_progress=False,**self.kwargs)
            graph  = Record.G.graph
        except Exception as e:
            print("Error running "+self.name_+": ")
            print(e)

        return graph
    

from globe.gds import GreedyDagSearch, compute_phi
class GLOBEX:
    def __init__(self,**kw_args):
        self.kwargs = kw_args
        self.name_  = "GLOBE_"+ self.kwargs.get('score_func','default')
        self.globe  = GreedyDagSearch(**self.kwargs)
        self.scm    = {}
        
    def run(self,data,headers,pre_select=None):
        n,d   = data.shape
        graph = np.eye(d)*0 #+ np.inf
        try:
            graph  = self.globe.learn(data,headers,pre_select)
        except Exception as e:
            print("Error running "+self.name_+": ")
            print(e)

        return graph
    
    
    def get_confidence(self,data,pa,ch):
        conf = None
        try:
            pa.sort()
            conf = self.globe.compute_confidence(data,pa,ch)
        except Exception as e:
            print("Error getting confidence from "+self.name_+": ")
            print(e)

        return conf
    

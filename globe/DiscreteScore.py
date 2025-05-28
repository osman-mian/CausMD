from typing import Any, Callable, Dict, List
import numpy as np
import gc
from caddie import anm, measures
import math

class DiscreteScore:
    
    def __init__(self):
        self.score_cache = {};
        self.measure = measures.StochasticComplexity
        self.use_regularization = False 

    def compute(self,data, i: int, PAi: List[int],flag=False, cache=True) -> float:
        if cache==False:
            return self.local_score(data, i, PAi,flag)
        
        if i not in self.score_cache: self.score_cache[i] = {}

        hash_key = tuple(sorted(PAi))

        if not self.score_cache[i].__contains__(hash_key):
            self.score_cache[i][hash_key] = self.local_score(data, i, PAi,flag)

        return self.score_cache[i][hash_key]
    
    
    def compute_gain(self,data, i: int, PAi: List[int], j: int) -> float:
        s1 = self.compute(data,i,PAi)
        PAi.append(j)
        s2 = self.compute(data,i,PAi)
        delta = s1 - s2
        delta = delta.item() if delta > 0 else 0
        return -1*delta
    
    
    def local_score(self,data,target_id:int,parent_ids: List[int],show=False) -> float:
        """Calculate the cost of a node given its parents.

        Args:
            data: The data to be processed. The inner lists are the different variables.
            measure: The measure to be used for the cost calculation.
            parent_ids: The parents of the target node.
            target_id: The target node.
            use_regularization: If True, regularization is used when calculating the cost of the node.

        Returns:
            The cost of the node given its parents.
        """
        if len(parent_ids) == 0:
            cost = self.measure.measure(data[:,target_id].tolist()) + self.normalization(0, len(data))
            if show: print(cost)
            return cost
        else:
            source_data = [data[:,parent_id].tolist() for parent_id in parent_ids]
            combined_sources = self.combine_variables(*source_data)
            target_data = data[:,target_id].tolist()
            cost = anm.discrete_regression(combined_sources, target_data, self.measure, 20, 0.05) + self.normalization(len(parent_ids), len(data))
            cost -= self.measure.measure(combined_sources)
            if show: print(cost)
            return cost
        
    def normalization(self,num_parents: int, num_nodes: int) -> float:
        """Calculate the cost of encoding the structure of one node given it parents.

        Args:
            num_parents: The number of parents of the target node.
            num_nodes: The number of nodes in the graph.

        Returns:
            The encoding cost.
        """
        if not self.use_regularization: return 0
        return math.log(num_nodes, 2) + num_parents * math.log(num_nodes, 2)


    def combine_variables(self,*args: List[int]) -> List[int]:
        """Combine any number of given variables into a single list. Necessary for calculating the cost of multiple parent variables.

        Each value must have a value between 0 and 99.

        Args:
            args: The variables to be combined.

        Returns:
            The combined variables.
        """

        # make sure that there are no negative values or to big values in the lists
        for current_list in args:
            if min(current_list) < 0:
                raise ValueError(
                    "negative values are not allowed, because the cross product implementation can't handle them"
                )
            if max(current_list) > 99:
                raise ValueError(
                    "values greater than 99 are not allowed, because the cross product implementation can't handle them"
                )

        if len(args) == 1:
            return args[0]

        res = []
        for row in zip(*args):
            current_value = 0
            for i, value in enumerate(row):
                current_value += value * (100**i)
            res.append(current_value)

        return res


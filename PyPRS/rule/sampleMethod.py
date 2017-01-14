# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:39:55 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

import numpy as np
from .. import utils
import random

def uniform(leaf,n):
    """draw n uniform distributed samples from this leaf
    Args:
        leaf: A class Tree() representing the leaf node region
        n: An integer representing sampling size
    Returns:
        leaf: This leaf
        samples: An n * dimX array representing the uniformlly samples     
    """
    samples = discretize(leaf, np.random.uniform(leaf.lb,leaf.ub,(n,len(leaf.lb))))
    return {'leaf':leaf,'samples':samples}
    

def discretize(leaf, samples):
    """ discretize the samples based on the problem's discretized level
    Args:
        leaf: A class Tree() representing the leaf node region
        samples: An n * dimX array representing the samples     
    Returns:
        discretizedSamples: An n * dimX array representing the discretized samples     
    """
    LB, UB, discreteLevel = leaf.problem.lb, leaf.problem.ub, leaf.problem.discreteLevel
    discretizedSamples = utils.discretize(samples, LB, UB, discreteLevel)
    return discretizedSamples
    
def normal(leaf, n):
    """draw n uniform distributed samples from this leaf
    Args:
        leaf: A class Tree() representing the leaf node region
        n: An integer representing sampling size
    Returns:
        leaf: This leaf
        samples: An n * dimX array representing the uniformlly samples     
    """
    visitedPoints = leaf.pool
    paretoSet = utils.identifyParetoSet(visitedPoints)        
    samples = []
    while len(samples) < n:
        if paretoSet:
            pivot = paretoSet[random.choice(paretoSet.keys())]
            loc = pivot.x
        else:
            loc = (leaf.lb + leaf.ub) / 2
        scale = (leaf.root.ub - leaf.root.lb) / leaf.problem.discreteLevel
        point = discretize(leaf, np.random.normal(loc,scale))
        if leaf.withinNode(point):
            samples.append(point)
    samples = np.array(samples)
    return {'leaf':leaf, 'samples':samples} 
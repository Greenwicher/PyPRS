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
from .. import _cutils
import random
from numba import jit
import copy

def uniform(leaf, n, args):
    """draw n uniform distributed samples from this leaf
    Args:
        leaf: A class Tree() representing the leaf node region
        n: An integer representing sampling size
        args: A dictionary of arguments for the function
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
    
def normal(leaf, n, args):
    """draw n uniform distributed samples from this leaf
    Args:
        leaf: A class Tree() representing the leaf node region
        n: An integer representing sampling size
        args: A dictionary of arguments for the function        
    Returns:
        leaf: This leaf
        samples: An n * dimX array representing the uniformlly samples     
    """
    epsilon = 0.5
    visitedPoints = leaf.pool
    paretoSet = utils.identifyParetoSet(visitedPoints)        
    samples = []
    t, T = 0, 2 * n
    while (len(samples) < n) and (t < T):
        if paretoSet:
            if np.random.uniform(0,1) <= epsilon:
                pivot = paretoSet[random.choice(list(paretoSet.keys()))]
                loc = pivot.x
            else:
                loc = np.random.uniform(leaf.lb, leaf.ub)
        else:
            loc = (leaf.lb + leaf.ub) / 2
        scale = (leaf.root.ub - leaf.root.lb) / leaf.problem.discreteLevel
        point = discretize(leaf, [np.random.normal(loc,scale)])[0]
        if leaf.withinNode(point):
            samples.append(point)
        t += 1
    samples = feasible(leaf, np.array(samples))
    return {'leaf':leaf, 'samples':samples} 
    
def elite(leaf, n, args):
    """draw n samples based on elite search algorithm
    Args:
        leaf: A class Tree() representing the leaf node region
        n: An integer representing sampling size
        args: A dictionary of arguments for the function        
    Returns:
        leaf: This leaf
        samples: An n * dimX array representing the uniformlly samples     
    """    
    import PyGMO
    problem = copy.deepcopy(args['elite']['problemGMO'])
    problem.lb = tuple(leaf.lb)
    problem.ub = tuple(leaf.ub)
    alg = copy.deepcopy(args['elite']['algGMO'])
    numPop = args['elite']['numPop']
    
    visitedPoints = leaf.root.visitedPoints()
    pool = leaf.pool
    _pool = utils.dictToSortedNumpyArray(pool, len(leaf.problem.objectives))  
    # check whether the leaf.pool is empty
    if not (len(_pool)):
        return uniform(leaf, n, args)
    # construct the elite initial population by sorting the dominationCount
    dominationCount = _cutils.calDominationCount(_pool, _pool, len(_pool))[1] 
    _candidate = sorted(zip([pool[k].x for k in pool], dominationCount), key=lambda x: x[1])
    candidate = [foo[0] for foo in _candidate]
    for _ in range(numPop - len(candidate)):
        candidate.append(uniform(leaf,1,args)['samples'][0])
    # construct initial population
    pop = PyGMO.population(problem, numPop)
    for x in candidate:
        if leaf.withinNode(x):
            pop.push_back(list(x))
            pop.erase(0)
    samples = []
    samplesKey = []
    # generate new elite points
    t, T = 0, 2 * n
    while(len(samples) < n and t < T and n < np.product((leaf.ub-leaf.lb)/((leaf.problem.ub-leaf.problem.lb)/(leaf.problem.discreteLevel + 1e-100)))):            
        for individual in pop:
            x = discretize(leaf, [individual.cur_x])[0]
            key = utils.generateKey(x)
            if not((key in visitedPoints) or (key in samplesKey)):
                samples.append(x)
                samplesKey.append(key)
        pop = alg.evolve(pop) 
        t += 1
    if not(samples):
        samples = uniform(leaf,n,args)['samples']
    elif len(samples) < n:
        rn = n - len(samples)
        for _ in range(rn):
            samples.append(uniform(leaf,1,args)['samples'][0])
    else:
        samples = np.array(samples[:n])
    samples = feasible(leaf, samples)
    return {'leaf':leaf, 'samples':samples}  
    
def feasible(leaf, samples):
    """ select the feasible samples 
    Args:
        leaf: A class Tree() representing the leaf node region
        samples: An n * dimX array representing the samples     
    Returns:
        feasibleSamples: An m * dimX array representing the feasible samples
    """
    feasibleSamples = []
    for s in samples:
        if utils.withinRegion(s, leaf.lb, leaf.ub): feasibleSamples.append(s)
    feasibleSamples = np.array(feasibleSamples)
    return feasibleSamples    
# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:37:11 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

import numpy as np
from .. import utils
from .. import _cutils

def bisection(leaf, args):
    """ Partition a given bounded hyperbox region into two subregions based
    on bisection methods
    Args:
        leaf: A class Tree() representing leaf node region
        args: A dictionary representing the arguments necessary for partition
    Returns:
        parent: A class Tree() representing the parent node
        thr: A double representing partition threshold
        subRegions: A list of two elements representing new subregions            
    """
    #retrieve the lower/upper bound of given Most Promising Region
    lb = leaf.lb
    ub = leaf.ub
    #find the dimension number of decision variabless
    dimX = len(lb)
    #determine the dimension that should be partitioned
    dimID = (leaf.level + 1) % dimX 
    #determine the partition threshold
    thr = (lb[dimID]+ub[dimID])/2.0
    #create new lower/upper middle bound [lb,umb], [lmd,ub]
    lmb,umb = [np.array([]) for i in range(2)]
    for i in range(dimX):
        lmb = np.append(lmb,[lb[i],thr][i==dimID])
        umb = np.append(umb,[ub[i],thr][i==dimID])
    subRegions = [[lb,umb],[lmb,ub]]
    return {'parent':leaf,'thr':thr,'subRegions':subRegions}

def xsection(leaf, args):
    """ Partition a given bounded hyperbox region into x subregions based 
    on xsection methods
    Args:
        leaf: A class Tree() representing leaf node region
        args: A dictionary representing the arguments necessary for partition        
    Returns:
        parent: A class Tree() representing the parent node
        thr: A double representing partition threshold
        subRegions: A list of two elements representing new subregions   
    """
    x = args['xsectionNum'] # test 10-section method
    # retrieve the lower/upper bound of given Most Promising Region
    lb = leaf.lb
    ub = leaf.ub
    # find the dimension number of decision variabless
    dimX = len(lb)
    # determine the dimension that should be partitioned
    #dimID = (leaf.level + 1) % dimX  # debug
    dimID = nextDim(leaf, args)
    # determine the partition unit distance 
    unit = (ub[dimID] - lb[dimID]) / x
    # construct the subRegions
    subRegions = []
    for i in range(x):
        _lb, _ub = [np.array([]) for _ in range(2)]
        # change the lower and upper bound value at dimID for subRegion x
        for j in range(dimX):
            _lb = np.append(_lb, lb[j] + (unit * i) * (j == dimID))
            _ub = np.append(_ub, ub[j] - (unit * (x - i - 1)) * (j == dimID))
        subRegions.append([_lb,_ub])      
    return {'parent':leaf,'thr':[unit * i for i in range(x)],'subRegions':subRegions}    
            
def nextDim(leaf, args):
    """ Determine the next dimension to partition
    Args:
        leaf: A class Tree() representing leaf node region
        args: A dictionary representing the arguments necessary for partition
    Returns:
        dim: An integer indicating the next dimension to partition
    """
    x = args['xsectionNum'] # number of subregions to partition for the leaf
    lb = leaf.lb # the lower bound of the leaf region
    ub = leaf.ub # the upper bound of the leaf region
    dimDiff = [] # store the diff value (e.g. max-min of dominantion count) for partition direction
    dimX = len(lb) # the number of dimension
    visitedPoints = leaf.visitedPoints() # all the visited points in the tree
    pool = leaf.pool # the visited points in this leaf
    #determine the deminsion of point's objective
    dim = len(leaf.problem.objectives)    
    #recorganize all the visited points together into one sorted array
    _visitedPoints = utils.dictToSortedNumpyArray(visitedPoints,dim)        
    # calculate the domination count for each point in this pool
    dominantionCount = {}    
    for key in pool:
        _p = np.array([pool[key].mean])
        dominantionCount[key] = _cutils.calDominationCount(_p, _visitedPoints, len(_p))[1][0]
    # enumerate all the possible next dimension to partition
    for dimID in range(dimX):
        # determine the partition unit distance 
        unit = (ub[dimID] - lb[dimID]) / x
        # initialize the promisingIndex for each subregion based on xsection
        promisingIndex = []        
        for i in range(x):
            _lb, _ub = [np.array([]) for _ in range(2)]
            # change the lower and upper bound value at dimID for subRegion x
            for j in range(dimX):
                _lb = np.append(_lb, lb[j] + (unit * i) * (j == dimID))
                _ub = np.append(_ub, ub[j] - (unit * (x - i - 1)) * (j == dimID))
            # calculate the promisingIndex for each subregions
            poolDominantionCount = [np.nan] # in case no points in this subregion
            for key in pool:
                p = pool[key]                
                if all(_lb <= p.x) and all(p.x < ub):
                    poolDominantionCount.append(dominantionCount[key])
            # calculate the promising index in this subregion            
            promisingIndex.append(np.nanmin(poolDominantionCount))
        # calculate the dimDiff for the dimension dimID            
        diff = np.nanmax(promisingIndex) - np.nanmin(promisingIndex)
        dimDiff.append(diff)        
    # select the dimension with largest dimDiff value as next dimension to partition
    maxDiff = np.nanmax(dimDiff)
    if not(np.isnan(maxDiff)):
        candidate = [i for i in range(dimX) if dimDiff[i] == maxDiff]        
        dim = candidate[np.random.randint(0,len(candidate))]
    else:
        dim = np.random.randint(0,dimX)
    print('Select Dim %d with maxDiff %.2f, range %.2f at level %d' % (dim, maxDiff, ub[dim]-lb[dim],leaf.level))
    return dim
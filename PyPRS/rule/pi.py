# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:40:31 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""
from .. import utils
import numpy as np
from itertools import repeat
from .. import _cutils

###DOMINATION COUNT###

def calDominationCount(p,visitedPoints):
    """calculate domination count for point p
    Args:
        p: A class Point()
        visitedPoints: A dictionary of all visitedn points
    Returns:
        dominationCount: An integer indicating the domination count of p
    """
    isDominated = utils.MultiThread(utils.dominating, zip([visitedPoints[k].mean for k in visitedPoints],repeat(p.mean)))
    dominationCount = sum(isDominated)
    print('Please _cutils.calDominantionCount(). This method is too slow.')
    return dominationCount

def averageDominationCount(leaf):
    """calculate average domination count for this leaf
    Args:
        leaf: A class Tree() representing leaf node region
    Returns:
        averageDominationCount: A double representing average of domination
                                count for this leaf node  
    """
    averageDominationCount = np.nanmean(leaf.calDominationCount())
    return averageDominationCount
    
def medianDominationCount(leaf):
    """calculate median domination count for this leaf
    Args:
        leaf: A class Tree() representing leaf node region
    Returns:
        medianDominationCount: A double representing median of domination
                                count for this leaf node  
    """
    medianDominationCount = np.nanmean(leaf.calDominationCount())
    return medianDominationCount    

def minimumDominationCount(leaf):
    """calculate minimum domination count for this leaf
    Args:
        leaf: A class Tree() representing leaf node region
    Returns:
        minimumDominationCount: A double representing minimum of domination
          count for this leaf node  
    """
    minimumDominationCount = np.nanmin(leaf.calDominationCount())
    return minimumDominationCount  
    
def maximumDominationCount(leaf):
    """calculate maximum domination count for this leaf
    Args:
        leaf: A class Tree() representing leaf node region
    Returns:
        maximumDominationCount: A double representing maximum of domination
          count for this leaf node  
    """
    maximumDominationCount = np.nanmax(leaf.calDominationCount())
    return maximumDominationCount        
    
def balancedEE(leaf):
    """ calculate the indicator for this leaf which can make a balance between
        exploration and exploitation
    Args:
        leaf: A class Tree() representing leaf node region
    Returns:
        balancedV: A double representing the balanced value
    """
    # determine the exploitation and exploration value for the leaf
    leafExploitation = medianDominationCount(leaf)
    leafExploration = leaf.n / np.product(leaf.ub-leaf.lb)
    # determine the exploitation and exploration rank for each leafs (not efficient, need to outside)
    leafNodes = leaf.leafNodes()
    leafExploitationRank = 0
    leafExplorationRank = 0
    for _leaf in leafNodes:
        _exploitation = medianDominationCount(_leaf)
        _exploration = _leaf.n / np.product(_leaf.ub-_leaf.lb)
        leafExploitationRank += _exploitation < leafExploitation
        leafExplorationRank += _exploration < leafExploration
    # determine the number of total observed samples
    N = np.nanmax([sum([l.n for l in leaf.root.leafNodes()]),1])    
    # calculate the balanced value with adaptive weight on the exploration part
    balancedV = leafExploitationRank + leafExplorationRank
    print(leafExploitationRank, leafExplorationRank)
    return balancedV
    
def isPareto(leaf):
    """ check whether the leaf contains the Pareto solution or not
    Args:
        leaf: A class Tree() representing leaf node region
    Returns:
        flag: A boolean indicating whether the leaf contains the Pareto solution
    """    
    # determine the deminsion of point's objective
    dim = len(leaf.problem.objectives)       
    # recorganize all the visited points together into one sorted array
    _visitedPoints = utils.dictToSortedNumpyArray(leaf.visitedPoints(),dim)      
    # check whether the leaf contains the Pareto solution
    flag = 0
    pool = leaf.pool
    for key in pool:
        _p = np.array([pool[key].mean])
        dominantionCount = _cutils.calDominationCount(_p, _visitedPoints, len(_p))[1][0] 
        if dominantionCount == 0: 
            flag = 1
            break            
    return flag

# -*- coding: utf-8 -*-
"""
    Created on Sat Apr  2 11:13:33 2016
    @author: liuweizhi (greenwicher.com， weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

#from . import hv
import PyGMO
from . import utils
import numpy as np


def calHyperVolume(front, referencePoint):
    """ calculate the hypervolume value
    Args:
        front: A list of "Pareto" front objectives value
        referencePoint: A numpy array representing the reference point
    Returns:
        hyperVolume: A doulbe representing the hypervolume value
    """
    try:
        HV = PyGMO.hypervolume(front)
        hyperVolume = HV.compute(r=list(referencePoint))
    except:
        hyperVolume = 0.0
    return hyperVolume
    
def calTrueParetoProportition(paretoSet, trueParetoSet):
    """ calculate the proportion of true Pareto solution given the estimated 
        Pareto set and true Pareto set
    Args:
        paretoSet: A list of decision vector of estimated Pareto set
        trueParetoSet: A list of decision vector of true Pareto set
    Returns:
        trueParetoProportition: A double indicating the proportion of true 
        Pareto solution given the estimated Pareto set and true Pareto set
    """
    paretoSetKey = set()
    trueParetoSetKey = set()
    for x in paretoSet:
        paretoSetKey.add(utils.generateKey(x))
    for x in trueParetoSet:
        trueParetoSetKey.add(utils.generateKey(x))
    trueParetoProportition = len(paretoSetKey & trueParetoSetKey) / len(trueParetoSetKey)
    return trueParetoProportition
    
def calHausdorffDistance(paretoSet, trueParetoSet):
    """ calculate Hausdorff distance (see https://en.wikipedia.org/wiki/Hausdorff_distance)
        between the estimated Pareto set and true Pareto set
    Args:
        paretoSet: A list of decision vector of estimated Pareto set
        trueParetoSet: A list of decision vector of true Pareto set
    Returns:
        HausdorffDistance: A double indicating Hausdorff distance
    """    
    m, n = len(paretoSet), len(trueParetoSet)
    d = [[np.nan] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            d[i][j] = np.linalg.norm(paretoSet[i] - trueParetoSet[j])
    d = np.array(d)
    rowDistance, colDistance = [], []
    for i in range(m):
        rowDistance.append(np.min(d[i]))
    for j in range(n):
        colDistance.append(np.min(d[:,j]))
    HausdorffDistance = np.max([np.max(rowDistance), np.max(colDistance)])
    return HausdorffDistance
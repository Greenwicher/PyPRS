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


def calHyperVolume(paretoSet,referencePoint):
    """ calculate the hypervolume value
    Args:
        paretoSet: A dictionary of class Point() representing the Pareto set
        referencePoint: A numpy array representing the reference point
    Returns:
        hyperVolume: A doulbe representing the hypervolume value
    """
    front = []
    for k in paretoSet:
        front.append(paretoSet[k].trueMean)
    # use hv.py
#    HV = hv.HyperVolume(referencePoint)
#    hyperVolume = HV.compute(front)
    # use PyGMO
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
        paretoSet: A dictionary of estimated Pareto set with class Point()
        trueParetoSet: A dictionary of true Pareto set with class Point()
    Returns:
        trueParetoProportition: A double indicating the proportion of true 
        Pareto solution given the estimated Pareto set and true Pareto set
    """
    paretoSetKey = set()
    trueParetoSetKey = set()
    for k in paretoSet:
        paretoSetKey.add(utils.generateKey(paretoSet[k].x))
    for k in trueParetoSet:
        trueParetoSetKey.add(utils.generateKey(trueParetoSet[k].x))
    trueParetoProportition = len(paretoSetKey & trueParetoSetKey) / len(trueParetoSetKey)
    return trueParetoProportition
    
def calHausdorffDistance(paretoSet, trueParetoSet):
    """ calculate Hausdorff distance (see https://en.wikipedia.org/wiki/Hausdorff_distance)
        between the estimated Pareto set and true Pareto set
    Args:
        paretoSet: A dictionary of estimated Pareto set with class Point()
        trueParetoSet: A dictionary of true Pareto set with class Point()
    Returns:
        HausdorffDistance: A double indicating Hausdorff distance
    """    
    paretoSetList = [paretoSet[k].x for k in paretoSet]
    trueParetoSetList = [trueParetoSet[k].x for k in trueParetoSet]
    m, n = len(paretoSetList), len(trueParetoSetList)
    d = [[np.nan] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            d[i][j] = np.linalg.norm(paretoSetList[i] - trueParetoSetList[j])
    d = np.array(d)
    rowDistance, colDistance = [], []
    for i in range(m):
        rowDistance.append(np.min(d[i]))
    for j in range(n):
        colDistance.append(np.min(d[:,j]))
    HausdorffDistance = np.max([np.max(rowDistance), np.max(colDistance)])
    return HausdorffDistance
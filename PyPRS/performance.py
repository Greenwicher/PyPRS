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
    
def calTrueParetoProportion(paretoSet, trueParetoSet):
    """ calculate the proportion of true Pareto solution given the estimated 
        Pareto set and true Pareto set
    Args:
        paretoSet: A list of decision vector of estimated Pareto set
        trueParetoSet: A list of decision vector of true Pareto set
    Returns:
        trueParetoProportition: A double indicating the proportion of true 
        Pareto solution given the estimated Pareto set and true Pareto set
    """
    if trueParetoSet:
        paretoSetKey = set()
        trueParetoSetKey = set()
        for x in paretoSet:
            paretoSetKey.add(tuple(x))
        for x in trueParetoSet:
            trueParetoSetKey.add(tuple(x))   
        trueParetoProportion = len(paretoSetKey & trueParetoSetKey) / len(trueParetoSetKey)
    else:
        trueParetoProportion = np.nan
    return trueParetoProportion
    
def calHausdorffDistance(paretoSet, trueParetoSet):
    """ calculate Hausdorff distance (see https://en.wikipedia.org/wiki/Hausdorff_distance)
        between the estimated Pareto set and true Pareto set
    Args:
        paretoSet: A list of decision vector of estimated Pareto set
        trueParetoSet: A list of decision vector of true Pareto set
    Returns:
        HausdorffDistance: A double indicating Hausdorff distance
    """    
    if trueParetoSet:
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
    else:
        HausdorffDistance = np.nan
    return HausdorffDistance
    
def fill(x, y, minLength):
    """ fill the empty (x, y) point given the discrete set of (x, y)
    Args:
        x: A list of discrete point in the xth coordinate
        y: A list of discrete point in the yth coordinate associated with x
        minLength: An integer representing the required minimum length of path        
    Returns:
        y: A list of "continuous" point in the yth coordinate associated with x
    """
    x = [1] + x
    y = [np.nan] + y
    _y = []
    for i in range(len(y)-1):
         _y += [y[i]] * (x[i+1] - x[i])
    _y += [y[-1]] * max(minLength - x[-1] + 1, 1)
    return _y
    
def calEnsembleMean(results, minLength = 0):
    """ calculate the ensemble mean of performance based on results of 
        different replications
    Args:
        results: A list of algorithm's results from different replications or 
            A dictionary of algorithm's results for single replication
        minLength: An integer representing the required minimum length of path
    Returns:
        ensembleMean: A dictionary of ensemble mean of algorithm's performance
    """
    ensembleMean = {}
    y = ['HV', 'GO', 'HD']
    for foo in y:
        ensembleMean[foo] = calSubEnsembleMean(results, 'sampleSize', foo, minLength)
    return ensembleMean
    
    
def calSubEnsembleMean(results, x, y, minLength = 0):
    """ calculate the ensemble of particular performance based on results of 
        different replications
    Args:
        results: A list of algorithm's results from different replications or 
            A dictionary of algorithm's results for single replication
        x: A string indicating the name of xth coordinate 
        y: A string indicating the name of yth coordinate
        minLength: An integer representing the required minimum length of path        
    Returns:
        ensembleMean: A list of ensemble mean of algorithm's performance y  
    """
    # check whether single or multiple replications    
    if isinstance(results, list):
        samplePath = []
        minmax = np.inf
        # fill the path for each replication
        for r in results:
            try:
                path = fill(r[x], r[y])
                minmax = min(len(path), minmax)                
                # only keep the sample path if its length >= minLength
                if len(path) >= minLength:
                    samplePath.append(path)
                else:
                    continue
            except Exception as e:
                ensembleMean = None
                print('Error: calSubEnsembleMean,', e)
                #return ensembleMean
        # calculate the ensemble mean
        _samplePath = []
        for path in samplePath:
            _samplePath.append(path[:minmax])
        # check if there are any valid sample path (length >= minLength)
        if samplePath:
            ensembleMean = np.mean(np.array(_samplePath), axis=0)
        else:
            ensembleMean = np.ones(minLength) * np.nan
    elif isinstance(results, dict):
        try:
            ensembleMean = fill(results[x], results[y])
        except:
            ensembleMean = None
            print('Error: calSubEnsembleMean')
            return ensembleMean
    else:
        ensembleMean = None
        print('Type Error: Invalid Type, only list/dictionary allowed')
    return ensembleMean
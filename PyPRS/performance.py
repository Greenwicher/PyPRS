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
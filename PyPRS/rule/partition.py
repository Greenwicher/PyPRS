# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:37:11 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

import numpy as np

def bisection(leaf):
    """ Partition a given bounded hyperbox region into two subregions based
    on bisection methods
    Args:
        leaf: A class Tree() representing leaf node region
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
    dimID = leaf.level % dimX 
    #determine the partition threshold
    thr = (lb[dimID]+ub[dimID])/2.0
    #create new lower/upper middle bound [lb,umb], [lmd,ub]
    lmb,umb = [np.array([]) for i in range(2)]
    for i in range(dimX):
        lmb = np.append(lmb,[lb[i],thr][i==dimID])
        umb = np.append(umb,[ub[i],thr][i==dimID])
    subRegions = [[lb,umb],[lmb,ub]]
    return {'parent':leaf,'thr':thr,'subRegions':subRegions}

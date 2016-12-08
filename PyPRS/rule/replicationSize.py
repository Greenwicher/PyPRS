# -*- coding: utf-8 -*-
"""
    Created on Mon Apr 11 12:13:19 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

import numpy as np

def equal(node,points,MPR,args):
    """determine replication size for each point to be sampled based on equal allocation
    """
    repSize = np.array([args['unitReplicationSize']]*len(points))
    return repSize
    
def mprGreedy(node,points,MPR,args):
    """determine replication size for each point to be sampled based on equal allocation
    """
#    repSize = np.array([args['unitReplicationSize']]*len(points)) * [1,args['replicationTimes']][node.parent in MPR]
    repSize = np.array([1]*len(points)) * [1,args['unitReplicationSize']][node.parent in MPR]    
    return repSize    
    
    
def mprGreedyN(node,points,MPR,args):
    """determine replication size for each point to be sampled based on equal allocation
    """
    if (node.parent in MPR) and (node.parent!=node.root):     
        pool = node.parent.pool
        maxStd = max([max(pool[k].std)for k in pool])
        times = int((maxStd/args['minimumStd'])**2)
    else:
        times = 1        
    repSize = np.array([args['unitReplicationSize']]*len(points)) * times
    return repSize
# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:41:18 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

import numpy as np

def ucb(leaf,args):
    """ calculate the sampling index of the leaf node based on ucb-like index
    Args:
        leaf: A class Tree() representing the leaf node region
        args: A dictionary of arguments for the function    
    Returns:
        value: A double representing the sampling index
    """
#    Q = sum([np.sqrt(2*np.log(N)/max(l.n,1)) for l in leaf.root.leafNodes()])
    #determine the maximum promixing index
    M = np.nanmax([l.promisingIndex for l in leaf.root.leafNodes()])
    if np.isnan(M):
        M = leaf.root.promisingIndex + 1
    #check whether the leaf region is explored or not
    if np.isnan(leaf.promisingIndex):
        value = (M-leaf.parent.promisingIndex)
    else:
        #calculate the ucb-like index
        value = (M-leaf.promisingIndex)+ args['ucb_c']*np.sqrt(2*np.log(max(leaf.parent.n,0))/np.nanmax([leaf.n,1]))
    #print(value)
    return value
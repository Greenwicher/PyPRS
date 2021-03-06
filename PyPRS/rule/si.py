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
    if not(leaf.parent):
        return 1
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
        if np.isnan(leaf.parent.n):
            leaf.parent.n = sum([l.n for l in leaf.parent.children])
        value = max((M-leaf.promisingIndex), 1e-100) / max(M, 1e-100) + args['ucb_c']*np.sqrt(2*np.log(max(leaf.parent.n,1))/np.nanmax([leaf.n,1]))
    return value
    
    
def equal(leaf, args):
    """ set the sampling index of each leaf node equal
    Args:
        leaf: A class Tree() representing the leaf node region
        args: A dictionary of arguments for the function    
    Returns:
        value: A double representing the sampling index
    """    
    return 1
    

    
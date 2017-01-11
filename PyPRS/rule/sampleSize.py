# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:39:19 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""
import numpy as np

def samplingIndex(leaf,args):
    """ determine sampling size for the leaf node region based on sampling
        index
    Args:
        leaf: A class Tree() representing the leaf node region
        args: A dictionary of arguments for the function
    Returns:
        alpha: sampling size for the leaf node 
    """
    leafNodes = leaf.root.leafNodes()
    deltaSampleSize = args['deltaSampleSize']
#    deltaSampleSize = args['unitSampleSize'] * len(leafNodes)    
    sumAlpha = 0.0
    for l in leafNodes:
        sumAlpha += l.samplingIndex
    alpha = max(np.round(leaf.samplingIndex / sumAlpha * deltaSampleSize),1.0)
#    print(alpha)
    return alpha
    
def equalSize(leaf,args):
    """ determine sampling size for the leaf node region based on equal 
        allocation
    Args:
        leaf: A class Tree() representing the leaf node region
        args: A dictionary of arguments for the function   
    Returns:
        alpha: sampling size for the leaf node
    """
    unitSampleSize = args['unitSampleSize']
    return unitSampleSize
    
def metropolisIndex(leaf,args):
    """ determine sampling size for the leaf node region based on sampling
        index
    Args:
        leaf: A class Tree() representing the leaf node region
        args: A dictionary of arguments for the function
    Returns:
        alpha: sampling size for the leaf node 
    """
    unitSampleSize = args['unitSampleSize']
    leafNodes = leaf.root.leafNodes()
    sumAlpha = 0.0
    for l in leafNodes:
        sumAlpha += l.samplingIndex
    acceptProb = leaf.samplingIndex / sumAlpha
    p = np.random.rand()
    if p<acceptProb:
        alpha = unitSampleSize           
    else:
        alpha = 0.0
    print('Prob=%.3f, AcceptProb=%.3f, alpha=%d'%(p,acceptProb,alpha))         
    return alpha
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
    alpha = int(np.nanmax([leaf.samplingIndex / sumAlpha * deltaSampleSize, 1]))
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
    maxSamplingIndex = np.nanmax([l.samplingIndex for l in leafNodes])
    p = np.random.rand()
    acceptProb = leaf.samplingIndex / maxSamplingIndex
    if p < acceptProb:
        alpha = unitSampleSize           
    else:
        alpha = [unitSampleSize, 1][leaf.pool!={}]
    print('Prob=%.3f, AcceptProb=%.3f, alpha=%d \n' % (p,acceptProb,alpha))         
    return alpha
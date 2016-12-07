# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:39:55 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

import numpy as np

def uniform(leaf,n):
    """draw n uniform distributed samples from this leaf
    Args:
        leaf: A class Tree() representing the leaf node region
        n: An integer representing sampling size
    Returns:
        leaf: This leaf
        samples: An n * dimX array representing the uniformlly samples     
    """
    samples = np.random.uniform(leaf.lb,leaf.ub,(n,len(leaf.lb)))
    return {'leaf':leaf,'samples':samples}
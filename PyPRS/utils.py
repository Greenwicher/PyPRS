# -*- coding: utf-8 -*-
"""
    Created on Tue Mar 15 13:39:33 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

from . import _cutils
import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool  
from numba import jit
import shelve

@jit
def dominating(p1,p2):
    """determines whether p1 dominates p2
    Args:
        p1,p2: Two double numpy arrays representing the performance averages 
               vector of this point/solutions
    Returns:
        flag: A boolean value indicating whether p1 dominates p2
    """
    maxDiff,minDiff = -np.inf,np.inf
    for i in range(p1.shape[0]):
        diff = p1[i] - p2[i]
        if diff > maxDiff: maxDiff = diff
        if diff < minDiff: minDiff = diff
    flag = maxDiff<=0 and minDiff<0
#    flag = all(p1<=p2) and any(p1<p2) #not efficient for numba  
    return flag
  
def dictToSortedNumpyArray(d,dim):
    """ convert dictionary of class Point() to sorted numpy array
    Args:
        d: A dictionary of class Point()
        dim: An integer indicating the dimension of point 
    Returns:
        _d: A sorted (first obj, ascending) numpy array of mean of point's objectives
    """
    _d = np.empty((0,dim),float)
    for k in d:
        _d = np.vstack([_d,d[k].mean])
    _d = _d[_d[:,0].argsort()]     
    return _d

def calDominationCount(_pool,_visitedPoints):
    """ calculate the domination count for points in _pool, c++ version in _cutils
    Args:
        _pool: A sorted (on 1st objectives) list of points' objectives mean in _pool
        _visitedPoints: A sorted (on 1st objectives) list of points' objectives mean in _visitedPoints
    Returns:
        dominationCount: A numpy array indicating the domination count of points in _pool        
    """
    #determine domination count for each point
    dominationCount = np.empty((0,1),int)
    for p1 in _pool:
        _dominationCount,i = 0,0
        flag = True
        #only points have smaller or same value of objective 1 can dominate p1
        while flag:
            #nonidentical k2 dominates k1, then increase domination count 
            #of k1 by 1
            p2 = _visitedPoints[i]
            if _cutils.dominating(p2,p1):                
                _dominationCount += 1
            i += 1
            #check if there is a need to search further
            #if not(i<len(_visitedPoints) and p1[0]>=p2[0]):
            if _cutils.stopSearch(i,len(_visitedPoints),p1[0],p2[0]):
                flag = False            
        dominationCount = np.vstack([dominationCount,_dominationCount])    
    return dominationCount
    
def MultiThread(fun,input):
    """ multithread programming
    Args:
        fun: A function handle
        input: A zipped list representing arguments for the function
    Returns:
        results: A list of results for each fun(input[i])
    """
    pool = ThreadPool()
    results = pool.starmap(fun,input)
    pool.close()
    pool.join()
    return list(filter(None.__ne__, results))
   
#def identifyMPR(tree):
#    """ identify the most promising region by leaf node's promising index
#    Args:
#        tree: A class Tree() representing the whole search Tree (i.e. root node)
#    Returns:
#        MPR: A list of class Tree representing the Most Promising Region (MPR)
#    """
#    MPR = []
#    bestPI = np.inf #the smaller, the better
#    for leaf in tree.leafNodes():
#        #check whether it's root node
#        if not(leaf.parent):
#            MPR.append(leaf)
#            break
#        else:
#            #update MPR
#            if leaf.promisingIndex < bestPI:
#                MPR = [leaf]
#                bestPI = leaf.promisingIndex
#            elif leaf.promisingIndex == bestPI:
#                MPR.append(leaf)
#    return MPR

def identifyMPRMetropolis(tree,alpha=0):
    """ identify the most promising region by leaf node's promising index
    Args:
        tree: A class Tree() representing the whole search Tree (i.e. root node)
        alpha: An double representing the percentile of PI threshold
    Returns:
        MPR: A list of class Tree representing the Most Promising Region (MPR)
    """
    MPR = []
    PI = np.array([])
    for leaf in tree.leafNodes():
        PI = np.append(PI,leaf.promisingIndex)
    thrPI = np.percentile(PI[~np.isnan(PI)],alpha)
    for leaf in tree.leafNodes():
        T = 100 / (leaf.level ** 2 + 1e-1)
        metropolis = np.exp(-(leaf.promisingIndex - thrPI)/T)
        #print('Leaf Level = %d, Metropolis = %.4f, PI=%.0f' % (leaf.level, metropolis, leaf.promisingIndex)) #debug
        #check whether it's root node
        if not(leaf.parent):
            MPR.append(leaf)
            break
        #elif leaf.promisingIndex <= thrPI:
        elif np.random.uniform(0,1) < metropolis:
            MPR.append(leaf)
    return MPR

def identifyMPR(tree,alpha=0):
    """ identify the most promising region by leaf node's promising index
    Args:
        tree: A class Tree() representing the whole search Tree (i.e. root node)
        alpha: An double representing the percentile of PI threshold
    Returns:
        MPR: A list of class Tree representing the Most Promising Region (MPR)
    """
    MPR = []
    PI = np.array([])
    for leaf in tree.leafNodes():
        PI = np.append(PI,leaf.promisingIndex)
    thrPI = np.percentile(PI[~np.isnan(PI)],alpha)
    for leaf in tree.leafNodes():
        #check whether it's root node
        if not(leaf.parent):
            MPR.append(leaf)
            break
        elif leaf.promisingIndex <= thrPI:
            MPR.append(leaf)
    return MPR


def updateObjAttr(obj,attr,value):
    """assign value to ojb's attribute attr
    Args:
        obj: a class object
        attr: obj's attribute name
        value: the value to be assigned to obj.attr
    Returns:
        None
    """
    setattr(obj,attr,value)
    return     
    
def identifyParetoSet(visitedPoints):
    """identify the Pareto Set given the visited points
    Args:
        visitedPoints: A dictionary including the visited points (no need to be
                       all points)
    Returns:
        paretoSet: A dictionary including current Pareto systems
    """  
    paretoSet = {}
    for k1 in visitedPoints:
        flag = True 
        for k2 in visitedPoints:
            if k2!=k1 and _cutils.dominating(visitedPoints[k2].mean,visitedPoints[k1].mean):
                flag = False
                break
        if flag: paretoSet = dict(paretoSet,**{k1:visitedPoints[k1]})
    return paretoSet

def identifyParetoSetParallel(tree):
    """identify the Pareto Set by divide & conquer
    Args:
        tree: A class Tree() representing the search tree
    Returns:
        paretoSet: A dictionary including current Pareto systems
    """
    #identify local Pareto set for each leaf nodes
    localParetoSet = MultiThread(identifyParetoSet,zip([l.pool for l in tree.leafNodes()]))
    #identify current global Pareto set given those local Pareto sets
    visitedPoints = {}
    for foo in localParetoSet:
        for key in foo:
            visitedPoints[key] = foo[key]
    paretoSet = identifyParetoSet(visitedPoints)    
    return paretoSet

                
def withinRegion(x,lb,ub):
    """ check whether x belongs to the region [lb,ub]
    Args:
        x: A double numpy array representing a point
        lb: A double numpy array representing the lower bound
        ub: A double numpy array representing the upper bound        
    Returns:
        flag: A boolean value indicating whether x belongs to this region
    """
    flag = all(lb<=x) and all(x<=ub)
    return flag        
    
    
def generateKey(x):
    """ generate dictionary key for point x by hash function
    Args:
        x: A double numpy array representing the coordinate of point x 
    Returns:
        key: A string of hash value
    """
    key = str(hash(tuple(list(x))))
    return key
    

def identifyTrueParetoSet(problem):
    lb = problem.lb
    ub = problem.ub
    xlim = [lb[0],ub[0]]
    ylim = [lb[1],ub[1]]
    xv,yv = generateMeshGrid(xlim,ylim,problem.num)
    objs = calculateMultiObjective(xv,yv,problem.objectives)    
    ## Determine Global Pareto Set or Pareto Layer
    isPareto,dominationCount = determinePareto(objs)  
    return {'xv':xv,'yv':yv,'isPareto':isPareto,'dominationCount':dominationCount,'objs':objs}
        
    
def generateMeshGrid(xlim,ylim,num):
    x = np.linspace(min(xlim),max(xlim),num)
    y = np.linspace(min(ylim),max(ylim),num)
    xv,yv = np.meshgrid(x,y)
    return xv,yv
    
def calculateMultiObjective(xv,yv,fun):
    num_xy = len(xv)
    objs = []
    for foo in fun:
        _objs = np.empty((num_xy,num_xy),float)
        for i in range(num_xy):
            for j in range(num_xy):
                _objs[i][j] = foo(np.array([xv[i][j],yv[i][j]]))
        objs.append(_objs)    
    return objs

def determinePareto(objs):
    num_objs = len(objs)
    num_xy = len(objs[0])
    isPareto = np.ones([num_xy,num_xy])
    dominationCount = np.zeros([num_xy,num_xy])
    # pick xth, yth point
    for x1 in range(num_xy):
        for y1 in range(num_xy):
            sys1 = np.array([objs[k][x1][y1] for k in range(num_objs)])
            # check if xth, yth point is non-dominated
            for x2 in range(num_xy):
                for y2 in range(num_xy):
                    sys2 = np.array([objs[k][x2][y2] for k in range(num_objs)])
                    #if ((sys2-sys1 <= 1e-8).all()) and (min(sys2-sys1) < -1e-4) and ((x1,y1) != (x2,y2)):
                    if _cutils.dominating(sys2,sys1):
                        isPareto[x1][y1] = 0
                        dominationCount[x1][y1] += 1 

    return isPareto,dominationCount

def determineParetoLayer(objs,layer):
    num_objs = len(objs)
    num_xy = len(objs[0])
    # layer of all points are identified
    if (layer>0).all():
        return layer
    # exists some points whose layer is not identified
    else:
        layer_id = np.max(layer) + 1
        print('..identifying Pareto layer %d' % layer_id)        
        # pick x1th, y1th point which are not identified yet
        for x1 in range(num_xy):
            for y1 in range(num_xy):
                if layer[x1][y1] == 0:
                    sys1 = np.array([objs[k][x1][y1] for k in range(num_objs)])
                    layer[x1][y1] = layer_id
                    # pick x2th, y2th point which are not identified yet
                    for x2 in range(num_xy):
                        for y2 in range(num_xy):
                            if layer[x2][y2] == 0:
                                sys2 = np.array([objs[k][x2][y2] for k in range(num_objs)])
                                #if ((sys2-sys1 <= 1e-8).all()) and (min(sys2-sys1) < -1e-4) and ((x1,y1) != (x2,y2)):
                                if _cutils.dominating(sys2,sys1):
                                    layer[x1][y1] = 0
                                    break
                                else:
                                    continue
                            else:
                                continue                                
                else:
                    continue
        # find next layer
        return determineParetoLayer(objs,layer)
        
def discretize(samples, LB, UB, discreteLevel):
    """ discretize the samples based on the problem's discretized level
    Args:
        samples: An n * dimX array representing the uniformlly samples 
        LB: A numpy array indicating the lower bound of solution space
        UB: A numpy array indicating the upper bound of solution space
        discreteLevel: An integer indicating the discrete level (0 means continuous)
    Returns:
        discretizedSamples: An n * dimX array representing the discretized samples   
    """
    # check whether the optimization problem is discrete
    if discreteLevel != 0:
        # calculate the unit distance
        unit = (UB - LB) / discreteLevel
        # transform the real-valued p to "discrete-valued p"
        transform = lambda p: LB + np.round(((p - LB) / unit), 0) * unit
        # transform all the samples to the required format
        discretizedSamples = np.array(list(map(transform, samples)))
    else:
        discretizedSamples = samples    
    return discretizedSamples

def openWorkspace(filename):
    ''' restore the workspace '''
    f_shelf = shelve.open(filename)
    for key in f_shelf:
        globals()[key] = f_shelf[key]
    f_shelf.close()
    
def saveWorkspace(filename):
    # save the current workspace 
    f_shelf = shelve.open(filename, 'n')
    for key in dir():
        try:
            f_shelf[key] = globals()[key]
        except Exception as e:
            print(str(e))
            print('ERROR shelving: {0}'.format(key))
    f_shelf.close()  
    
def paretoSetToFront(paretoSet):
    """ derive the Pareto front gievn a dictionary of Pareto Set
    Args:
        paretoSet: A dictionary of class Point() representing the Pareto set
    Returns:
        front: A list of "Pareto" front objectives value
    """  
    front = [paretoSet[k].mean for k in paretoSet]
    return front

def paretoSetToTrueFront(paretoSet):
    """ derive the true Pareto front gievn a dictionary of estimated Pareto Set
    Args:
        paretoSet: A dictionary of class Point() representing the Pareto set
    Returns:
        front: A list of "Pareto" front objectives value
    """  
    front = [paretoSet[k].trueMean for k in paretoSet]
    return front
    
def paretoSetToList(paretoSet):
    """ convert a dictionary of paretoSet to list
    Args:
        paretoSet: A dictionary of class Point() representing the Pareto set
    Returns:
        x: A list of decision vector of Pareto set
    """
    x = [paretoSet[k].x for k in paretoSet]
    return x
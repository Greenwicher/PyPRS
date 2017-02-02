# -*- coding: utf-8 -*-
"""
    Created on Tue Mar 15 10:15:01 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""
from . import _cutils
from . import utils
from . import rule
from .  import objective
from . import visualize
from . import performance
from . import problemGMO
from . import images2gif
from . import hv
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
import datetime
import time
from numba import jit
import os
import copy

__all__ = ['utils','rule','_cutils','objective','images2gif','visualize','performance','hv']


class Case:
    """ summary of class Case()
    Attributes:
        description: A string of description of this instance
        problem: A class Problem() storing the information of the problem to be
                 solved
        prs: A class Core() storing the information of the customized PRS algorithm
        results: A dictionay storing the numerical results given problem and np
    Methods:
        init:
        run:
        visualize2DParetoFront:
        visualize2DAnimation:
        visualize2DScatter:
        calPerformance:
    """
    def __init__(self):
        self.description = ''
        self.dir = ''
        self.problem = None
        self.prs = None        
        self.results = {}
        return
        
    def init(self,args):
        """ initialize the arguments of the instance
        Args:
            args: A dictionary of the arguments to be initialized
        Returns:
            None
        """
        for attr in args:
            utils.updateObjAttr(self,attr,args[attr])
        #creater buffer directory to save the workspace and output
        if not(self.dir): self.dir = 'output/'+self.description + '_' +datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.problem.dim ==2 and not(os.path.exists(self.dir)):
            os.makedirs(self.dir)            
        return 
        
        
    def run(self):
        """
        Args:
            None
        Returns:
            self.results: A dictionay storing the numerical results given 
                          problem and np
        """
        self.results = self.prs.run(self.problem,self.dir)
        return self.results

        

class Core:
    """ summary of class Core()
    Attributes:
        description: A string illustrating customized algorithms        
        tree: A class Tree() storing the information of search tree
        rule: A class RuleSet() storing the information of PRS's component rule
        MPR: A list of class Tree representing the Most Promising Region (MPR)
        paretoSet: A dictionary of current Pareto set
        currentIteration: An integer indicating current iteration #
        currentPI: A double indicating current promixing index
        currentTreeLevel: An integer indicating current tree level
        startTime: A datetime list indicating start time of algorithm/iteration
        endTime: A datetime list indicating end time of algorithm/iteration
        computationTime: A double list indicating the duration time of 
                        algorithm/iteration
        sampleSize: An integer list indicating the number of samples at each iteration
        hyperVolume: A double list indicating the hyper volume value at each iteration
        trueParetoProportion: A double list indicating the true Pareto proportion
            at each iteration
        hausdorffDistance: A double list indicating the Hausdorff distance between
            estimated Pareto set and true Pareto set
    Methods:
        run: framework of Partition-based Random Search (PRS) algorithms
    """    
    def __init__(self):
        self.description = ''
        self.tree = None
        self.rule = None
        self.MPR = []
        self.paretoSet = {}
        self.currentIteration = np.NaN
        self.currentPI = np.inf
        self.currentTreeLevel = 0
        self.currentSampleSize = 0
        self.startTime = []
        self.endTime = []
        self.computationTime = []
        self.sampleSize = []
        self.hyperVolume = []
        self.trueParetoProportion = []
        self.hausdorffDistance = []
        return
    
    def init(self,args):
        """ initialize the arguments of the instance
        Args:
            args: A dictionary of the arguments to be initialized
        Returns:
            None
        """
        for attr in args:
            utils.updateObjAttr(self,attr,args[attr])
        return 
       
    def run(self,problem,outputDir):
        """ framework of Partition-based Random Search (PRS) algorithms
        Args:
            problem: A class Problem() storing the description of the problem 
                     to be solved
        Returns:
            results: A dictionary storing the numerical results of PRS algorithm
        """          
        #record start time of the PRS algorithm
        self.startTime.append(datetime.datetime.now())
        #initialize the search tree
        self.tree.addNode(None,problem.lb,problem.ub,problem)
        #initialize iteration information
        self.currentIteration = 1
        # retrieve all the visited points
        visitedPoints = self.tree.root.visitedPoints()          
        #partitioning->sampling->evaluation
        while not(self.rule.stop(self,self.rule.stopArgs)):               
            #record start time of this iteration
            self.startTime.append(datetime.datetime.now())            
            ### PARTIONING ####
            t = time.time()
            #determine MPR
            self.MPR = utils.identifyMPR(self.tree,self.rule.alphaPI)
            #exclude too small MPR
            largeMPR = [leafMPR for leafMPR in self.MPR if max(leafMPR.ub-leafMPR.lb) > self.rule.atomPartitionScale]          
            #partition MPR into subregions
            subregions = utils.MultiThread(self.rule.partition,zip(largeMPR, repeat(self.rule.partitionArgs)))          
            #add new node into the Tree
            for MPR in subregions:
                parent = MPR['parent']
                parent.thr = MPR['thr'] #update partition threshold of parent
                #add new children nodes
                for sub in MPR['subRegions']:                     
                    _node = Tree()
                    _node.addNode(parent,sub[0],sub[1],problem)  
            t1 = time.time() - t 
            ### SAMPLING ###     
            t = time.time()
            leafNodes = self.tree.leafNodes() #update leaf nodes
            #update sampling index
            samplingIndex = utils.MultiThread(self.rule.si,zip(leafNodes,repeat(self.rule.siArgs)))                    
            utils.MultiThread(utils.updateObjAttr,zip(leafNodes,repeat('samplingIndex'),samplingIndex))
            #determine sample size
            sampleSize = utils.MultiThread(self.rule.sampleSize,zip(leafNodes,repeat(self.rule.sampleSizeArgs)))
            #draw samples from each leaf nodes
            samples = utils.MultiThread(self.rule.sampleMethod,zip(leafNodes,sampleSize,repeat(self.rule.sampleMethodArgs)))
            t2 = time.time() - t 
            ### EVALUATION ###  
            t = time.time()                      
            #evaluate samples in each leaf         
            for spl in samples:
                #observe spls multi-objectives
                _points = spl['samples']
                node = spl['leaf'] 
                # remove already visited points (even for stochastic case), the feasiblity check is not elegant
                points = [p for p in _points if not(utils.generateKey(p) in visitedPoints) and utils.withinRegion(p, node.lb, node.ub)]
                #determine replication size for each points to be sampled
                if problem.stochastic:
                    #stochastic case
                    repSize = self.rule.replicationSize(node,points,self.MPR,self.rule.replicationSizeArgs)
                else:
                    #deterministic case
                    repSize = np.array([1]*len(points))                            
                #objectives = utils.MultiThread(problem.evaluate,zip(points,repSize))  #not efficient than the following methods          
                objectives = []
                for i in range(len(points)):
                    objectives.append(problem.evaluate(points[i], repSize[i]))
                #update pool for each node                            
                node.updatePool(points,objectives,problem)  
                
            #identify current Pareto set and draw more replications for them
#            if problem.stochastic:
#                paretoSet = utils.identifyParetoSetParallel(self.tree)                 
#                points = [paretoSet[k].x for k in paretoSet]
#                repSize = np.array([self.rule.replicationSizeArgs['paretoReplicationSize']]*len(paretoSet))
#                objectives = utils.MultiThread(problem.evaluate,zip(points,repSize))                
#                node.updatePool(points,objectives,problem) 
            paretoSet = utils.identifyParetoSetParallel(self.tree)
            visitedPoints = self.tree.root.visitedPoints()
            #update promising index                       
            promisingIndex = utils.MultiThread(self.rule.pi,zip(leafNodes))                         
            utils.MultiThread(utils.updateObjAttr,zip(leafNodes,repeat('promisingIndex'),promisingIndex))    
            t3 = time.time() - t 
            ### OTHERS - Iteration Update ###    
            t = time.time()
            self.currentIteration += 1
            self.currentTreeLevel = max([leaf.level for leaf in leafNodes])
            self.currentPI = min([leaf.promisingIndex for leaf in leafNodes])            
            visitedPoints = self.tree.root.visitedPoints()
            self.currentSampleSize = sum([len(visitedPoints[k].history) for k in visitedPoints])
            self.sampleSize.append(self.currentSampleSize)
            #calculate algorithm's performance metric
            self.hyperVolume.append(performance.calHyperVolume(utils.paretoSetToFront(paretoSet),problem.referencePoint))                        
            self.trueParetoProportion.append(performance.calTrueParetoProportion(utils.paretoSetToList(paretoSet), utils.paretoSetToList(problem.trueParetoSet)))
            self.hausdorffDistance.append(performance.calHausdorffDistance(utils.paretoSetToList(paretoSet), utils.paretoSetToList(problem.trueParetoSet)))
            #visualize current search progress
            if problem.dim == 2 and self.rule.animationOn:
                visualize.generateAnimationFrame(outputDir,self.currentIteration-1,self,problem.trueParetoSetInfo)
            #record end time of this iteration
            self.endTime.append(datetime.datetime.now()) 
            self.computationTime.append((self.endTime[-1]-self.startTime[-1]).total_seconds())   
            t4 = time.time() - t
            #message to the screen 
            print('Iteration %d [%s] \t Tree Level %d \t Num of LeafNodes %d \t  \
            Num of Samples %d \t PI = %.4f \t HV = %.4f \t GO = %.4f \t HD = %.4f \
            \t Duration = %.2fs [%.2f,%.2f,%.2f,%.2f]'  % (self.currentIteration-1,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),self.currentTreeLevel,
            len(leafNodes),self.currentSampleSize,self.currentPI,self.hyperVolume[-1],
            self.trueParetoProportion[-1],self.hausdorffDistance[-1],
            self.computationTime[-1],t1,t2,t3,t4))                        
        #make gif animation
        if problem.dim == 2 and self.rule.animationOn:
            try:
                visualize.generateAnimation(outputDir)
            except Exception as e:
                print(str(e))
        #return current Pareto set and MPR
        self.MPR = utils.identifyMPR(self.tree,self.rule.alphaPI)          
        self.paretoSet =  utils.identifyParetoSetParallel(self.tree)
        results = {'MPR':self.MPR,'paretoSet':self.paretoSet}  
        #record end time
        self.endTime.append(datetime.datetime.now())
        self.computationTime.append((self.endTime[-1]-self.startTime[0]).total_seconds())
        print('Total elapsed time: %.2fs' % (self.computationTime[-1]))
        return results   
        
    def moprs(maximumSampleSize=1000,deltaSampleSize=30,unitSampleSize=5,
              stop = rule.stop.exceedMaximumSampleSize,
              sampleMethod=rule.sampleMethod.uniform,
              sampleMethodArgs={},
              sampleSize=rule.sampleSize.samplingIndex,
              si_ucb_c=0.5,
              pi=rule.pi.minimumDominationCount,
              alphaPI=0,
              partition=rule.partition.bisection,atomPartitionScale=0,
              replicationSize=rule.replicationSize.equal,unitReplicationSize=5,replicationTimes=5,
              partitionArgs={},
              animationOn=True):
        #define rules
        ruleArgs = {
                'description' : 'Default Rule',
                'stop' : stop,
                'partition' : partition,
#                'sampleSize' : rule.sampleSize.samplingIndex,
                'sampleSize' : sampleSize,
                'replicationSize' : replicationSize,
                'sampleMethod' : sampleMethod,
                'sampleMethodArgs': sampleMethodArgs,
                'pi' : pi,
                'si' : rule.si.ucb,
                'siArgs': {'ucb_c':si_ucb_c},
                'stopArgs':{'maximumSampleSize': maximumSampleSize,},
                'sampleSizeArgs' : {'deltaSampleSize': deltaSampleSize,
                                    'unitSampleSize': unitSampleSize},
                'replicationSizeArgs' : {'unitReplicationSize':unitReplicationSize,
                                         'replicationTimes':replicationTimes,
                                         'paretoReplicationSize':10,
                                         'minimumStd':0.1},
                'alphaPI': alphaPI,
                'atomPartitionScale': atomPartitionScale,
                'partitionArgs': partitionArgs,
                'animationOn': animationOn,
            }
        r = RuleSet()
        r.init(ruleArgs)
        #new search tree
        tree = Tree()
        #define PRS algorithms
        algoArgs = {'description':'Default MO-PRS','rule':r,'tree':tree,}
        algo = Core()
        algo.init(algoArgs)  
        return algo
              
        
class Point:
    """
    Attributes:
        x: A numpy array representing the point's coordinate
        key: A string representing the key of the point (hash string)
        trueMean: A dobule array representing the true objectives mean
        history: A numpy array storing history sampling information of objectives
        mean: A double array representing estimated objectives mean
        std: A double array representing objectives std
        node: A class Tree() of leaf node containing the point
    Methods:
        init: initialize the class Point()
    """
    def __init__(self):
        self.x = None
        self.key = ''
        self.trueMean = np.NaN
        self.history = np.array([])        
        self.mean = np.NaN
        self.std = np.NaN        
        self.node = None
        return
    def init(self,x,node,problem):
        """initialize the class Point()
        Args:
            x: A double numpy array reppresenting the coordinate of point p
            node: A class Tree() of leaf nodes containing this point
            problem: A class Problem() of underlying problem
        Returns:
            None
        """
        self.x = x
        self.key = utils.generateKey(x)
        self.node = node
        flag = utils.withinRegion(x,problem.lb,problem.ub)        
        trueMean = np.array([])          
        for obj in problem.objectives:
            trueMean = np.append(trueMean,[np.NaN,obj(x)][flag])
        self.trueMean = trueMean
        return        

class Problem:
    """summary of class Problem()
    Attributes:
        description: A string representing the problem description
        lb: A numpy array representing the lower bound of the region
        ub: A numpy array representing the upper bound of the region
        objectives: A list of function handles representing multi-objectives
        num: An integer representing the size of xv of meshgrid 
        stochastic: A boolean indicating whether the problem is stochastic
        std: A double representing the std of objectives
        discreteLevel: An integer indicating the discrete level (0 means continuous)
        referencePoint: A numpy array representing random selected reference
          point
        trueParetoSet: A numpy array storing the discretized true Pareto set
        bestHyperVolume: A double representing the best hypervolume value given
          trueParetoSet
        dim:
         
    Methods:
        init: initialize the arguments of the instance
        evaluate: evaluate the objectives of point x
    """
    def __init__(self):
        self.description = ''
        self.lb = np.array([])
        self.ub = np.array([])
        self.objectives = None
        self.num = np.NaN
        self.stochastic = False
        self.std = 0.0
        self.discreteLevel = 0
        self.referencePoint = np.array([])
        self.trueParetoSet = np.array([])
        self.bestHyperVolume = np.NaN
        self.dim = np.NaN
        return
        
    def init(self,args):
        """ initialize the arguments of the instance
        Args:
            args: A dictionary of the arguments to be initialized
        Returns:
            None
        """        
        for attr in args:
            utils.updateObjAttr(self,attr,args[attr])
        if self.dim == 2:
            self.trueParetoSetInfo = utils.identifyTrueParetoSet(self)
        #find referencePoint to calculate hypervolume
        if not(self.referencePoint.size):
            objs = self.trueParetoSetInfo['objs']
            referencePoint = []
            for foo in objs:
                referencePoint.append(np.max(foo))
            self.referencePoint = np.array(referencePoint)
        #calculate best hypervolume
        if len(self.trueParetoSet):
            _trueParetoSet = {}    
            for p in self.trueParetoSet:
                point = Point()
                point.init(p,None,self)
                _trueParetoSet[point.key] = point
            self.trueParetoSet = _trueParetoSet
            front = [_trueParetoSet[k].trueMean for k in _trueParetoSet]            
            self.bestHyperVolume = performance.calHyperVolume(front, self.referencePoint)
        return 
        
    def evaluate(self,x,num=1):
        """ evaluate the objectives of point x
        Args:
            x: A double numpy array representing a point x
            num: An integer indicating the number of evaluations
        Returns:
            results: A list of double numpy array representing the observations 
                     of the objectives of this point
        """
        results = []
        flag = utils.withinRegion(x,self.lb,self.ub)
        for _ in range(num):
            result = np.array([])            
            for obj in self.objectives:
                result = np.append(result,[np.NaN,obj(x)+self.stochastic*np.random.normal(0,self.std)][flag])
            results.append(result)
        return results
        
    def evaluateTrue(self,x,num=1):
        """ evaluate the true objectives of point x
        Args:
            x: A double numpy array representing a point x
            num: An integer indicating the number of evaluations
        Returns:
            results: A list of double numpy array representing the observations 
                     of the objectives of this point
        """
        results = []
        flag = utils.withinRegion(x,self.lb,self.ub)
        for _ in range(num):
            result = np.array([])            
            for obj in self.objectives:
                result = np.append(result,[np.NaN,obj(x)][flag])
            results.append(result)
        return results        
        
    def zdt1(num,isStochastic,std=1,dim=2,referencePoint = np.array([]), discreteLevel = 0):
        """optimal solution = {x1=[0,1], xi=0}
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        objectives = [objective.zdt11,objective.zdt12]   
        if discreteLevel != 0:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0,1,discreteLevel+1)])
        else:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0.05,0.95,1000)])
        problemArgs = {
                        'description':'ZDT1',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,
                        'trueParetoSet': trueParetoSet,
                        'dim': dim,
                        'discreteLevel': discreteLevel,                        
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem     
        
    def zdt2(num,isStochastic,std=1,dim=2,referencePoint = np.array([]), discreteLevel = 0):
        """optimal solution = {x1=[0,1], xi=0}
        """        
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        objectives = [objective.zdt21,objective.zdt22]   
        if discreteLevel != 0:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0,1,discreteLevel+1)])
        else:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0.05,0.95,1000)])
        problemArgs = {
                        'description':'ZDT2',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,   
                        'dim': dim,          
                        'discreteLevel': discreteLevel,                        
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem         
        
    def zdt3(num,isStochastic,std=1,dim=2,referencePoint = np.array([]), discreteLevel = 0):
        """optimal solution = {x1=[0,1], xi=0}, only some x1=[0,1]
        """        
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        objectives = [objective.zdt31,objective.zdt32]    
        if discreteLevel != 0:
            _trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0,1,discreteLevel+1)])
        else:
            _trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0.05,0.95,1000)])
        objValue = [np.array([f(p) for f in objectives]) for p in _trueParetoSet]
        trueParetoSet = []
        _size = len(_trueParetoSet)
        for i in range(_size):
            flag = True
            for j in range(_size):
                if i!=j and all((objValue[j] - objValue[i]) <= 0.0): 
                    flag = False
                    break
            if flag: trueParetoSet.append(_trueParetoSet[i])
        trueParetoSet = np.array(trueParetoSet)
        problemArgs = {
                        'description':'ZDT3',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint, 
                        'trueParetoSet': trueParetoSet,    
                        'dim': dim,      
                        'discreteLevel': discreteLevel,                        
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem         

    def zdt4(num,isStochastic,std=1,dim=2,referencePoint = np.array([]), discreteLevel = 0):
        """optimal solution = {x1=[0,1], xi=0}
        """        
        lb = np.array([0.0]+[-5.0]*(dim-1))
        ub = np.array([1.0]+[5.0]*(dim-1))
        objectives = [objective.zdt41,objective.zdt42]   
        if discreteLevel != 0:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0,1,discreteLevel+1)])
        else:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0.05,0.95,1000)])
        problemArgs = {
                        'description':'ZDT4',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint, 
                        'trueParetoSet': trueParetoSet, 
                        'dim': dim,    
                        'discreteLevel': discreteLevel,                        
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem         

    def zdt6(num,isStochastic,std=1,dim=2,referencePoint = np.array([]), discreteLevel = 0):
        """optimal solution = {x1=[0,1], xi=0}
        """        
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        objectives = [objective.zdt61,objective.zdt62]  
        if discreteLevel != 0:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0,1,discreteLevel+1)])
        else:
            trueParetoSet = np.array([[x1]+[0]*(dim-1) for x1 in np.linspace(0.05,0.95,1000)])
        problemArgs = {
                        'description':'ZDT6',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem   

    def circle(num,isStochastic,std=1):
        """optimal solution = {x1^2+x2^2=1}
        """        
        lb = np.array([-1,-1])
        ub = np.array([1,1])
        objectives = [objective.circle,objective.circle]        
        problemArgs = {
                        'description':'Circle',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem       

    def line(num,isStochastic,std=1):
        lb = np.array([0,0])
        ub = np.array([4,4])
        objectives = [objective.line45,objective.circle2]        
        problemArgs = {
                        'description':'Line',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem  

    def fon(num,isStochastic,std=1,dim=2,referencePoint = np.array([]), discreteLevel = 0):
        """optimal solution = {x1=...=xn=[-1/sqrt(n),1/sqrt(n)]}
        """        
#        dim = 2
        lb = np.array([-4.0,]*dim)
        ub = np.array([4.0,]*dim)
        objectives = [objective.fon1,objective.fon2]   
        trueParetoSet = [[x1,]*dim for x1 in np.linspace(-1/np.sqrt(dim),1/np.sqrt(dim),10000)]   
        trueParetoSet = utils.discretize(trueParetoSet, lb, ub, discreteLevel)
        problemArgs = {
                        'description':'FON',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint, 
                        'trueParetoSet': trueParetoSet,   
                        'dim': dim,          
                        'discreteLevel': discreteLevel,                        
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem  

    def heart(num,isStochastic,std=1):
        """optimal solution = {x1^2+(x2-(x1^2)^(1/3))^2=1}
        """        
        lb = np.array([-2,-2])
        ub = np.array([2,2])
        objectives = [objective.Heart,objective.Heart]        
        problemArgs = {
                        'description':'Heart',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem    

    def NB(num,isStochastic,std=1):
        """optimal solution = {(1+2*np.sqrt(-(abs(x[1])-1)**2+1)-x[0])*(x[0]**3+x[0]**2-2*x[0])*(x[1]+2*x[0]+2)=0}
        """              
        lb = np.array([-3,-2])
        ub = np.array([4,2])
        objectives = [objective.NB,objective.NB]        
        problemArgs = {
                        'description':'NB',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem 

    def face(num,isStochastic,std=1):
        """optimal solution = {((x[0]-1)**2+x[1]**2-4)*((x[0]+1)**2+x[1]**2-4)*(x[0]**2+(x[1]-np.sqrt(3))**2-4)-6<=0}
        """                
        lb = np.array([-4,-3])
        ub = np.array([4,4])
        objectives = [objective.Face,objective.Face]        
        problemArgs = {
                        'description':'Face',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem                      

    def sch(num,isStochastic,std=1,referencePoint = np.array([]), discreteLevel = 0):
        """optimal solution = {x=[0,2]}
        """                
        lb = np.array([-10.0**3])
        ub = np.array([10.0**3])
        objectives = [objective.sch1,objective.sch2]        
        trueParetoSet = utils.discretize(np.array([x1 for x1 in np.linspace(0.0,2.0,1000)]), lb, ub, discreteLevel)
        problemArgs = {
                        'description':'SCH',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint, 
                        'trueParetoSet': trueParetoSet,   
                        'discreteLevel': discreteLevel, 
                        'dim': 1,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem    
        
    def pol(num,isStochastic,std=1,referencePoint=np.array([]), discreteLevel = 0):
        lb = np.array([-np.pi,-np.pi])
        ub = np.array([np.pi,np.pi])
        objectives = [objective.pol1,objective.pol2]        
        problemArgs = {
                        'description':'POL',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint':referencePoint,   
                        'discreteLevel': discreteLevel,                        
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem    
        
    def kur(num,isStochastic,std=1,dim=2,referencePoint=np.array([]), discreteLevel = 0):
        lb = np.array([-5.0,]*dim)
        ub = np.array([5.0,]*dim)
        objectives = [objective.kur1,objective.kur2]        
        problemArgs = {
                        'description':'KUR',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint, 
                        'dim': dim,      
                        'discreteLevel': discreteLevel,                        
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem     
        
    def dtlz1(num,isStochastic,std=1,dim=2,obj=2,referencePoint=np.array([]), discreteLevel = 0):
        """ Optimal solution = {x: x[-dim2:] = 0.5}
            see "Scalable Test Problems for Evolutionary Multi-Objective Optimization"
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        dim1, dim2 = obj - 1, dim - obj + 1
        objectives = []
        g = lambda x: 100 * (dim2 + np.sum((x[-dim2:] - .5)**2 - np.cos(20 * np.pi * (x[-dim2:] - .5))))
        objectives.append(lambda x: .5 * np.product(x[:dim1]) * (1 + g(x)))
        for i in range(2, obj):
            objectives.append(lambda x: .5 * np.product(x[:dim1-i+1]) * (1 - x[dim1-i+1]) * (1 + g(x)))
        objectives.append(lambda x: .5 * (1 - x[0]) * (1 + g(x)))
        
        n_bins = [discreteLevel+1, 100][discreteLevel == 0] * np.ones(dim1)
        bounds = np.repeat([(0,1)], dim1, axis = 0)
        x_free = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]].reshape(dim1,-1).T
        trueParetoSet = np.concatenate((x_free, .5 * np.ones([len(x_free), dim2])), axis=1)
        problemArgs = {
                        'description':'DTLZ1',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem
        
    def dtlz2(num,isStochastic,std=1,dim=2,obj=2,referencePoint=np.array([]), discreteLevel = 0):
        """ Optimal solution = {x: x[-dim2:] = 0.5}
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        dim1, dim2 = obj - 1, dim - obj + 1
        objectives = []
        g = lambda x: np.sum((x[-dim2:] - .5) ** 2)
        objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(x[:dim1] * np.pi / 2)))
        for i in range(2, obj):
            objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(x[:dim1-i+1] * np.pi / 2)) * np.sin(x[dim1-i+1] * np.pi / 2))
        objectives.append(lambda x: (1 + g(x)) * np.sin(x[0] * np.pi / 2))
        
        n_bins = [discreteLevel+1, 100][discreteLevel == 0] * np.ones(dim1)
        bounds = np.repeat([(0,1)], dim1, axis = 0)
        x_free = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]].reshape(dim1,-1).T
        trueParetoSet = np.concatenate((x_free, .5 * np.ones([len(x_free), dim2])), axis=1)
        problemArgs = {
                        'description':'DTLZ2',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem       
        
    def dtlz3(num,isStochastic,std=1,dim=2,obj=2,referencePoint=np.array([]), discreteLevel = 0):
        """ Optimal solution = {x: x[-dim2:] = 0.5}
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        dim1, dim2 = obj - 1, dim - obj + 1
        objectives = []
        g = lambda x: 100 * (dim2 + np.sum((x[-dim2:] - .5) ** 2 - np.cos(20 * np.pi * (x[-dim2:] - .5))))
        objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(x[:dim1] * np.pi / 2)))
        for i in range(2, obj):
            objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(x[:dim1-i+1] * np.pi / 2)) * np.sin(x[dim1-i+1] * np.pi / 2))
        objectives.append(lambda x: (1 + g(x)) * np.sin(x[0] * np.pi / 2))
        
        n_bins = [discreteLevel+1, 100][discreteLevel == 0] * np.ones(dim1)
        bounds = np.repeat([(0,1)], dim1, axis = 0)
        x_free = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]].reshape(dim1,-1).T
        trueParetoSet = np.concatenate((x_free, .5 * np.ones([len(x_free), dim2])), axis=1)
        problemArgs = {
                        'description':'DTLZ3',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem       

    def dtlz4(num,isStochastic,std=1,dim=2,obj=2,referencePoint=np.array([]), discreteLevel = 0, alpha=100):
        """ Optimal solution unknown
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        dim1, dim2 = obj - 1, dim - obj + 1
        objectives = []
        g = lambda x: np.sum((x[-dim2:] - .5) ** 2)
        objectives.append(lambda x: (1 + g(x)) * np.product(np.cos((x[:dim1] ** alpha) * np.pi / 2)))
        for i in range(2, obj):
            objectives.append(lambda x: (1 + g(x)) * np.product(np.cos((x[:dim1-i+1] ** alpha) * np.pi / 2)) * np.sin(x[dim1-i+1] * np.pi / 2))
        objectives.append(lambda x: (1 + g(x)) * np.sin((x[0] ** alpha) * np.pi / 2))
        trueParetoSet = np.array([])
        problemArgs = {
                        'description':'DTLZ4',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem   
        
    def dtlz5(num,isStochastic,std=1,dim=2,obj=2,referencePoint=np.array([]), discreteLevel = 0):
        """ Optimal solution unknown
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        dim1, dim2 = obj - 1, dim - obj + 1
        objectives = []
        g = lambda x: np.sum((x[-dim2:] - .5) ** 2)
        theta = lambda x1, x2: np.pi / 4 / (1 + g(x2)) * (1 + 2 * g(x2) * x1)
        objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(theta(x[:dim1], x[-dim2:]) * np.pi / 2)))
        for i in range(2, obj):
            objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(theta(x[:dim1-i+1], x[-dim2:]) * np.pi / 2)) * np.sin(x[dim1-i+1] * np.pi / 2))
        objectives.append(lambda x: (1 + g(x)) * np.sin(theta(x[0], x[-dim2:]) * np.pi / 2))

        trueParetoSet = np.array([])
        problemArgs = {
                        'description':'DTLZ5',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem       
        
    def dtlz6(num,isStochastic,std=1,dim=2,obj=2,referencePoint=np.array([]), discreteLevel = 0):
        """ Optimal solution unknown
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        dim1, dim2 = obj - 1, dim - obj + 1
        objectives = []
        g = lambda x: np.sum(x[-dim2:] ** 0.1)
        theta = lambda x1, x2: np.pi / 4 / (1 + g(x2)) * (1 + 2 * g(x2) * x1)
        objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(theta(x[:dim1], x[-dim2:]) * np.pi / 2)))
        for i in range(2, obj):
            objectives.append(lambda x: (1 + g(x)) * np.product(np.cos(theta(x[:dim1-i+1], x[-dim2:]) * np.pi / 2)) * np.sin(x[dim1-i+1] * np.pi / 2))
        objectives.append(lambda x: (1 + g(x)) * np.sin(theta(x[0], x[-dim2:]) * np.pi / 2))

        trueParetoSet = np.array([])
        problemArgs = {
                        'description':'DTLZ6',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem       
        
    def dtlz7(num,isStochastic,std=1,dim=2,obj=2,referencePoint=np.array([]), discreteLevel = 0):
        """ Optimal solution unknown
        """
        lb = np.array([0.0,]*dim)
        ub = np.array([1.0,]*dim)
        dim1, dim2 = obj - 1, dim - obj + 1
        objectives = []
        g = lambda x: 1 + 9 / dim2 * np.sum(x)
        for i in range(obj-1):
            objectives.append(lambda x: x[i])
        h = lambda x: obj - np.sum(x[:dim1] / (1 + g(x)) * (1 + np.sin(3 * np.pi * x[:dim1]))) 
        objectives.append(lambda x: (1 + g(x)) * h(x))

        trueParetoSet = np.array([])
        problemArgs = {
                        'description':'DTLZ7',
                        'lb':lb,
                        'ub':ub,
                        'objectives':objectives,
                        'num': num,
                        'stochastic': isStochastic,
                        'std': std,
                        'referencePoint': referencePoint,   
                        'trueParetoSet': trueParetoSet,  
                        'dim': dim,  
                        'discreteLevel': discreteLevel,
        }
        problem = Problem()
        problem.init(problemArgs)
        return problem       
        
class Race:
    """ Summary of class Race()
    Attributes:
        problemPRS: A PRS class Problem() 
        problemGMO: A PyGMO class Problem()
        PyGMONumPop: An integer indicating the number of population for PyGMO
        maximumSampleSize" An integer indicating the maximum sample size
        dir: A string indicating the output directory
    Methods:
        init: initialize the arguments of the instance
        runPRS: run PRS algorithm for once 
        runGMO: run PyGMO algorithm for once
        runRep: run algorithm independently for multiple times
    """
    def __init__(self):
        self.problemPRS = None
        self.problemGMO = None
        self.PyGMONumPop = -1
        self.maximumSampleSize = -1
        self.dir = ''
        return
        
    def init(self, args):
        """ initialize the arguments of the instance
        Args:
            args: A dictionary of the arguments to be initialized
        Returns:
            None
        """   
        for attr in args:
            utils.updateObjAttr(self,attr,args[attr])   
        # new output directory
        if not(self.dir):
            self.dir = 'output/%s (xdim=%d, ydim=%d, discrete=%d, stochastic=%s) - %s/' \
            % (self.problemPRS.description, self.problemPRS.dim, 
               len(self.problemPRS.objectives), self.problemPRS.discreteLevel,
               str(self.problemPRS.stochastic), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if self.output and not(os.path.exists(self.dir)):
            os.makedirs(self.dir)                       
        return

    def runPRS(self, key, alg):
        """ run PRS algorithm for once 
        Args:
            key: A string indicating the name of algorithm
            alg: A class Core() of PRS algorithm
        Returns:
            resutls: A dictionary of algorithm's resutls
        """
        alg = copy.deepcopy(alg)
        case = Case()
        caseArgs = {
                    'description': '%s_%s_' % (self.problemPRS.description,str(self.problemPRS.stochastic)) + key,
                    'problem': self.problemPRS,
                    'prs': alg,
                    'dir': self.dir + key + '-' +datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),        
        }
        case.init(caseArgs)
        case.run()
        visualize.All(case)        
        HV = case.prs.hyperVolume
        GO = case.prs.trueParetoProportion
        HD = case.prs.hausdorffDistance
        sampleSize = case.prs.sampleSize  
        paretoPoint = case.results['paretoSet']
        paretoSet = utils.paretoSetToList(paretoPoint)
        front = utils.paretoSetToFront(paretoPoint)
        results = {'sampleSize':sampleSize, 
                   'HV':HV,
                   'GO':GO,
                   'HD':HD,
                   'paretoSet':paretoSet,
                   'front':front,
                   'case':case,
                   }
        if self.output: self.archive(key, results)
        return results  

    def runGMO(self, key, alg):
        """ run PyGMO algorithm for once 
        Args:
            key: A string indicating the name of algorithm
            alg: A PyGMO class of PyGMO algorithm
        Returns:
            resutls: A dictionary of algorithm's resutls
        """        
        import PyGMO
        alg = copy.deepcopy(alg)        
        pop =  PyGMO.population(self.problemGMO, self.PyGMONumPop)
        HV, GO, HD, sampleSize, paretoSet, front = [[] for _ in range(6)]
        LB, UB, discreteLevel = self.problemPRS.lb, self.problemPRS.ub, self.problemPRS.discreteLevel
        trueParetoSet = utils.paretoSetToList(self.problemPRS.trueParetoSet)
        t = 0
        while(max(pop.problem.fevals, t) < self.maximumSampleSize):
            pop = alg.evolve(pop)
            # update paretoSet
            popList = [utils.discretize([np.array(individual.cur_x)], 
                        LB, UB, discreteLevel)[0] for individual in pop]
            objValue = {}
            candidateList = paretoSet + popList
            paretoSet, front = [], []
            for p in candidateList:
                objValue[utils.generateKey(p)] = [p, self.problemPRS.evaluate(p)[0]]
            for k1 in objValue:
                flag = True
                for k2 in objValue:
                    if k2 != k1 and _cutils.dominating(objValue[k2][1], objValue[k1][1]):
                        flag = False
                        break
                if flag: 
                    paretoSet.append(objValue[k1][0])
                    front.append(objValue[k1][1])
            HV.append(performance.calHyperVolume(front,self.problemPRS.referencePoint))
            GO.append(performance.calTrueParetoProportion(paretoSet, trueParetoSet))
            HD.append(performance.calHausdorffDistance(paretoSet, trueParetoSet))        
            sampleSize.append(pop.problem.fevals)    
            t += 1
            print('%s - Iteration %d \t HV = %.4f \t GO = %.4f \t HD = %.4f \t sampleSize = %d' % (key, len(HV), HV[-1], GO[-1], HD[-1], sampleSize[-1]))
        results = {'sampleSize':sampleSize, 
                   'HV':HV,
                   'GO':GO,
                   'HD':HD,
                   'paretoSet':paretoSet,
                   'front':front,
                   'pop':pop,
                   }   
        if self.output: 
            self.archive(key, results)                   
        return results
        
    def runRep(self, key, alg, numRep):
        """ run algorithm alg for numRep times
        Args:
            key: A string indicating the name of algorithm
            alg: A PyGMO class of PyGMO algorithm
            numRep: An integer indicating the replication times
        Returns:
            resutls: A list of algorithm's resutls for each replication
        """
        if isinstance(alg, Core):
            results = [self.runPRS(key, alg) for _ in range(numRep)]
        else:
            results = [self.runGMO(key, alg) for _ in range(numRep)]
        return [results, results[0]][numRep==1]
        
    def archive(self, key, results):
        """ store the archive information of single sample path
        Args:
            key: A string of algorithm's name
            results: A dictionary storing the algorithm's output
        Returns:
            None
        """
        import time
        time.sleep(1)
        filename = '%s_%s' % (key, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # write output to txt files
        f_txt = open(self.dir + filename + '.txt', "w", encoding='utf8')
        #for key in ['paretoSet', 'front', 'sampleSize', 'HV', 'GO', 'HD']:
        for key in results:
            try:
                for foo in results[key]:
                    f_txt.write('%s %s\n' % (key, str(foo)))
            except Exception as e:
                print('Archive error: ', e)
            f_txt.write('\n'*5)
        f_txt.close()
        # write output to db
        import shelve
        f_db = shelve.open(self.dir + filename, 'n')
        try:
            f_db['key'] = locals()['key']
            f_db['results'] = locals()['results']
        except Exception as e:
            print(str(e))
        f_db.close()          
        return
        
    def retrieve(in_dir, in_dir_id):
        """ retrieve the algorithm's performance output from .txt files
        Args:
            in_dir: A string indicating the path of .txt files
            in_dir_id: An integer to help us get the algorithm's name
        Returns:
            results: A dictionary storing the algorihtm's performance
        """
        import glob
        srccomp = "%s/*.txt" % in_dir
        srclst = glob.glob(srccomp)  
        
        results = {}
        # process every .txt files
        for src in srclst:
            # get the algorithm key 
            key = src.split('/')[in_dir_id].split('_')[0].strip()
            # initialize the results dict for algorithm
            if not(key in results):
                results[key] = {}
                results[key]['path'] = []
            # process the particular .txt file
            f = open(src, "r")
            result = {}
            for line in f:
                # get the performance name and content
                content = line.strip().split(' ')
                if len(content) == 2:
                    if not(content[0] in result):
                        result[content[0]] = []
                    try:
                        v = [float(content[1]), int(content[1])][content[0]=='sampleSize']
                        result[content[0]].append(v)
                    except:
                        continue
            results[key]['path'].append(result)
        return results
        
              
        
class RuleSet:
    """summary of class RuleSet()
    Attributes:        
        description: A string describing the instance
        stop: A function handle representing stopping rule
        stopArgs: A dictionary of arguments of function stop()
        partition: A function handle representing partition rule
        partitionArgs: A dictionary of arguments of function partition()
        sampleSize: A function handle representing sampleSize rule
        sampleSizeArgs: A dictionary of arguments of function sampleSize()
        sampleMethod: A function handle representing sampleMethod rule
        sampleMethodArgs: A dictionary of arguments of function sampleMethod()
        pi: A function handle representing how to calculate promising index
        piArgs: A dictionary of arguments of function pi()
        si: A function handle representing how to calculate sampling index
        siArgs: A dictionary of arguments of function si()
        alphaPI: A double incidating the percentile to determine promising index
        atomPartitionScale: A double indicating the smallest scale of atom leaf node region
        animationOn: A boolean indicating whether to generate the 2D animation
    Methods:
        init: initialize the arguments of the instance
    """
    def __init__(self):
        self.description = ''
        self.stop = None
        self.stopArgs = {}
        self.partition = None
        self.partitionArgs = {}        
        self.sampleSize = None
        self.sampleSizeArgs = {}
        self.replicationSize = None
        self.replicationSizeArgs = {}
        self.sampleMethod = None
        self.sampleMethodArgs = {}
        self.pi = None
        self.piArgs = {}
        self.si = None
        self.siArgs = {}  
        self.alphaPI = 0
        self.atomPartitionScale = 0 
        self.animationOn = True           
        return
        
    def init(self,args):
        """ initialize the arguments of the instance
        Args:
            args: A dictionary of the arguments to be initialized
        Returns:
            None
        """   
        for attr in args:
            utils.updateObjAttr(self,attr,args[attr])
        return
                
class Tree:
    """ Summary of class Tree()
    Attributes:
        root: A class Tree() representing the root node of search tree
        parent: A class Tree() representing the parent node 
        children: A list of class Tree() representing the children node 
        nodeSet: A list of all nodes of Tree
        leafNode: A boolean indicating whether it's leaf node
        thr: A double representing the partition threshold
        lb: A vector of lower bound of decision variables determining 
            feasible region 
        ub: A vector of upper bound of decision variables determining 
            feasible region 
        pool: A dictionary of sequence of sampled points with key as id of each
              point and the content as list of history sampling information
        samplingIndex: A double determining the size of sample points
        promisingIndex: A double determining which leaf node should be splitted
                        further
        level: An integer indicating the tree level 
        n: An integer indicating the number of sampled observations 
        problem: A class Problem() representing the underlying problem
    Methods:
        addNode: add new leaf node to the tree
        updatePool: update the information of sampled pool of this node
        leafNodes: return all the leaf nodes
        visitedPoints: return all the visited points
        withinNode: check if solution x is within the node region
        calDominationCount: calculate domination count of each point of the leaf node
    """
    def __init__(self):
        self.root = None
        self.parent = None
        self.children = []
        self.nodeSet = []
        self.leafNode = True
        self.thr = np.NaN
        self.lb = np.array([]) #larger or equal to, lb <= x
        self.ub = np.array([]) #strictly less than, x < ub 
        self.pool = {}
        self.samplingIndex = np.NaN
        self.promisingIndex = np.NaN
        self.level = np.NaN
        self.n = np.NaN
        self.problem = None
        return None
    
    def addNode(self, parent, lb, ub, problem):
        """ add node to the Tree
        Args: 
            parent: parent node 
            lb: lower bound of decision variables, lb <= x
            ub: upper bound of decision variables, x <= ub
            problem: A class Problem() 
        Returns:
            None
        """
        self.parent = parent
        #Two cases: root node or not
        if self.parent:
            #Update self
            self.root = parent.root            
            self.level = parent.level + 1
            #Update parent
            parent.children.append(self)
            self.parent.leafNode = False #parent node is not leaf node anymore   
            #filtering pool
            pool = {}
            for key in parent.pool:
                point = parent.pool[key]
                if utils.withinRegion(point.x,lb,ub):
                    pool = dict({key:point},**pool)
            self.pool = pool
            self.n = sum([len(pool[key].history) for key in pool])            
        else:
            #Update self as root 
            self.root = self
            self.level = 0
            self.pool = {}
            self.promisingIndex = 1
        self.root.nodeSet.append(self) #add new nodes to the nodeSet
        self.lb = lb
        self.ub = ub 
        self.problem = problem
        return 
        
    def updatePool(self,points,objectives,problem):
        """ update the information of sampled pool of this node
        Args:
            points: A list of numpy array representing the sampled points                
            objectives: A list of numpy array representing the realizations of
                        objectives of each points
            problem: A class Problem()
        Returns:
            None
        """
        #add spls to the pool
        for p,ol in zip(points,objectives):
            key = utils.generateKey(p) # determine the key of this point
            if key in self.pool:
                # only increase history list for stochastic optimization
                if self.problem.stochastic:
                    # add new observations to the already sampled points                
                    _p = self.pool[key] 
                    for o in ol:
                        _p.history = np.vstack([_p.history,o])     
                else:
                    # for deterministic case
                    continue
            else:
                # add observations to the new sampled points
                _p = Point()
                _p.init(p,self,problem)                
                _p.history = np.array([ol[0]])
                for i in range(1,len(ol)):
                    _p.history = np.vstack([_p.history,ol[i]])                      
                self.pool[key] = _p
            self.n += 1 #update the number of sampled observations
        #update statistic information of the pool
        for key in self.pool:            
            p = self.pool[key]
            p.mean = np.mean(p.history,axis=0)
            p.std = np.std(p.history,axis=0)
        return 
        
    def leafNodes(self):
        """ return all the leaf nodes
        Args:
            self: self.root plays a major role
        Returns:
            leaf: A list of all the leaf nodes
        """
        leaf = []
        for node in self.root.nodeSet:
            if node.leafNode:
                leaf.append(node)
        return leaf
        
    def visitedPoints(self):
        """return all the visited points
        Args:
            self: The class Tree() itself
        Returns:
            visitedPoints: A dictionary storing all the visied points
        """
        visitedPoints = {}
        for leaf in self.leafNodes():
            visitedPoints = dict(visitedPoints, **leaf.pool)
        return visitedPoints

    def withinNode(self,x):
        """ check if point x belongs to the node region
        Args:
            x: A double numpy array representing a point 
        Returns:
            A boolean value that indicates whether x is within the node
        """
        return all(self.lb<=x) and all(x<=self.ub)
    
    @jit
    def calDominationCount(self):
        """calculate domination count of each point of the leaf node
        Args:        
        Returns:
            dominationCount: A numpy array indicating the domination count of 
                             each point
        """
        #determine the deminsion of point's objective
        dim = len(self.problem.objectives)
        #recorganize all the visited points together into one sorted array
        _visitedPoints = utils.dictToSortedNumpyArray(self.root.visitedPoints(),dim)    
        #recorganize the points in the pool together into one sorted array 
        _pool = utils.dictToSortedNumpyArray(self.pool,dim)        

        if self.pool:
            #call swig c extension (nearly 4 times faster than original python function)
            dominationCount = _cutils.calDominationCount(_pool,_visitedPoints,len(_pool))[1]        
#            #call python function
#            dominationCount = utils.calDominationCount(_pool,_visitedPoints)
        else:
            dominationCount = np.array([np.nan])
        
        return dominationCount
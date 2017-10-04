# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 00:54:48 2016

@author: liuweizhi
"""

import PyPRS as prs
import numpy as np
import importlib
importlib.reload(prs)
import PyGMO
import time

t_begin = time.time()

# experiments setting
#np.random.seed(2017)
problemKey = 'ZDT1'
isStochastic = False
std = 1
numTrials = 100
numLatticeDim = 20
numPop = 60
animationOn = False

problemDim =5
numObj = 2
ref_point = (1000,) * numObj
discreteLevel = 20
numAtom = discreteLevel
numRep = 100
maximumSampleSize = 3000

#problem dictionary
probDict = {
    'ZDT1': ['ZDT1',prs.Problem.zdt1(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(1,problemDim)],
    'ZDT2': ['ZDT2',prs.Problem.zdt2(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(2,problemDim)],
    'ZDT3': ['ZDT3',prs.Problem.zdt3(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(3,problemDim)],
    'ZDT4': ['ZDT4',prs.Problem.zdt4(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(4,problemDim)],
    'ZDT6': ['ZDT6',prs.Problem.zdt6(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(6,problemDim)],
}
problemName,problemPRS,problemGMO = probDict[problemKey]
if isStochastic:
    problemGMO = PyGMO.problem.noisy(problemGMO, trials=numTrials, param_first=0.0, param_second=std, noise_type=PyGMO.problem.noisy.noise_distribution.NORMAL)

#%% Algorithm Definition
algoDict = {
    'NSGA-II':[PyGMO.algorithm.nsga_II(gen=1, cr=0.95, eta_c=10, m=0.01, eta_m=10),'g','^'],
    'SPEA2':[PyGMO.algorithm.spea2(gen=1, cr = 0.95, eta_c = 10, m = 0.01, eta_m = 50),'b','s'],
    'VEGA':[PyGMO.algorithm.vega(gen=10, cr=0.95, m=0.02, elitism=1, mutation=PyGMO.algorithm._algorithm._vega_mutation_type.GAUSSIAN, width=0.1, crossover=PyGMO.algorithm._algorithm._vega_crossover_type.EXPONENTIAL),'orange','d'],
    'SMSEMOA':[PyGMO.algorithm.sms_emoa(hv_algorithm=None, gen=100, sel_m=2, cr=0.95, eta_c=10, m=0.01, eta_m=10),'m','v'],
    'NSPSO':[PyGMO.algorithm.nspso(gen=1, minW = 0.4, maxW = 1.0, C1 = 2.0, C2 = 2.0, CHI = 1.0, v_coeff = 0.5),'olive','x'],
    'MPRS with NSGA-II':[prs.Core.moprs(
                             sampleMethod=prs.rule.sampleMethod.elite,
                             sampleMethodArgs = {'elite':{
                                                          'problemGMO': problemGMO,
                                                          'algGMO': PyGMO.algorithm.nsga_II(gen=1, cr=0.95, eta_c=10, m=0.01, eta_m=10),
                                                          'numPop': 16}},
                             deltaSampleSize = 100,
                             sampleSize=prs.rule.sampleSize.samplingIndex,
                             pi=prs.rule.pi.minimumDominationCount,
                             alphaPI=0,
                             atomPartitionScale=min((problemPRS.ub-problemPRS.lb)/numAtom),
                             partition=prs.rule.partition.xsection,
                             partitionArgs={'xsectionNum':2},
                             animationOn = animationOn,
                             stop = prs.rule.stop.exceedMaximumSampleSize,
                             stopArgs = {'maximumSampleSize':maximumSampleSize,
                                         'optimalityOn': True,},
                             replicationSize=prs.rule.replicationSize.equal, unitReplicationSize=numTrials),'r','*',],
   'MPRS Uniform':[prs.Core.moprs(
                            sampleMethod=prs.rule.sampleMethod.uniform,
                            deltaSampleSize = 100,
                            sampleSize=prs.rule.sampleSize.samplingIndex,
                            pi=prs.rule.pi.minimumDominationCount,
                            alphaPI=0,
                            atomPartitionScale=min((problemPRS.ub-problemPRS.lb)/numAtom),
                            partition=prs.rule.partition.xsection,
                            partitionArgs={'xsectionNum':2},
                            animationOn = animationOn,
                            stop = prs.rule.stop.exceedMaximumSampleSize,
                            stopArgs = {'maximumSampleSize':maximumSampleSize,
                                        'optimalityOn': True,},
                            replicationSize=prs.rule.replicationSize.equal, unitReplicationSize=numTrials),'r','o'],
}
#%% Invoke PyPRS and PyGMO algorithms    

# define the race
race = prs.Race()
race.init({'problemPRS': problemPRS, 
           'problemGMO': problemGMO,
           'PyGMONumPop': numPop,
           'maximumSampleSize': maximumSampleSize,
           'output': True,
           })

#%% Run tests

# run different algorithms
results = {}
for key in algoDict: 
    results[key] = {}
    results[key]['path'] = race.runRep(key, algoDict[key][0], numRep)
    results[key]['ensemble'] = prs.performance.calEnsembleMean(results[key]['path'], maximumSampleSize)

        
# visualization 
def convertToY(results, algoDict, yName):
    y = {}
    for key in results:
        y[key] = {}
        y[key]['ensemble'] = results[key]['ensemble'][yName][:maximumSampleSize]
        y[key]['color'] = algoDict[key][1]
        y[key]['marker'] = algoDict[key][2]
    return y
title = {}
for yName in ['Hypervolume', 'True Pareto Proportion', 'Hausdorff Distance']:
    title[yName] = 'Algorithm Comparisons on the Convergence of %s - %s (Dim=%d, Rep=%d)' % (yName, problemKey, problemDim, numRep)    
prs.visualize.plotConvergence(problemKey, 'Hypervolume', convertToY(results, algoDict, 'HV'), title['Hypervolume'], problemPRS.bestHyperVolume, race.dir)    
prs.visualize.plotConvergence(problemKey, 'True Pareto Proportion', convertToY(results, algoDict, 'GO'), title['True Pareto Proportion'], 1.0, race.dir)
prs.visualize.plotConvergence(problemKey, 'Hausdorff Distance', convertToY(results, algoDict, 'HD'), title['Hausdorff Distance'], 0.0, race.dir)    

# save workspace
prs.utils.saveWorkspace(race.dir+'workspace-%s' % (problemKey+'_'+str(isStochastic)))
print("Total Elapsed Time: %.2f minutes with Rep = %d" % ((time.time() - t_begin)/60, numRep))
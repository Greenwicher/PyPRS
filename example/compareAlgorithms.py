# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 00:54:48 2016

@author: liuweizhi
"""

import PyPRS as prs
import numpy as np
# import cProfile
import importlib
importlib.reload(prs)
import PyGMO

#experiments setting
np.random.seed(2016)
problemKey = 'ZDT6'
isStochastic = False
std = 1
numLatticeDim = 4
ref_point = (1000,) * 2
numPop = 12
numTrials = 5

problemDim = 2
discreteLevel = 100
numAtom = discreteLevel
numRep = 2
maximumSampleSize = 1000

#problem dictionary
probDict = {
    'ZDT1': ['ZDT1',prs.Problem.zdt1(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(1,problemDim)],
    'ZDT2': ['ZDT2',prs.Problem.zdt2(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(2,problemDim)],
    'ZDT3': ['ZDT3',prs.Problem.zdt3(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(3,problemDim)],
    'ZDT4': ['ZDT4',prs.Problem.zdt4(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(4,problemDim)],
    'ZDT6': ['ZDT6',prs.Problem.zdt6(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.zdt(6,problemDim)],
    'FON': ['FON',prs.Problem.fon(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.fon()],
    'KUR': ['KUR',prs.Problem.kur(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point),discreteLevel),PyGMO.problem.kur(problemDim)],
    'POL': ['POL',prs.Problem.pol(numLatticeDim,isStochastic,std,np.array(ref_point),discreteLevel),PyGMO.problem.pol()],
}
problemName,problemPRS,problemGMO = probDict[problemKey]
if isStochastic:
    problemGMO = PyGMO.problem.noisy(problemGMO, trials=numTrials, param_first=0.0, param_second=std, noise_type=PyGMO.problem.noisy.noise_distribution.NORMAL)

#%% Algorithm Definition
algoDict = {
    'NSGA-II':[PyGMO.algorithm.nsga_II(gen=1, cr=0.95, eta_c=10, m=0.01, eta_m=10),'g','o'],
    'SPEA2':[PyGMO.algorithm.spea2(gen = 1, cr = 0.95, eta_c = 10, m = 0.01, eta_m = 50),'b','x'],
    'SMSEMOA':[PyGMO.algorithm.sms_emoa(hv_algorithm=None, gen=1, sel_m=2, cr=0.95, eta_c=10, m=0.01, eta_m=10),'m','.'],
    'IHS':[PyGMO.algorithm.ihs(iter=1, hmcr=0.85, par_min=0.35, par_max=0.99, bw_min=1e-05, bw_max=1),'pink','+'],              
    'VEGA':[PyGMO.algorithm.vega(gen=1, cr=0.95, m=0.02, elitism=1, mutation=PyGMO.algorithm._algorithm._vega_mutation_type.GAUSSIAN, width=0.1, crossover=PyGMO.algorithm._algorithm._vega_crossover_type.EXPONENTIAL),'orange','x'],
    'NSPSO':[PyGMO.algorithm.nspso(gen =1, minW = 0.4, maxW = 1.0, C1 = 2.0, C2 = 2.0, CHI = 1.0, v_coeff = 0.5),'purple','o'],
    'PADE':[PyGMO.algorithm.pade(gen=1, decomposition='tchebycheff', weights='grid', solver=None, threads=8, T=8, z=[]),'firebrick','*'],
}

if isStochastic:
    algoDict.update({
        'MO-PRS_EqualRep':[prs.Core.moprs(maximumSampleSize=maximumSampleSize,replicationSize=prs.rule.replicationSize.equal,unitReplicationSize=5),'r','*'],                
#        'MO-PRS_GreedyRep':[prs.Core.moprs(maximumSampleSize=maximumSampleSize,replicationSize=prs.rule.replicationSize.mprGreedy,unitReplicationSize=5,replicationTimes=5),'brown','v'],                             
    })
else:
    algoDict.update({
        'MO-PRS':[prs.Core.moprs(maximumSampleSize=maximumSampleSize, 
                                 sampleMethod=prs.rule.sampleMethod.elite, 
                                 sampleMethodArgs = {'elite':{
                                                              'problemGMO': problemGMO,
                                                              'algGMO': algoDict['NSGA-II'][0]}}, 
                                 deltaSampleSize=30, unitSampleSize=16,
                                 sampleSize=prs.rule.sampleSize.equalSize,
                                 pi=prs.rule.pi.minimumDominationCount,
                                 alphaPI=0,                                 
                                 atomPartitionScale=min((problemPRS.ub-problemPRS.lb)/numAtom), 
                                 partition=prs.rule.partition.xsection,
                                 partitionArgs={'xsectionNum':2}),'r','*'],                                  
                                 
    })
#%% Invoke PyPRS and PyGMO algorithms    

race = prs.Race()
race.init({'problemPRS': problemPRS, 
           'problemGMO': problemGMO,
           'PyGMONumPop': numPop,
           'maximumSampleSize': maximumSampleSize,
           })
results = {}
for key in algoDict:
    results[key] = {}
    results[key]['path'] = race.runRep(key, algoDict[key][0], numRep)
    results[key]['ensemble'] = prs.performance.calEnsembleMean(results[key]['path'])
    
def convertToY(results, algoDict, yName):
    y = {}
    for key in results:
        y[key] = {}
        y[key]['ensemble'] = results[key]['ensemble'][yName][:maximumSampleSize]
        y[key]['color'] = algoDict[key][1]
        y[key]['marker'] = algoDict[key][2]
    return y

title = {}
for yName in ['Hypervolume', 'True Pareto Solution Proportion', 'Hausdorff Distance']:
    title[yName] = 'Algorithm Comparisons on the Convergence of %s - %s (Dim=%d, Rep=%d)' % (yName, problemKey, problemDim, numRep)

    
prs.visualize.plotConvergence(problemKey, 'Hypervolume', convertToY(results, algoDict, 'HV'), title['Hypervolume'], problemPRS.bestHyperVolume)    
prs.visualize.plotConvergence(problemKey, 'True Pareto Solution Proportion', convertToY(results, algoDict, 'GO'), title['True Pareto Solution Proportion'], 1.0)    
prs.visualize.plotConvergence(problemKey, 'Hausdorff Distance', convertToY(results, algoDict, 'HD'), title['Hausdorff Distance'], 0.0)    

#%% save workspace
prs.utils.saveWorkspace('output/workspace-%s' % (problemKey+'_'+str(isStochastic))) 

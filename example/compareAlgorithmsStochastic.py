# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 00:54:48 2016

@author: liuweizhi
"""

import PyNP as pnp
from PyNP.objective import *
import numpy as np
import matplotlib.pyplot as plt
# import cProfile
import importlib
importlib.reload(pnp)
import PyGMO
#from sklearn import gaussian_process
import copy

#experiments setting
#np.random.seed(2016)
problemKey = 'ZDT6'
problemDim = 1000
std = 1
isStochastic = False
numLatticeDim = 40
numRep = 20
ref_point = (100,) * 2
maximumSampleSize = 1000
numPop = 20
numTrials = 5

#problem dictionary
probDict = {
    'ZDT1': ['ZDT1',pnp.Problem.zdt1(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point)),PyGMO.problem.zdt(1,problemDim)],
    'ZDT2': ['ZDT2',pnp.Problem.zdt2(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point)),PyGMO.problem.zdt(2,problemDim)],
    'ZDT3': ['ZDT3',pnp.Problem.zdt3(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point)),PyGMO.problem.zdt(3,problemDim)],
    'ZDT4': ['ZDT4',pnp.Problem.zdt4(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point)),PyGMO.problem.zdt(4,problemDim)],
    'ZDT6': ['ZDT6',pnp.Problem.zdt6(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point)),PyGMO.problem.zdt(6,problemDim)],
    'FON': ['FON',pnp.Problem.fon(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point)),PyGMO.problem.fon()],
    'KUR': ['KUR',pnp.Problem.kur(numLatticeDim,isStochastic,std,problemDim,np.array(ref_point)),PyGMO.problem.kur(problemDim)],
    'POL': ['POL',pnp.Problem.pol(numLatticeDim,isStochastic,std,np.array(ref_point)),PyGMO.problem.pol()],
}
problemName,problemNP,problemGMO = probDict[problemKey]
if isStochastic:
    problemGMO = PyGMO.problem.noisy(problemGMO, trials=numTrials, param_first=0.0, param_second=std, noise_type=PyGMO.problem.noisy.noise_distribution.NORMAL)

algoDict = {
    'NSGA-II':[PyGMO.algorithm.nsga_II(gen=1, cr=0.95, eta_c=10, m=0.01, eta_m=10),'g','o'],
    'SPEA2':[PyGMO.algorithm.spea2(gen = 1, cr = 0.95, eta_c = 10, m = 0.01, eta_m = 50),'b','x'],
    'SMSEMOA':[PyGMO.algorithm.sms_emoa(hv_algorithm=None, gen=1, sel_m=2, cr=0.95, eta_c=10, m=0.01, eta_m=10),'m','.'],
    'IHS':[PyGMO.algorithm.ihs(iter=1, hmcr=0.85, par_min=0.35, par_max=0.99, bw_min=1e-05, bw_max=1),'orange','+'],              
#    'VEGA':[PyGMO.algorithm.vega(gen=1, cr=0.95, m=0.02, elitism=1, mutation=PyGMO.algorithm._algorithm._vega_mutation_type.GAUSSIAN, width=0.1, crossover=PyGMO.algorithm._algorithm._vega_crossover_type.EXPONENTIAL),'orange','x'],
#    'NSPSO':[PyGMO.algorithm.nspso(gen =1, minW = 0.4, maxW = 1.0, C1 = 2.0, C2 = 2.0, CHI = 1.0, v_coeff = 0.5),'orange','o'],
#    'PADE':[PyGMO.algorithm.pade(gen=1, decomposition='tchebycheff', weights='grid', solver=None, threads=8, T=8, z=[]),'orange','*'],
}

if isStochastic:
    algoDict.update({
        'MO-PRS_EqualRep':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.equal,unitReplicationSize=5),'r','*'],                
#        'MO-PRS_GreedyRep':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.mprGreedy,unitReplicationSize=5,replicationTimes=5),'brown','v'],                             
    })
else:
    algoDict.update({
        'MO-PRS':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize),'r','*'],                         
    })
#%% select good replication size rule for stochastic MO-PRS
#algoDict = {
#    'MO-PRS_Greedy_5_2':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.mprGreedy,unitReplicationSize=5,replicationTimes=2),'r','*'],
#    'MO-PRS_Greedy_10_2':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.mprGreedy,unitReplicationSize=10,replicationTimes=2),'g','*'],
#    'MO-PRS_Greedy_5_5':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.mprGreedy,unitReplicationSize=5,replicationTimes=5),'b','*'],
#    'MO-PRS_Equal_5':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.equal,unitReplicationSize=5),'m','*'],              
#    'MO-PRS_Equal_10':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.equal,unitReplicationSize=10),'orange','*'],                                  
#    'MO-PRS_Equal_20':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,replicationSize=pnp.rule.replicationSize.equal,unitReplicationSize=20),'y','*'],                                     
#}

#%% select good promising index / sample size rule for deterministic MO-PRS
#algoDict = {
#    'MO-PRS_equalSize_min':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.equalSize,pi=pnp.rule.pi.minimumDominationCount),'r','*'],
#    'MO-PRS_equalSize_mu':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.equalSize,pi=pnp.rule.pi.averageDominationCount),'g','*'],
#    'MO-PRS_equalSize_max':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.equalSize,pi=pnp.rule.pi.maximumDominationCount),'b','*'],
#    'MO-PRS_samplingIndex_min':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.samplingIndex,pi=pnp.rule.pi.minimumDominationCount),'m','*'],
#    'MO-PRS_samplingIndex_mu':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.samplingIndex,pi=pnp.rule.pi.averageDominationCount),'orange','*'],
#    'MO-PRS_samplingIndex_max':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.samplingIndex,pi=pnp.rule.pi.maximumDominationCount),'y','*'],
#}

#%% select good alphaPI for deterministic MO-PRS
#algoDict = {
#    'MO-PRS_equalSize_min':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.equalSize,pi=pnp.rule.pi.minimumDominationCount,alphaPI=0),'r','*'],
#    'MO-PRS_equalSize_mu':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.equalSize,pi=pnp.rule.pi.minimumDominationCount,alphaPI=25),'g','*'],
#    'MO-PRS_equalSize_max':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.equalSize,pi=pnp.rule.pi.minimumDominationCount),alphaPI=50,'b','*'],
#    'MO-PRS_samplingIndex_min':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.samplingIndex,pi=pnp.rule.pi.minimumDominationCount,alphaPI=0),'m','*'],
#    'MO-PRS_samplingIndex_mu':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.samplingIndex,pi=pnp.rule.pi.minimumDominationCount,alphaPI=25),'orange','*'],
#    'MO-PRS_samplingIndex_max':[pnp.NP.moprs(maximumSampleSize=maximumSampleSize,sampleSize=pnp.rule.sampleSize.samplingIndex,pi=pnp.rule.pi.minimumDominationCount,alphaPI=50),'y','*'],
#}
#%%Algorithms Comparisons
def runMOPRS(key,alg):
    case = pnp.Case()
    caseArgs = {
                'description': '%s_%s_' % (problemName,str(isStochastic)) + key,
                'problem': problemNP,
                'np': alg,
    }
    case.init(caseArgs)
    case.run()
    pnp.visualize.All(case)        
    HV = case.np.hyperVolume
    sampleSize = case.np.sampleSize     
    return sampleSize,HV
def runPyGMO(key,alg):
    pop =  PyGMO.population(problemGMO, numPop)
    HV = []
    sampleSize = []
    i=1
    while(pop.problem.fevals<maximumSampleSize):
        pop = alg.evolve(pop)
        if isStochastic:
            front = []
            for individual in pop:
                front.append(problemNP.evaluateTrue(np.array(individual.cur_x))[0])
            hv = PyGMO.hypervolume(front)
        else:
            hv = PyGMO.hypervolume(pop)
        sampleSize.append(pop.problem.fevals)
        hypervolume = hv.compute(ref_point)
        HV.append(hypervolume)
        print('%s - Iteration %d \t HV = %.5f \t sampleSize = %d' % (key, i, HV[-1],sampleSize[-1]))
        i+=1
    return sampleSize,HV
    
    
_HV,HVL,HVM,HVU = {},{},{},{}
sampleSize = {}
case = {}
for key in algoDict: 
    sampleSize[key] = np.arange(1,maximumSampleSize+1,1)      
    _HV[key] = []  
    for _ in range(numRep):
        alg = copy.deepcopy(algoDict[key][0])        
        if 'MO-PRS' in key:
            s,h = runMOPRS(key,alg)
        else:        
            s,h = runPyGMO(key,alg)
        plt.plot(s,h)
        #extend to continuous case
        s = [1]+s
        h = [np.nan]+h
        _h = []
        for i in range(len(h)-1):
             _h+= [h[i]]*(s[i+1]-s[i])
        _h+=[h[-1]]
        _HV[key].append(_h)  
    HVM[key] = []
    HVL[key] = []
    HVU[key] = []   
    for x in range(maximumSampleSize):
        hvs = np.array([_HV[key][rep][x] for rep in range(numRep)])
        meanHVS = np.mean(hvs)
        stdHVS = np.std(hvs)
        HVL[key].append(meanHVS-1.96*stdHVS) 
        HVM[key].append(meanHVS) 
        HVU[key].append(meanHVS+1.96*stdHVS) 
    HVL[key] = np.array(HVL[key])
    HVM[key] = np.array(HVM[key])
    HVU[key] = np.array(HVU[key])    


#%%Plotting figures
def plotHVConvergence(sampleSize,HVL,HVM,HVU):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #sns.set_style('darkgrid',{'axes.grid' : True}) 
    miny, maxy = np.inf, -np.inf
    for key in HVM:
        markers_on = list(np.linspace(1,maximumSampleSize-1,10))
        plt.plot(sampleSize[key],HVM[key],color=algoDict[key][1],ls='-',linewidth=1)        
        plt.plot(sampleSize[key][markers_on],HVM[key][markers_on],color=algoDict[key][1],marker=algoDict[key][2],ls='None',label=key)
        miny = min(miny,np.nanmin(HVM[key]))
        maxy = max(maxy,np.nanmax(HVM[key]))

#        if 'MO-PRS' in key:
#            plt.fill_between(sampleSize[key],HVL[key],HVU[key],color=algoDict[key][1],alpha=.1)
#            #estimate GaussianProcess
#            X = np.atleast_2d(sampleSize[key]).T
#            y = np.array(HVM[key]).T
#            gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#            gp.fit(X,y)
#            x = np.atleast_2d(np.linspace(0,max(sampleSize[key]),10)).T
#            y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)
#            sigma = np.sqrt(sigma2_pred)
#            #plot GaussianProcess
#            plt.plot(X, y, color=algoDict[key][1],marker=algoDict[key][2],linewidth=1,label=key,ls='None')
#    #        plt.plot(x, y_pred, color=algoDict[key][1],ls='-')
#            plt.fill(np.concatenate([x, x[::-1]]),
#                    np.concatenate([y_pred - 1.96 * sigma,
#                                   (y_pred + 1.96 * sigma)[::-1]]),
#                    alpha=.2, fc=algoDict[key][1], ec='None')  
                
    bestHyperVolume = probDict[problemKey][1].bestHyperVolume
    maxy = np.nanmax([maxy,bestHyperVolume])
#    miny = max([np.percentile(HVM[key],5) for key in HVM])
#    miny = 1000
    if bestHyperVolume:
        plt.plot(range(1,maximumSampleSize+1),[bestHyperVolume]*maximumSampleSize,ls='-',color='k',linewidth=1,label='Best Hypervolume')
    ax.legend(loc='best')    
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2)
    plt.title('Algorithm Comparisons on the Convergence of Hypervolume - %s (Dim=%d,Rep=%d)' % (problemKey,problemDim,numRep))
    plt.xlabel('Sample Size')
    plt.ylabel('Hypervolume')
    plt.grid()
    ax.set_ylim([miny,maxy+0.1*(maxy-miny)])
    ax.set_xlim([1,maximumSampleSize])
    fig.set_size_inches(12, 8)
    #sns.set(rc={"figure.figsize": (12, 8)})
    fig.savefig('output/algo-comparison-%s.png' % (problemKey+'_'+str(isStochastic)), dpi=200) 
    fig.savefig('output/algo-comparison-%s.eps' % (problemKey+'_'+str(isStochastic)), dpi=200) 
    plt.close()
    return 
    
    
plotHVConvergence(sampleSize,HVL,HVM,HVU)

#%% save workspace
pnp.utils.saveWorkspace('output/workspace-%s' % (problemKey+'_'+str(isStochastic))) 

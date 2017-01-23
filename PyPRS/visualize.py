# -*- coding: utf-8 -*-
"""
    Created on Sat Apr  2 11:12:17 2016
    @author: liuweizhi (greenwicher.com， weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""
from . import utils
from . import images2gif
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import re

def partitionBounds(leafNodes):
    for leaf in leafNodes:
        lb = leaf.lb
        ub = leaf.ub
        x1,y1 = lb
        x2,y2 = ub
        plt.plot([x1,x2],[y1,y1],'r-')
        plt.plot([x2,x2],[y1,y2],'r-')
        plt.plot([x2,x1],[y2,y2],'r-')
        plt.plot([x1,x1],[y2,y1],'r-')  
    return 

def Objective2D(ax,xv,yv,objs,title):
    # 3D Surface Plot for Objective 1
    ax.plot_surface(xv,yv,objs,rstride=3,cstride=3,alpha=0.3,cmap=cm.BuPu)
    ax.contourf(xv,yv,objs,zdir='z',cmap=cm.coolwarm, offset=np.min(objs))
    ax.contourf(xv,yv,objs,zdir='x',cmap=cm.coolwarm, offset=np.min(xv))
    ax.contourf(xv,yv,objs,zdir='y',cmap=cm.coolwarm, offset=np.max(yv))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    #Objective i
    plt.title(title)
    return    
        
def GlobalParetoFront2D(objs,title):
    num = len(objs[0])
    plt.scatter(objs[0].reshape(num**2),objs[1].reshape(num**2))
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    #'Pareto Front - (%d Solutions)' % num**2
    plt.title(title) 
    plt.grid()
    return    
    
def Contour2D(fig,ax,xv,yv,mapping,leafNodes,title,partition):
    cs = ax.contourf(xv,yv,mapping,levels=np.arange(np.min(mapping),np.max(mapping)+0.5,0.5))
    fig.colorbar(cs, ax=ax)
    plt.grid()
    plt.xlabel('x1')
    plt.ylabel('x2')
    if partition: partitionBounds(leafNodes)    
    #'Domination Count - (%d Solutions)' % num**2
    plt.title(title)
    return    
          
def Scatter2D(pointSet,leafNodes,title,partition):
    for key in pointSet:
        point = pointSet[key]
        plt.scatter(*point.x)
    if partition: partitionBounds(leafNodes)
    plt.grid()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    return
    
def Scatter2DColor(pointSet,attr,leafNodes,title,xlabel,ylabel,partition):
    import numpy as np
    from scipy.stats import gaussian_kde
    x,y = [np.array([]) for i in range(2)]
    for key in pointSet:
        point = pointSet[key]
        foo = getattr(point,attr)
        x = np.append(x,foo[0])
        y = np.append(y,foo[1])

    try:
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)    
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        plt.scatter(x,y,c=z,s=100,edgecolor='')       
    except Exception as e:
        print(str(e))
        plt.scatter(x,y)

    if partition: partitionBounds(leafNodes)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return 
    
def Series(x,y,title,xlabel,ylabel):
    plt.plot(x,y,'r-')
    plt.xlim(min(x),max(len(y),30))
    plt.ylim(min(y)-0.1*(max(y)-min(y)),max(y)+0.1*(max(y)-min(y)))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()    
    return 
  
def colorRegion(leafNodes,colorLeafs,color,title,xlabel,ylabel):
    partitionBounds(leafNodes)
    for leaf in colorLeafs:
        lx,ly = leaf.lb
        ux,uy = leaf.ub        
        x = np.linspace(lx,ux,1000)
        ly = np.array([ly]*len(x))
        uy = np.array([uy]*len(x))
        plt.fill_between(x,ly,uy,color=color,alpha='0.5')
#    for leaf in leafNodes:
#        lx,ly = leaf.lb
#        ux,uy = leaf.ub            
#        spls = sum([len(leaf.pool[k].history) for k in leaf.pool])        
#        plt.text((lx+ux)/2,(ly+uy)/2,'(%.2f,%.2f,%d)' % (leaf.promisingIndex,leaf.samplingIndex,spls),fontsize=1)        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()    
    return 
        
    
def All(case):
    if case.problem.dim != 2:
        return 
    num = case.problem.num
    leafNodes = case.prs.tree.root.leafNodes()
    results = case.problem.trueParetoSetInfo
    xv = results['xv']
    yv = results['yv']
    isPareto = results['isPareto']
    dominationCount = results['dominationCount']
    objs = results['objs']    
    #layer = np.zeros([num,num])
    #paretoLayer = utils.determineParetoLayer(objs,layer)
    fig = plt.figure()        

    # 3D Surface Plot for Objective 1
    ax = fig.add_subplot(241,projection='3d')
    Objective2D(ax,xv,yv,objs[0],'Objective 1 - (%d Solutions)' % num**2)
            
    # 3D Surface Plot for Objective 2
    ax = fig.add_subplot(242,projection='3d')
    Objective2D(ax,xv,yv,objs[1],'Objective 2 - (%d Solutions)' % num**2)
    
    # Global Pareto Set Contour Plot
    ax = fig.add_subplot(243)
    Contour2D(fig,ax,xv,yv,isPareto,leafNodes,'Scatter of True Pareto Set - (%d Solutions)' % num**2,True)
    
    # Pareto Front Plot
    ax = fig.add_subplot(244) 
    GlobalParetoFront2D(objs,'Pareto Front - (%d Solutions)' % num**2)                              
    
    # Domination Count Contour Plot
    ax = fig.add_subplot(245)
    Contour2D(fig,ax,xv,yv,dominationCount,leafNodes,'Domination Count - (%d Solutions)' % num**2,True)
    
#    # Pareto Layer Contour Plot
#    ax = fig.add_subplot(246)
#    Contour2D(fig,ax,xv,yv,paretoLayer,leafNodes,'Pareto Layer - (%d Solutions)' % num**2,True)
    
    #visitedSolutions
    fig.add_subplot(246)
    visitedPoints = case.prs.tree.visitedPoints()
    Scatter2DColor(visitedPoints,'x',leafNodes,'Scatter of All Visited Points','x1','x2',True)    
        
    #paretoSet
    fig.add_subplot(247)
    paretoSet = case.results['paretoSet']
    Scatter2DColor(paretoSet,'x',leafNodes,'Scatter of Estimated Pareto Set','x1','x2',True)

    #estimated Pareto front
    fig.add_subplot(248)
    Scatter2DColor(visitedPoints,'mean',leafNodes,'Estimated Pareto Front','obj1','obj2',False)    
    
    #save figures
    fig.set_size_inches(20, 10)        
    fig.savefig(case.dir+'/2D.png',dpi=200)
    fig.savefig(case.dir+'/2D.eps',dpi=200)    
    plt.close()
    
    return 
    
def generateAnimationFrame(outputDir,i,prs,trueParetoSet):
    tree = prs.tree
    leafNodes = tree.root.leafNodes()
    xv = trueParetoSet['xv']
    yv = trueParetoSet['yv']
    isPareto = trueParetoSet['isPareto']
    dominationCount = trueParetoSet['dominationCount']
    fig = plt.figure()             
    
    # Domination Count Contour Plot
    ax = fig.add_subplot(241)
    Contour2D(fig,ax,xv,yv,dominationCount,leafNodes,'Domination Count (Iteration %d)' % (i),True)
    
#    # Global Pareto Set Contour Plot
#    ax = fig.add_subplot(232)
#    Contour2D(fig,ax,xv,yv,isPareto,leafNodes,'True Pareto Set (Iteration %d)' % (i),True)      
    
    #visitedSolutions
    fig.add_subplot(242)
    MPR = utils.identifyMPR(tree,prs.rule.alphaPI)
    colorRegion(leafNodes,MPR,'blue','Convergence of MPR (Iteration %d)' % (i), 'x1','x2')    
    
    # Convergence of Hypervolume
    ax = fig.add_subplot(243)
    Series(range(1,len(prs.hyperVolume)+1),prs.hyperVolume,'Hypervolume (Iteration %d)' % (i),'iteration','hypervolume')

    # Convergence of True Pareto Proportion
    ax = fig.add_subplot(244)
    Series(range(1,len(prs.trueParetoProportion)+1),prs.trueParetoProportion,'True Pareto Proportion (Iteration %d)' % (i),'iteration','true pareto proportion')

    #visitedSolutions
    fig.add_subplot(245)
    visitedPoints = tree.visitedPoints()
    Scatter2DColor(visitedPoints,'x',leafNodes,'All Visited Points (Iteration %d)' % (i),'x1','x2',True)
            
    #paretoSet
    fig.add_subplot(246)
    paretoSet = utils.identifyParetoSetParallel(tree)
    Scatter2DColor(paretoSet,'x',leafNodes,'Estimated Pareto Set (Iteration %d)' % (i),'x1','x2',True)
    
    #estimated Pareto front
    fig.add_subplot(247)
    Scatter2DColor(visitedPoints,'mean',leafNodes,'Estimated Pareto Front (Iteration %d)' % (i),'obj1','obj2',False)    
    
    # Convergence of Hausdorff Distance
    ax = fig.add_subplot(248)
    Series(range(1,len(prs.hausdorffDistance)+1),prs.hausdorffDistance,'Hausdorff Distance (Iteration %d)' % (i),'iteration','Hausdorff Distance')
    
    
    #save figures
    fig.set_size_inches(32, 16)
    fig.savefig(outputDir+'/animation-%d.png' % i, dpi=200, bbox_inches="tight")   
    fig.savefig(outputDir+'/eps-animation-%d.eps' % i, dpi=200, bbox_inches="tight")     
    plt.close()
    
    return 

def atoi(text):
    """convert text to integer if it's string of integer, otherwise do nothing
    Args:
        text: A string to be processed
    Returns:
        converted test string
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text) ]        
    
def generateAnimation(outputDir):
    """ generate GIF animation by combining all frames
    Args:
        outputDir: A string indicating the output directory path
    Returns:
        None
    """
    from PIL import Image
    import os
    
    #retrieve all the image frames' filename and sort them
    file_names = [fn for fn in os.listdir(outputDir+'/.') if fn.endswith('.png')]
    file_names.sort(key=natural_keys)

    #open those image frames
    images = [Image.open(outputDir+'/'+fn) for fn in file_names]
    
#    size = (150,150)
#    for im in images:
#        im.thumbnail(size, Image.ANTIALIAS)
   
    #generate GIF
    filename = "animation.gif"
    images2gif.writeGif(outputDir+'/'+filename, images, duration=0.5)   
    return 
    
def HVIteration(problemKey,CASE):
    """plot the convergence of hypervolume versus iteration
    Args:
        problemKey: A string representing the problem description
        CASE: A list indicating different cases (same problem but different algorithms)
    Returns:
        None
    """
    for case in CASE:
        prs = case.prs
        plt.plot(range(1,prs.currentIteration),prs.hyperVolume,label=prs.description)
    plt.title(problemKey)
    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    #plt.legend()
    return 

def HVComputationTime(problemKey,CASE):
    """plot the convergence of hypervolume versus computationTime
    Args:
        problemKey: A string representing the problem description
        CASE: A list indicating different cases (same problem but different algorithms)
    Returns:
        None
    """
    for case in CASE:
        prs = case.prs
        time = []
        for t in prs.endTime[:-1]:
            time.append((t-prs.startTime[1]).total_seconds())
        plt.plot(time,prs.hyperVolume,label=prs.description)
    plt.title(problemKey)
    plt.xlabel('Computation Time (s)')
    plt.ylabel('Hypervolume')
    #plt.legend()
    return 

def HVSampleSize(ax,problemKey,CASE):
    """plot the convergence of hypervolume versus sample size
    Args:
        problemKey: A string representing the problem description
        CASE: A list indicating different cases (same problem but different algorithms)
    Returns:
        None
    """
    for case in CASE:
        prs = case.prs
        plt.plot(prs.sampleSize,prs.hyperVolume,label=prs.description)
    plt.title(problemKey)
    plt.xlabel('Sample Size')
    plt.ylabel('Hypervolume')
    plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    return 
    
def HVAll(problemKey,CASE):
    """
    """
    fig = plt.figure()        
    # Hypervolume vs Iteration
    fig.add_subplot(131)  
    HVIteration(problemKey,CASE)
    # Hypervolume vs ComputationTime
    fig.add_subplot(132)  
    HVComputationTime(problemKey,CASE)
    # Hypervolume vs SampleSize
    ax=fig.add_subplot(133)  
    HVSampleSize(ax,problemKey,CASE)  
    
    #save figures
    fig.set_size_inches(40, 16)
    fig.savefig('output/HV-%s.png' % (problemKey), dpi=200)
    fig.savefig('output/HV-%s.eps' % (problemKey), dpi=200)     
    plt.close()      
    return 
    
def plotConvergence(problemKey, yName, y, title, bestY = np.nan, outputdir='output/'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    miny, maxy = np.inf, -np.inf
    N = len(y[list(y.keys())[0]]['ensemble'])
    x = list(range(1, N + 1))
    for key in y:
        markers_on = list(np.linspace(1, N - 1, 10))
        plt.plot(x, y[key]['ensemble'], color=y[key]['color'], ls='-', linewidth=1)        
        plt.plot(np.array(x)[markers_on], np.array(y[key]['ensemble'])[markers_on], 
                 color=y[key]['color'], marker=y[key]['marker'], 
                 ls='None', label=key)
        miny = min(miny, np.nanmin(y[key]['ensemble']))
        maxy = max(maxy, np.nanmax(y[key]['ensemble']))
    maxy = np.nanmax([maxy, bestY])
    miny = np.nanmin([miny, bestY])
    if not(np.isnan(bestY)):
        plt.plot(x,[bestY] * len(x), ls='-', color='k', linewidth=1, label='Best Value')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=2)
    plt.title(title)
    plt.xlabel('# of Evaluations')
    plt.ylabel(yName)
    plt.grid()
    ax.set_ylim([miny - 0.1 * (maxy - miny), maxy + 0.1 * (maxy - miny)])
    ax.set_xlim([1, len(x)])
    fig.set_size_inches(12, 8)
    fig.savefig(outputdir+'algo-comparison-%s-%s.png' % (problemKey, yName), dpi=200, bbox_inches="tight", additional_artist=[lgd]) 
    fig.savefig(outputdir+'algo-comparison-%s-%s.eps' % (problemKey, yName), dpi=200, bbox_inches="tight", additional_artist=[lgd]) 
    plt.close()
    return 
    

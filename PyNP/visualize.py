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
    ax.contourf(xv,yv,objs,zdir='y',cmap=cm.coolwarm)
    ax.contourf(xv,yv,objs,zdir='x1',cmap=cm.coolwarm)
    ax.contourf(xv,yv,objs,zdir='x2',cmap=cm.coolwarm)
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
    except:
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
    return 
        
    
def All(case):
    if case.problem.dim != 2:
        return 
    num = case.problem.num
    leafNodes = case.np.tree.root.leafNodes()
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
    visitedPoints = case.np.tree.visitedPoints()
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
    
def generateAnimationFrame(outputDir,i,np,trueParetoSet):
    tree = np.tree
    leafNodes = tree.root.leafNodes()
    xv = trueParetoSet['xv']
    yv = trueParetoSet['yv']
    isPareto = trueParetoSet['isPareto']
    dominationCount = trueParetoSet['dominationCount']
    fig = plt.figure()             
    
    # Domination Count Contour Plot
    ax = fig.add_subplot(231)
    Contour2D(fig,ax,xv,yv,dominationCount,leafNodes,'Domination Count (Iteration %d)' % (i),True)
    
#    # Global Pareto Set Contour Plot
#    ax = fig.add_subplot(232)
#    Contour2D(fig,ax,xv,yv,isPareto,leafNodes,'True Pareto Set (Iteration %d)' % (i),True)      
    
    #visitedSolutions
    fig.add_subplot(232)
    MPR = utils.identifyMPR(tree,np.rule.alphaPI)
    colorRegion(leafNodes,MPR,'blue','Convergence of MPR (Iteration %d)' % (i), 'x1','x2')    
    
    # Convergence of Hypervolume
    ax = fig.add_subplot(233)
    Series(range(1,len(np.hyperVolume)+1),np.hyperVolume,'Hypervolume (Iteration %d)' % (i),'iteration','hypervolume')

    #visitedSolutions
    fig.add_subplot(234)
    visitedPoints = tree.visitedPoints()
    Scatter2DColor(visitedPoints,'x',leafNodes,'All Visited Points (Iteration %d)' % (i),'x1','x2',True)
            
    #paretoSet
    fig.add_subplot(235)
    paretoSet = utils.identifyParetoSetParallel(tree)
    Scatter2DColor(paretoSet,'x',leafNodes,'Estimated Pareto Set (Iteration %d)' % (i),'x1','x2',True)
    
    #estimated Pareto front
    fig.add_subplot(236)
    Scatter2DColor(visitedPoints,'mean',leafNodes,'Estimated Pareto Front (Iteration %d)' % (i),'obj1','obj2',False)    
    
    #save figures
    fig.set_size_inches(24, 16)
    fig.savefig(outputDir+'/animation-%d.png' % i, dpi=200)   
#    fig.savefig(outputDir+'/eps-animation-%d.eps' % i, dpi=200)     
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
        np = case.np
        plt.plot(range(1,np.currentIteration),np.hyperVolume,label=np.description)
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
        np = case.np
        time = []
        for t in np.endTime[:-1]:
            time.append((t-np.startTime[1]).total_seconds())
        plt.plot(time,np.hyperVolume,label=np.description)
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
        np = case.np
        plt.plot(np.sampleSize,np.hyperVolume,label=np.description)
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
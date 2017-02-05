# -*- coding: utf-8 -*-
"""
    Created on Wed Mar 16 17:36:26 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

from .. import utils
import datetime

###BUDGET PERSPECTIVE###

def exceedMaximumIteration(context,args):
    """Exit the PRS algorithm if exceeding maximum iteration number
    Args:
        context: A class Context() recording the information of this case
    Returns:
        flag: A boolean value indicating whether the algorithim should stop
    """
    
    flag = context.currentIteration > args['maximumIteration']
    return flag
        
def exceedMaximumSampleSize(context,args):
    """Exit the PRS algorithm if exceeding maximum sample size
    Args:
        context: A class Context() recording the information of this case
    Returns:
        flag: A boolean value indicating whether the algorithim should stop        
    """    
    flag = context.currentSampleSize >= args['maximumSampleSize']
    return flag
    
def exceedMaximumComputationTime(context,args):
    """Exit the PRS algorithm if exceeding maximum computation time
    Args:
        context: A class Context() recording the information of this case
    Returns:
        flag: A boolean value indicating whether the algorithim should stop        
    """     
    now = datetime.datetime.now()
    currentComputationTime = (now - context.startTime[0]).total_seconds()
    flag = currentComputationTime >= args['maximumComputationTime']
    return flag

###PRECISION PERSPECTIVE####

def exceedMaximumTreeLevel(context,args):
    """Exit the PRS algorithm if exceeding maximum tree level
    Args:
        context: A class Context() recording the information of this case
    Returns:
        flag: A boolean value indicating whether the algorithim should stop        
    """
    flag = context.currentTreeLevel >= args['maximumTreeLevel']
    return flag
    
def exceedMPRMaximumTreeLevel(context,args):
    """Exit the PRS algorithm if minimum tree levels of MPR exceeds maximum tree level
    Args:
        context: A class Context() recording the information of this case
    Returns:
        flag: A boolean value indicating whether the algorithim should stop        
    """    
    MPR = utils.identifyMPR(context.tree,context.rule.alphaPI)        
    minimumMPRTreeLevel = min([leaf.level for leaf in MPR])
    flag = minimumMPRTreeLevel >= args['maximumTreeLevel'] 
    return flag    
    
def exceedPIThreshold(context,args):
    """Exit the PRS algorithm if satisfying the threshold of promising index
    Args:
        context: A class Context() recording the information of this case
    Returns:
        flag: A boolean value indicating whether the algorithim should stop        
    """
    flag = context.currentPI <= args['thresholdPI']
    return flag
    
def optimality(context, args):
    """Exit the PRS algorithm if all the Pareto solutions are found
    Args:
        context: A class Context() recording the information of this case
        args: A dictionary of necessary aditional arguments
    Returns:
        flag: A boolean value indicating whether the algorithim should stop
    """
    if context.trueParetoProportion:
        if context.trueParetoProportion[-1] < 1:
            flag = False
        else:
            flag = True
    else:
        flag = False
    return flag

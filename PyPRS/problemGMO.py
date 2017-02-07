#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:09:24 2017

@author: liuweizhi
"""

from PyGMO.problem import base
import numpy as np

class fon(base):
    """ test problem FON
    """
    def __init__(self, dim = 3):
        super(fon, self).__init__(dim, 0, 2)
        self.set_bounds(-4.0, 4.0)
        
    def _objfun_impl(self, x):
        try:
            f0 = 1 - np.exp(-sum((x-1/np.sqrt(len(x)))**2))
            f1 = 1 - np.exp(-sum((x+1/np.sqrt(len(x)))**2))
        except Exception as e:
            print(e)
        return (f0, f1, )
        
    def human_readable_extra(self):
        return "\n\tTest Problem FON"
        
class sch(base):
    """ test problem SCH
    """
    def __init__(self, dim = 1):
        super(sch, self).__init__(dim, 0, 2)
        self.set_bounds(-10.0**3, 10.0**3)
        
    def _objfun_impl(self, x):
        try:           
            f0 = x[0]**2
            f1 = (x[0]-2)**2
        except Exception as e:
            print(e)
        return (f0, f1, )
        
    def human_readable_extra(self):
        return "\n\tTest Problem SCH"
        
class zdt4(base):
    """ test problem ZDT4
    """
    def __init__(self, dim = 2):
        super(zdt4, self).__init__(dim, 0, 2)
        self.set_bounds(0.0, 1.0)
        
    def _objfun_impl(self, x):
        try:
            x = np.array(x)
            zdt4g = 1 + 10*(len(x)-1) + sum(x[1:]**2 - 10*np.cos(4*np.pi*x[1:]))
            f0 = x[0]
            f1 = zdt4g * (1-np.sqrt(x[0]/zdt4g))
        except Exception as e:
            print(e)
        return (f0, f1, )
        
    def human_readable_extra(self):
        return "\n\tTest Problem ZDT4"        
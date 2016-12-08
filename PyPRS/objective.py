# -*- coding: utf-8 -*-
"""
    Created on Mon Apr 11 09:20:35 2016
    @author: liuweizhi (greenwicher.com, weizhiliu2009@gmail.com)
    Copyright (C) 2016
    All rights reserved.
    GPL license.
"""

import numpy as np

circle2 = lambda x: sum((x-2)**2)
line45 = lambda x: x[0]+x[1]-1
circle = lambda x: (x[0]**2+x[1]**2-1)**2
Heart = lambda x: (x[0]**2+(x[1]-(x[0]**2)**(1/3))**2-1)**2
NB = lambda x: ((1+2*np.sqrt(-(abs(x[1])-1)**2+1)-x[0])*(x[0]**3+x[0]**2-2*x[0])*(x[1]+2*x[0]+2))**2
Face = lambda x: (((x[0]-1)**2+x[1]**2-4)*((x[0]+1)**2+x[1]**2-4)*(x[0]**2+(x[1]-np.sqrt(3))**2-4)-6>0)*(sum(x**2))

GoldsteinPrice = lambda x: (1+((x[0]+x[1]+1)**2)*(19-4*x[0]+3*(x[0]**2)-14*x[1]+6*x[0]*x[1]+3*(x[1]**2)))*(30+((2*x[0]-3*x[1])**2)*(18-32*x[0]+12*(x[0]**2)+48*x[1]-36*x[0]*x[1]+27*(x[1]**2)))
Rastrigin = lambda x: 40 + sum(x**2-10*np.cos(2*np.pi*x))

sch1 = lambda x: x**2
sch2 = lambda x: (x-2)**2

fon1 = lambda x: 1 - np.exp(-sum((x-1/np.sqrt(len(x)))**2))
fon2 = lambda x: 1 - np.exp(-sum((x+1/np.sqrt(len(x)))**2))

polA1 = 0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
polA2 = 1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
polB1 = lambda x: 0.5*np.sin(x[0]) - 2*np.cos(x[0]) + np.sin(x[1]) - 1.5*np.cos(x[1])
polB2 = lambda x: 1.5*np.sin(x[0]) - np.cos(x[0]) + 2*np.sin(x[1]) - 0.5*np.cos(x[1])
pol1 = lambda x: 1 + (polA1 - polB1(x))**2 + (polA2 - polB2(x))**2 
pol2 = lambda x: (x[0]+3)**2 + (x[1]+1)**2

kur1 = lambda x: sum(-10*np.exp(-0.2*np.sqrt(x[:-1]**2+x[1:]**2)))
kur2 = lambda x: sum(np.abs(x)**0.8 + 5*np.sin(x**3))

zdt1g = lambda x: 1 + 9*np.mean(x[1:])
zdt11 = lambda x: x[0]
zdt12 = lambda x: zdt1g(x) * (1 - np.sqrt(x[0]/zdt1g(x)))

zdt2g = lambda x: 1 + 9*np.mean(x[1:])
zdt21 = lambda x: x[0]
zdt22 = lambda x: zdt2g(x) * (1 - (x[0]/zdt2g(x))**2)

zdt3g = lambda x: 1 + 9*np.mean(x[1:])
zdt31 = lambda x: x[0]
zdt32 = lambda x: zdt3g(x) * (1 - np.sqrt(x[0]/zdt3g(x)) - x[0]/zdt3g(x)*np.sin(10*np.pi*x[0]))

zdt4g = lambda x: 1 + 10*(len(x)-1) + sum(x[1:]**2 - 10*np.cos(4*np.pi*x[1:]))
zdt41 = lambda x: x[0]
zdt42 = lambda x: zdt4g(x) * (1-np.sqrt(x[0]/zdt4g(x)))

zdt6g = lambda x: 1 + 9*(np.mean(x[1:])**0.25)
zdt61 = lambda x: 1 - np.exp(-4*x[0])*(np.sin(6*np.pi*x[0])**6)
zdt62 = lambda x: zdt6g(x) * (1 - (zdt61(x)/zdt6g(x))**2)

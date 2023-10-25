#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benjamintagg

QSTIMB: Q-matrix and Stochastic simulation based Ion Channel Model Builder

Department of Neuroscience, Physiology, and Pharmacology, UCL, UK
--------------------------------LICENSE-------------------------
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"),to deal in
the Software without restriction, including without limitation the rights to
use,copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject tothe following conditions:
The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT,TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contains various methods for simulating ion channels

The focus is on testing performance of methods for models of varying complexity.

Within, exists code for:
    
    [1]----- Models -----
    ==================
    - in-built models
        - which illustrate the expected structure of models created (including transition, and Q matrices)
            - respectively to be used in stochastic and Q matrix (CME) simulations
        - Some models used were previously implemented together in Harveit and Veruki (2006),
            which are labelled using their codes.
        - Additional models from: Coombs et al., 2017
        
    [2]--- Stochastic Simulation Methods----
    ====================================
    - Classical Gillespie Walk
        - this is very slow and is not recommended, but the base code is provided for interest.
        
    -- [Recommended for large number of states and large number of concentration-dependent states]
    - Fixed Tau (interval size) Tau-leaping Gillespie Walk
        - for relaxations of ion channels
        - for realistic fast agonist applications 
            (method of discretising concentrations credited to AP Plested)
    
    -- [Recommended Method For simpler models] --
    - Adaptive Tau leaping Gillespie Walk, altered to work with ion channels
        - for relaxations
        - for realistic fast agonist applications
            (method of discretising concentrations credited to AP Plested)
            
    [NB] Relaxations (i.e. the non-agonist application procedures) are much faster
    and should be used in first instance for most cases.
    
    [3]---- CME Simulation Methods ----
    ================================
    - Q matrix method for relaxations
    - Q matric method for agonist applications
    
    [4]---- Master Simultaion Runner ----
    - Simulate function
    - File handling functions (saving and loading)
    
    [5]------ Built-in functions -----
    ================================
    - For graphing transition or Q matrix
    - For adding gaussian noise
    - Methods for enforcing microscopic reversibility onto Q matrix (CME)
    - Realistic concentration jumps (stochastic and CME) (Credit AP Plested) 
    - Functions associated with Q matrix method calculations (CME)
    
    [6]------ Unused (working) code for model fitting -----
    - Firefly algorithm
    - 'Fittigration' method - an avolutionary algorithm
"""

# =============================================================================
# Imports
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import interp1d
import copy
import networkx as nx # for visualising transition matrices and Q, for microscopic reversibility
import random
import pickle # for saving simulate outputs
import pandas as pd # for converting simulate outputs
from tqdm import tqdm # progress bar for simulate
from itertools import combinations

# config
import matplotlib.colors as mcolors
from cycler import cycler
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colorlist = ['black','dimgrey','teal','darkturquoise', 'midnightblue','lightskyblue','steelblue','royalblue','lightsteelblue','darkorange', 'orange','darkgoldenrod','goldenrod','gold','khaki','yellow']
mycycle = cycler(color=[colors[item] for item in colorlist])

# import other libraries of module
import QSTIMB_poisson
# =============================================================================
# [1] In-built Models for testing: Q (transition matrix) construction and modification
# =============================================================================
# all models to be implemented as dicts containing:
    # rates as type np.ndarray
    # conducting states, where key = state num, and value = conductance in PS
    # voltage-dependent rates: where the transition rate itself is a function of voltage - not in reference to driving force
    # concentration-dependent rates
    # (deprecated) propensities for selection of transition
    # (deprecated) scale parameters for selection of dwell time
    
# nb, any non-zero concentration can be used to generate Q for use in 
    # an agonist application
# but the concentration should be the desired concentration for a relaxation in
    # constant concentration of agonist
    
# note that Q['rates] is a transition matrix. It differs from
# Q matrix (in Q['Q'])by assigment of diags, where
    # by convention Q[ii] = - Sum_(jES, j!=i) *  (q(ij). i.e. Q[i,i] = - sum(Q[i,1:n])
# In Q['rates], Q[ii] = 0
# In Q['Q'], Q[ii] = - Sum_(jES, j!=i) *  (q(ij). i.e. Q[i,i] = - sum(Q[i,1:n])
    
###---- Models from Harveit and Veruki, 2006 #---
# use their codes

# three state model with some of their rates (model 30)

def threeS(agonist_conc =5*(10**-3)):
    """
   Classical three state ion channel receptor model
   using constants from Harveit and Veruki (2006 - their Smod34)
   
   - [0] Unbound, closed state
   - [1] Bound, closed state
   - [2] open state
   
   
   such that:
       [0]--[1]--[2]
   
    """
    Q = {}
    tr = np.zeros([3,3]) # for 3 states
    tr[0,1] = (6*10**6)*agonist_conc
    tr[1,0] = 100
    tr[1,2] = 1000
    tr[2,1] = 750
    tr[tr==0] = np.nan
    Q.update({'rates':tr}) # transition matrix: used by stochastic methods
    Q.update({'conc':agonist_conc})
    Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
    Q['conducting states'].update({2:50*10**-12}) # state 3 is open with conductance 50 pS
    Q.update({'conc-dep':{}}) # concentration-dependent rates
    Q['conc-dep'].update({(0,1)})
    Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
    Q.update({'initial states':{}})
    Q['initial states'].update({0:1}) # initialise in state 0 with probability of 1
    q = np.copy(tr)
    for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
        q[row,row] = - np.nansum(q[row])
    Q.update({'Q':q})
    Q['Q'][~np.isfinite(Q['Q'])]=0
    return(Q)

# GlyAQ
def GlyAG(gly_conc=5*(10**-3)):
    """
    
    For Glycine conc in M (5mM used as default)
    
    
    Six state Glycine model from Harveit and Veruki (2006) with following states:
        - [0] Unbound, closed state
        - [1] closed, 1 Glycine molecule bound
        - [2] closed, 2 Glycine molecules bound
        - [3] open (2 glycine molecules bound)
        - [4] Fast desensitised state
        - [5] Slow desensitised state
        
    Such that:
        [0]--[1]--[2]---[3]
                   |
                  [4]
                   |
                  [5]
    """
    Q = {}
    tr = np.zeros([6,6]) # for 6 states
    tr[0,1] = (10*(10**6)) * gly_conc
    tr[1,0] = 100
    tr[1,2] = (5*(10**6)) * gly_conc
    tr[2,1] = 200
    tr[2,3] = 1700
    tr[3,2] = 750
    tr[2,4] = 60
    tr[4,2] = 10
    tr[4,5] = 1
    tr[5,4] = 1
    tr[tr==0] = np.nan
    Q.update({'rates':tr})
    Q.update({'conc':gly_conc})
    Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
    Q['conducting states'].update({3:50*10**-12}) # state 3 is open with conductance 50 pS
    Q.update({'conc-dep':{}}) # concentration-dependent rates
    Q['conc-dep'].update({(0,1)})
    Q['conc-dep'].update({(1,2)})
    Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
    Q.update({'initial states':{}})
    Q['initial states'].update({0:1}) # initialise in state 0 with probability of 1
    # make Q matrices
    q = np.copy(tr)
    for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
        q[row,row] = - np.nansum(q[row])
    Q.update({'Q':q})
    Q['Q'][~np.isfinite(Q['Q'])]=0
    return(Q)

# GlyLeg98 (Legendre,98; Harveit and Veruki,2006)

def GlyLeg98(gly_conc = 5*10**-3):
    """
    Legendre (1998) Model for Glycine receptors in Zebrafish hindbrain, as cited by
    Harveit and Veruki, 2006
    
    [0] Unbound, closed State
    [1] closed, 1 Glycine molecule bound
    [2] closed, 2 Glycine molecules bound
    [3] Open state 1
    [4] Reluctant state
    [5] Open state 2
    
    Such that:
            [0]--[1]--[2]---[3]
                       |
                      [4]
                       |
                      [5]
    """
    Q = {}
    tr = np.zeros([6,6]) # for 6 states
    tr[0,1] = (12*(10**6)) * gly_conc
    tr[1,0] = 1452
    tr[1,2] = (6*(10**6)) * gly_conc
    tr[2,1] = 2904
    tr[2,3] = 8938
    tr[3,2] = 680
    tr[2,4] = 536
    tr[4,2] = 136
    tr[4,5] = 3180
    tr[5,4] = 1300
    tr[tr==0] = np.nan
    Q.update({'rates':tr})
    Q.update({'conc':gly_conc})
    Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
    Q['conducting states'].update({3:50*10**-12}) # state 3 is open with conductance 50 pS
    Q['conducting states'].update({5:50*10**-12}) # state 3 is open with conductance 50 pS
    Q.update({'conc-dep':{}}) # concentration-dependent rates
    Q['conc-dep'].update({(0,1)})
    Q['conc-dep'].update({(1,2)})
    Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
    Q.update({'initial states':{}})
    Q['initial states'].update({0:1}) # initialise in state 0 with probability of 1
    q = np.copy(tr)
    for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
        q[row,row] = - np.nansum(q[row])
    Q.update({'Q':q})
    Q['Q'][~np.isfinite(Q['Q'])]=0
    return(Q)

# GlyBur04 (Burzomato,2004; Harveit and Veruki,2006)
# def GlyBur04Q(gly_conc = 5*10**-3):
#     """
#     Currently doesn't work - despite following outlined rate constants.
    
    
#     Burzomato (2004) Model for rat heteromeric alpha1beta Glycine receptors, as cited by
#     Harveit and Veruki, 2006
    
#     [0] Unbound, closed state
#     [1] closed, 1 Glycine molecule bound
#     [2] closed, 2 Glycine molecules bound
#     [3] closed, 3 Glycine molecules bound
#     [4] pre-open state 1 Glycine molecule bound
#     [5] pre-open state, 2 Glycine molecules bound
#     [6] pre-open state, 3 Glycine molecules bound
#     [7] open, 1 Glycine molecule bound
#     [8] open, 2 Glycine molecules bound
#     [9] open, 3 Glycine molecules bound
    
#     Such that
#             [0]--[1]--[2]---[3]
#                   |    |     |
#                  [4]--[5]---[6]
#                   |    |     |
#                  [7]  [8]   [9]
#     """
#     Q = {}
#     tr = np.zeros([10,10]) # for 6 states
#     k_ = (0.59*10**-6) * gly_conc
#     k_m = 302
#     gamma_1 = 29266
#     delta_1 = 180
#     gamma_2 = 18000
#     delta_2 = 6800
#     gamma_3 = 948.1
#     delta_3 = 22000
#     kf_ = (150*10**-6)*gly_conc
#     kf_m = 1250
#     alpha_1 = 3400
#     beta_1 = 4200
#     alpha_2 = 2100
#     beta_2 = 28000
#     alpha_3 = 6700
#     beta_3 = 130000
#     tr[0,1] = 3*k_#
#     tr[1,0] = k_m
#     tr[1,2] = 2*k_#
#     tr[2,1] = 2*k_m
#     tr[2,3] = k_#
#     tr[3,2] = 3*k_m
#     tr[1,4] = delta_1
#     tr[4,1] = gamma_1
#     tr[2,5] = delta_2
#     tr[5,2] = gamma_2
#     tr[3,6] = delta_3
#     tr[6,3] = gamma_3
#     tr[4,5] = 2*kf_#
#     tr[5,4] = 2*kf_m
#     tr[5,6] = kf_#
#     tr[6,5] = 3* kf_m
#     tr[4,7] = beta_1
#     tr[7,4] = alpha_1
#     tr[5,8] = beta_2
#     tr[8,5] = alpha_2
#     tr[6,9] = beta_3
#     tr[9,6] = alpha_3
#     tr[tr==0] = np.nan
#     Q.update({'rates':tr})
#     Q.update({'conc':gly_conc})
#     Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
#     Q['conducting states'].update({7:50*10**-12}) # state 3 is open with conductance 50 pS
#     Q['conducting states'].update({8:50*10**-12}) # state 3 is open with conductance 50 pS
#     Q['conducting states'].update({9:50*10**-12}) # state 3 is open with conductance 50 pS
#     Q.update({'conc-dep':{}}) # concentration-dependent rates
#     Q['conc-dep'].update({(0,1)})
#     Q['conc-dep'].update({(1,2)})
#     Q['conc-dep'].update({(2,3)})
#     Q['conc-dep'].update({(4,5)})
#     Q['conc-dep'].update({(5,6)})
#     Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
#     Q.update({'initial states':{}})
#     Q['initial states'].update({0:1}) # initialise in state 0 with probability of 1
    # q = np.copy(tr)
    # for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
    #     q[row,row] = - np.nansum(q[row])
    # Q.update({'Q':q})
    #Q['Q'][~np.isfinite(Q['Q'])]=0
#     return(Q)


#Glugei99
def GluGei99(glu_conc=5*10**-3):
    """
    Geiger, 1999 Model for hippocampal, interneuronal AMPARs as cited by
    Harveit and Veruki, 2006
    
    [0] Unbound, closed state
    [1] closed, 1 glutamate molecule bound
    [2] closed, 2 glutamate molecules bound
    [3] open state
    [4] desensitised state, 1 glutamate molecule bound
    [5] desensitised state, 2 glutamate molecules bound
    [6] desensitised state,2 glutamate molecules bound

    Such that:
            [0]--[1]--[2]---[3]
                  |    |     |     
                 [4]--[5]---[6]
    """
    Q = {}
    tr = np.zeros([7,7]) 
    tr[0,1] = (17.1*10**6)*glu_conc
    tr[1,0] = 157
    tr[1,2] = (3.24*10**6)*glu_conc
    tr[2,1] = 3.76*10**3
    tr[2,3] = 14.9*10**3
    tr[3,2] = 4*10**3
    tr[1,4] = 1.53*10**3
    tr[4,1] = 408
    tr[2,5] = 502
    tr[5,2] = 0.377
    tr[3,6] = 121
    tr[6,3] = 191
    tr[4,5] = (0.611*10**6)*glu_conc
    tr[5,4] = 2
    tr[5,6] = 1.59*10**3
    tr[6,5] = 899
    tr[tr==0] = np.nan
    Q.update({'rates':tr})
    Q.update({'conc':glu_conc})
    Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
    Q['conducting states'].update({3:8.5*10**-12}) # state 3 is open with conductance 8.5 pS
    Q.update({'conc-dep':{}}) # concentration-dependent rates
    Q['conc-dep'].update({(0,1)})
    Q['conc-dep'].update({(1,2)})
    Q['conc-dep'].update({(5,6)})
    Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
    Q.update({'initial states':{}})
    Q['initial states'].update({0:1}) # initialise in state 0 with probability of 1
    q = np.copy(tr)
    for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
        q[row,row] = - np.nansum(q[row])
    Q.update({'Q':q})
    Q['Q'][~np.isfinite(Q['Q'])]=0
    return(Q)

#GluMom03
def GluMom03(glu_conc = 5*10**-3):
    """
    Momiyama,2003 Model for Purkinje AMPARs as cited by
    Harveit and Veruki, 2006
    
    [0] Unbound, closed state
    [1] closed, 1 glutamate molecule bound
    [2] closed, 2 glutamate molecules bound
    [3] open state
    [4] desensitised state
    [5] desensitised state
    [6] desensitised state
    [7] desensitised state
    [8] desensitised state
    Such that:
            [0]--[1]--[2]---[3]---[4]
                  |    |     |     |
                 [5]--[6]---[7]---[8]
    """
    Q = {}
    tr = np.zeros([9,9]) # for 6 states
    tr[0,1] = (13.66*10**6)*glu_conc
    tr[1,0] = 2093
    tr[1,2] = (6.019*10**6)*glu_conc
    tr[2,1] = 4.719*10**3
    tr[2,3] = 17.23*10**3
    tr[3,2] = 3.74*10**3
    tr[3,4] = 114.1
    tr[4,3] = 90.47
    tr[1,5] = 4.219*10**2
    tr[5,1] = 31.15
    tr[2,6] = 855.3
    tr[6,2] = 46.65
    tr[3,7] = 3.108
    tr[7,3] = 0.6912
    tr[4,8] = 18.78
    tr[8,4] = 0.3242
    tr[5,6] = (6.019*10**6)*glu_conc
    tr[6,5] = 3.486*10**3
    tr[6,7] = 476.4
    tr[7,6] = 420.9
    tr[7,8] = 1.034*10**4
    tr[8,7] = 636.3
    tr[tr==0] = np.nan
    Q.update({'rates':tr})
    Q.update({'conc':glu_conc})
    Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
    Q['conducting states'].update({3:5*10**-12}) # state 3 is open with conductance 5 pS
    Q.update({'conc-dep':{}}) # concentration-dependent rates
    Q['conc-dep'].update({(0,1)})
    Q['conc-dep'].update({(1,2)})
    Q['conc-dep'].update({(5,6)})
    Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
    Q.update({'initial states':{}})
    Q['initial states'].update({0:1}) # initialise in state 0 with probability of 1
    q = np.copy(tr)
    for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
        q[row,row] = - np.nansum(q[row])
    Q.update({'Q':q})
    Q['Q'][~np.isfinite(Q['Q'])]=0
    return(Q)

#GlyRH03
def GluRH03(glu_conc = 5*10**-3):
    """Robert and Howe (2003) Model for AMPARs, as cited by Harveit and Veruki, 2006
    
    [0] Open state: subconductance level 1
    [1] Open state: subconductance level 2
    [2] Open state: subconductance level 3
    [3] Unbound, closed state
    [4] Closed, 1 glutamate molecule bound
    [5] Closed, 2 glutamate molecules bound
    [6] Closed, 3 glutamate molecules bound
    [7] Closed, 4 glutamate molecules bound
    [8] Desensitised state
    [9] Desensitised state
    [10] Desensitised state
    [11] Desensitised state
    [12] Desensitised state
    [13] Deep desensitised state
    [14] Deep desensitised state
    [15] Deep desensitised state
    
    Such that:
                      [0]   [1]   [2]
                       |     |     |
            [3]--[4]--[5]---[6]---[7]
             |    |    |     |     |
            [8]--[9]--[10]--[11]--[12]
                       |     |     |
                     [13]--[14]--[15]            
    
    """
    Q = {}
    tr = np.zeros([16,16]) # for 6 states
    alpha = 8000
    beta = 20000
    k1 = (10**7)* glu_conc
    k_m1 = 10**4
    k2 = (10**3)* glu_conc
    k_m2 = 1
    delta_0 = 3.5*10**-3
    gamma_0 = 6
    delta_1 = 800
    gamma_1 = 45
    delta_2 = 4000
    gamma_2 = 220
    tr[5,0],tr[0,5],tr[6,1],tr[1,6],tr[7,2],tr[2,7] = 2*beta,alpha,3*beta,alpha,4*beta,alpha
    tr[3,4],tr[4,5],tr[5,6],tr[6,7],tr[4,3],tr[5,4],tr[6,5],tr[7,6] = 4*k1,3*k1,2*k2,k1,k_m1,2*k_m1,3*k_m1,4*k_m1
    tr[3,8],tr[4,9],tr[5,10],tr[6,11],tr[7,12] = 4*delta_0,delta_1,2*delta_1,3*delta_1,4*delta_1
    tr[8,3],tr[9,4],tr[10,5],tr[11,6],tr[12,7] = gamma_0,gamma_1,gamma_1,gamma_1,gamma_1
    tr[8,9],tr[9,10],tr[10,11],tr[11,12],tr[9,8],tr[10,9],tr[11,10],tr[12,11] = 3*k1, 3*k1,2*k1,k1,k_m2,k_m1,2*k_m1,3*k_m1
    tr[10,13],tr[11,14],tr[12,15],tr[13,10],tr[14,11],tr[15,12] = delta_2,2*delta_2,3*delta_2,gamma_2,gamma_2,gamma_2
    tr[13,14],tr[14,15],tr[14,13],tr[15,14] = 2*k1, k1,k_m1,2*k_m1
    tr[tr==0] = np.nan
    Q.update({'rates':tr})
    Q.update({'conc':glu_conc})
    Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
    Q['conducting states'].update({0:8*10**-12}) # state 3 is open with conductance 5 pS
    Q['conducting states'].update({1:16*10**-12}) # state 3 is open with conductance 5 pS
    Q['conducting states'].update({2:24*10**-12}) # state 3 is open with conductance 5 pS
    Q.update({'conc-dep':{}}) # concentration-dependent rates
    Q['conc-dep'].update({(3,4)})
    Q['conc-dep'].update({(4,5)})
    Q['conc-dep'].update({(5,6)})
    Q['conc-dep'].update({(6,7)})
    Q['conc-dep'].update({(8,9)})
    Q['conc-dep'].update({(9,10)})
    Q['conc-dep'].update({(10,11)})
    Q['conc-dep'].update({(11,12)})
    Q['conc-dep'].update({(13,14)})
    Q['conc-dep'].update({(14,15)})
    Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
    Q.update({'initial states':{}})
    Q['initial states'].update({3:1}) # initialise in state 3 with probability 1
    q = np.copy(tr)
    for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
        q[row,row] = - np.nansum(q[row])
    Q.update({'Q':q})
    Q['Q'][~np.isfinite(Q['Q'])]=0
    return(Q)
    
###---- Other models -----#
#=========================#

def GluCoo17(glu_conc = 10*10**-3):
    """
    Coombs et al., 2017 model for GluA1 homomers saturated with TARP gamma-2
    
    Also note that entry to desensitisation and recovery from desensitisation 
    are occupancy-depende, whereas in this model, entry to is concentration-dependent
    (not occupancy-dependent), and recovery is dimensionless.
    
    Nb see example of the relationships matrix
    
    [0] Open state: subconductance level 1
    [1] Open state: subconductance level 2
    [2] Open state: subconductance level 3
    [3] Open state: subconductance level 4
    [4] Unbound, closed state
    [5] Closed, 1 glutamate molecule bound
    [6] Closed, 2 glutamate molecules bound
    [7] Closed, 3 glutamate molecules bound
    [8] Closed, 4 glutamate molecules bound
    [9] Desensitised state
    [10] Desensitised state
    [11] Desensitised state
    [12] Desensitised state
    [13] Desensitised state
    [14] Deep desensitised state
    [15] Deep desensitised state
    [16] Deep desensitised state
    
    such that:
        
                 [0]   [1]   [2]    [3]
                  |     |     |      |
            [4]--[5]---[6]---[7]----[8]
             |    |     |     |     |
            [9]--[10]--[11]--[12]--[13]
                        |     |     |
                       [14]--[15]--[16]            

    """
    
    Q= np.zeros([17,17])

    k_on = (1.3*(10)**7)*glu_conc
    k_off = 3000 # s^-1
    alpha = 1000 # s^-1
    beta = 6000 # s^-1
    gamma = 16 # s^-1
    gamma_0 = 4.4 # s^-1
    delta_0 = 0.48 # s^-1
    delta_1 = 1200
    delta_2 = 1300
    delta_3 = 250# s^-1
    k_off_2 = 63# s^-1
    k_off_3 = 630# s^-1
    gamma_2 = 3900# s^-1

    # indexing rates for gating: occupied (R0-4) to open (O1-4) transitions, in forward and reverse directions
    Q[5,0],Q[6,1],Q[7,2],Q[8,3],Q[0,5],Q[1,6],Q[2,7],Q[3,8] = beta,(2*beta),(2*beta),(4*beta),alpha,alpha,alpha,alpha
    # indexing rates for ligand binding and unbinding (R0-R4)
    Q[4,5],Q[5,6],Q[6,7],Q[7,8],Q[5,4],Q[6,5],Q[7,6],Q[8,7] = (4*k_on),(3*k_on),(2*k_on),k_on,k_off,(2*k_off),(3*k_off),(4*k_off)
    # indexing rates for transition to and from desensitised states (D1-4)
    Q[4,9],Q[5,10],Q[6,11],Q[7,12],Q[8,13],Q[9,4],Q[10,5],Q[11,6],Q[12,7],Q[13,8] = 4*delta_0,delta_3,(2*delta_1),(3*delta_1),(4*delta_1),gamma_0,gamma,gamma,gamma,gamma
    # indexing rates for transitions between desensitised states
    Q[9,10],Q[10,11],Q[11,12],Q[12,13],Q[10,9],Q[11,10],Q[12,11],Q[13,12] = (3*k_on),(3*k_on),(2*k_on),k_on,k_off_2,k_off_3,(2*k_off),(3*k_off)
    # indexing rates for transitions to deep desensitised states
    Q[11,14],Q[12,15],Q[13,16],Q[14,11],Q[15,12],Q[16,13] = delta_2,(2*delta_2),(3*delta_2),gamma_2,gamma_2,gamma_2
    # indexing rates for transitions between deep desensitised states
    Q[14,15],Q[15,16],Q[15,14],Q[16,15] = (2*k_on),k_on,k_off,(2*k_off)
    tr = copy.deepcopy(Q)
    tr[tr==0] = np.nan
    Q = {}
    Q.update({'rates':tr})
    Q.update({'conc':glu_conc})
    Q.update({'conducting states':{}}) # allows defiance of convention that lowest numbered states are open
    Q['conducting states'].update({0:3.7*10**-12}) # state 3 is open with conductance 5 pS
    Q['conducting states'].update({1:16.1*10**-12}) # state 3 is open with conductance 5 pS
    Q['conducting states'].update({2:30.6*10**-12}) # state 3 is open with conductance 5 pS
    Q['conducting states'].update({3:38.6*10**-12}) # state 3 is open with conductance 5 pS
    Q.update({'conc-dep':{}}) # concentration-dependent rates
    Q['conc-dep'].update({(4,5)})
    Q['conc-dep'].update({(5,6)})
    Q['conc-dep'].update({(6,7)})
    Q['conc-dep'].update({(7,8)})
    Q['conc-dep'].update({(9,10)})
    Q['conc-dep'].update({(10,11)})
    Q['conc-dep'].update({(11,12)})
    Q['conc-dep'].update({(12,13)})
    Q['conc-dep'].update({(14,15)})
    Q['conc-dep'].update({(15,16)})
    Q.update({'voltage-dep':{}}) # voltage-dependent rates (none here)
    Q.update({'initial states':{}})
    Q['initial states'].update({4:1}) # initialise in state 3 with probability 1
    q = np.copy(tr)
    for row in range(0,np.size(q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
        q[row,row] = - np.nansum(q[row])
    Q.update({'Q':q})
    Q['Q'][~np.isfinite(Q['Q'])]=0
    #### example of relationship constraint matrices
    kk_0 = np.zeros_like(q) # contains one hot encodings to identify relaitonship existence
    kk_1 = np.ones_like(q) # contains mutliple rules for the relationships
    kk_0[5,0],kk_0[6,1],kk_0[7,2],kk_0[8,3] = 469,469,469,469 # beta one-hot encoding
    kk_1[5,0],kk_1[6,1],kk_1[7,2],kk_1[8,3] = 1,2,2,4 # beta multiples
    kk_0[0,5],kk_0[1,6],kk_0[2,7],kk_0[3,8] = 323,323,323,323 # alpha
    kk_1[0,5],kk_1[1,6],kk_1[2,7],kk_1[3,8] = 1,1,1,1
    kk_0[4,5],kk_0[5,6],kk_0[6,7],kk_0[7,8],kk_0[9,10],kk_0[10,11],kk_0[11,12],kk_0[12,13]  = 121,121,121,121,121,121,121,121# k_on encoding
    kk_1[4,5],kk_1[5,6],kk_1[6,7],kk_1[7,8],kk_1[9,10],kk_1[10,11],kk_1[11,12],kk_1[12,13] = 4,3,2,1,3,3,2,1 # multiples k_on
    kk_0[5,4],kk_0[6,5],kk_0[7,6],kk_0[8,7],kk_0[12,11],kk_0[13,12],kk_0[15,14],kk_0[16,15] = 768,768,768,768,768,768,768,768 # k_off encoding
    kk_1[5,4],kk_1[6,5],kk_1[7,6],kk_1[8,7],kk_1[12,11],kk_1[13,12],kk_1[15,14],kk_1[16,15] = 1,2,3,4,2,3,1,2 # k_off multiples
    kk_0[6,11],kk_0[7,12],kk_0[8,13] = 567,567,567 # delta_1
    kk_1[6,11],kk_1[7,12],kk_1[8,13] = 2,3,4 # delta_1
    kk_0[11,14],kk_0[12,15],kk_0[13,16] = 987,987,987 # delta_2
    kk_1[11,14],kk_1[12,15],kk_1[13,16] = 1,2,3 # delta_2
    kk_0[10,5],kk_0[11,6],kk_0[12,7],kk_0[13,8] = 434,434,434,434 # gamma
    kk_1[10,5],kk_1[11,6],kk_1[12,7],kk_1[13,8] = 1,1,1,1 
    kk = np.stack((kk_0,kk_1),axis=2)
    Q.update({'relationships':kk})

    return(Q)

# =============================================================================
# [2] Stochastic Simulation Methods
# =============================================================================
# =============================================================================
# Gillespie Method - probably best not to use. See recommended.
    # slow:
        # takes 19ms-586ms for 100ms simulation (0.1s) of GlyQ
        # unfeasibly longer for 200ms
# =============================================================================
# Coded here for N =1
def Gillespie_walk(Q,t_final):
    """
    Takes Q dict of type generated for in-built models for testing 
    
    that contains information about initial states, conducting states, transition rates, transition propensities etc
    
    Performs Gillespie walk for N = 1 receptor. Very slow with multiple states
    """
    # initialise at  t=0
    print("Warning! This is very slow.")
    t =[]
    states = []
    t.append(0)
    states.append([int(item) for item in Q['initial states'].keys()][0])
    # for first transition from initial states
    # draw random exponential with scale = 1/rates leaving state i
    dwells = np.random.exponential(scale = Q['scales'][[int(item) for item in Q['initial states'].keys()]],size = np.size(Q['initial states'].keys()))
    t.append(dwells[0])
    # 
    transition_draws = np.random.uniform(0,1,size = np.size(Q['initial states'].keys())) # random uniform draw with scale 1
    which_transition = np.where(Q['propensities'][[int(item) for item in Q['initial states'].keys()]]>transition_draws)
    if np.size(which_transition[0])==0:
        which_transition = [int(item) for item in Q['initial states'].keys()] # original state
    else:
        which_transition = which_transition[1]
    states.append(which_transition[0])
    while t[-1]<t_final:
        dwells = np.random.exponential(scale = Q['scales'][states[-1]],size = np.size(states[-1]))
        t.append(dwells[0])
        transition_draws = np.random.uniform(0,1,np.size(states[-1])) # random uniform draw with scale 1
        try:
            which_transition = np.argmin(Q['propensities'][states[-1]][Q['propensities'][states[-1]]>transition_draws]) # which transition > draw
        except Exception:
            which_transition = states[-1] # ex eption when cnanot do argmin
            # assumes N = 1
        states.append(which_transition)
    return(t,states)
# the problem is that the transitions are fast compared to usual reactions used for Gillespie
# and multiple transitions slow down
    
# =============================================================================
# Tau leap Gillespie:
    # classic fixed interval
    # adaptive tau
# =============================================================================
# =============================================================================
# binomial trial-based Tau leaping
# =============================================================================
def Tau_leap_Gillespie(N,Q,t_final,interval = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True):
    """
    Classical method for Gillespie fixed Tau (interval) leaping - please read all notes
    
    Tracks trajectories for N receptors during relaxations using a single Q matrix
    that holds transitions rates (s^-1) as i,j for transition from i to j.
    
    The relaxation is repeated for the number of times in iterations. Occupancies are
    averaged, and the current is plotted and returned if a conducitng state is specified in the model
    using the average occupancies. Thus, for a single trajectory, set iterations = 1.
    
    If conductance is not specified, mean occupacnies are returned.
    
    The size of tau (the interval) is fixed (Defeault = 5e-05 = 1/20000 = 20kHz),
    and may require some trial-and-error.
    An alternative would be to perform adaptive Tau leaping - see Weighted_agonist_application_Tau_leap
    A value for a reasonable fixed interval can be estimated by 1/np.nanmax(Q['rates']) -
    i.e.
    a timestep selected is such that only one of even the fastest reaction could occur per timestep.
    This can lead to 'ringing' of open states in relaxations, so some optimisation is required.
    
    Performance is related to the number of iterations, the size of the interval (i.e. number of steps).
    Models with large numbers of states will run slower, but unlike adaptive Tau,
    having many concentration-dependent states does not majorly alter runtime.
        For jumps from 0 concentration to some other concentration,
        Weighted adaptive Tau leaping may be faster unless the model is complex.
            e.g. Weighted is faster for thrresQ and GlyAG
    """
    # number of excepted occrs = 
    # draw poisson as np.random.poisson(propensities*interval)    
    
    # times
    t = np.arange(0,t_final,interval)
        
    # pre-allocate to store occupancy 
    occupancy = np.zeros([np.size(Q['rates'],1),np.size(t)])
    Rates = np.copy(Q['rates'])
    Rates[np.isnan(Rates)]=0
        
    #take initial states for N with probability given in Q['intial states']
    initial_states = [int(i) for i in Q['initial states'].keys()]
    initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
    initial_occupancy = np.zeros(np.size(Q['rates'],axis=0))
    for item, value in enumerate(initial_states):
        initial_occupancy[value] = initial_probabilities[item]
    #populate t = 0 with initial states
    occupancy[:,0] = initial_occupancy
    
    # initialise transition matrices
    transitions = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0)])
    compar_accumulator = np.zeros([np.size(Q['rates'],0)])
    
    iteration_occupancy = np.zeros([np.size(occupancy,0),np.size(occupancy,1),iterations])

    for iteration in np.arange(iterations):
        for intervalcount, timet in enumerate(t[:-1]):
            #for each state populated by receptors at intervalcount
            # = N to draw for exit that state
            prevstates = occupancy[:,intervalcount]
    
            for statefrom, N in enumerate(prevstates):
                for stateto in np.arange(np.size(Q['rates'],1)):
                    if Rates[statefrom,stateto]>0:
                        compar_accumulator[stateto] = np.random.binomial(N,(Rates[statefrom,stateto]*interval)/(np.nansum(Rates[statefrom,:],0)*interval))
                    else:
                        compar_accumulator[stateto] = 0
                    # because can generate negative or excessive populations of receptors,
                    # rescale draws such that sum(draws) = N
                    # and round to nearest integer -- make sure still not generating excess
                compar_accumulator = np.floor(np.divide((np.clip(compar_accumulator,0,N)),(np.sum(np.clip(compar_accumulator,0,N))),where = compar_accumulator>0)*N)
                transitions[statefrom,:] = compar_accumulator
                
            # subtract transitions to sum(row) from prevstates elementwise
            # add to new states sum(col)
            newstates = prevstates - np.nansum(transitions,axis=1)
            newstates = newstates + np.nansum(transitions,axis=0)
            
            #update next occupancy
            occupancy[:,intervalcount+1] = newstates
        iteration_occupancy[:,:,iteration] = occupancy
        
    # get mean occupancy
    mean_occ = np.mean(iteration_occupancy,2)
    occupancy = mean_occ
    p_t = np.divide(occupancy,np.max(np.nansum(occupancy,0)),where=occupancy>0)

    if 'conducting states' in Q.keys():
        conducting_occupancies = np.zeros([len(Q['conducting states'].keys()),np.size(occupancy,1)])
        for item, value in enumerate(Q['conducting states'].keys()): # multiply conductance of each state by occupancy and drivign force
            conducting_occupancies[item,:] = occupancy[value,:]*(((voltage-Vrev) *10**-3)*Q['conducting states'][value])
        currents = np.nansum(conducting_occupancies,0)
        currents[np.isnan(currents)] = 0
        p_t[np.isnan(p_t)]=0 # as above
        if plot:   
                plt.style.use('ggplot')
                figure,axes = plt.subplots(2,1)
                axes[0].plot(t,currents*10**12,color='black') # plot current
                axes[0].set_title("Simulated Current, N = {}, agonist pulse conc = {} M".format(N,Q['conc']))
                axes[0].set_xlabel("t (s)")
                axes[0].set_ylabel("pA")
                # plotting occupancy probabilities over time
                axes[1].set_prop_cycle(mycycle)
                for state in np.arange(np.size(p_t[:,0],axis=0)):
                    axes[1].plot(t,p_t[state,:],label="{}".format(state))
                axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format((1/interval)/1000))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
        return(t,p_t,occupancy,currents*10**12)

    else:
        return(t,p_t,occupancy)

def agonist_application_tau_leap_Gillespie(N,Q,t_final,agonist_time,agonist_duration,first_conc,second_conc,interval = 5e-05,voltage=0,Vrev = 0,iterations = 1,plot=True,rise_time = 250*10**-6,decay_time = 300*10**-6):
    """
    Simulation of an agonist pulse using a fixed interval tau leaping approach.
    
    For efficiency, the simulation is only performed until the concentration becomes zero after the pulse
    which means that the current is only plotted until then. If you want the entire duration
    make the agonist time = 0.5* t_final.
    
    Multiple iterations should be performed to generate macroscopic like behaviour
    
    Models with large numbers of states will run slower, but unlike adaptive Tau,
    having many concentration-dependent states does not majorly alter runtime.
    """
    #precalculate concentrations at discrete intervals, and get corrsponding times during the pulse
    concentrations,t = concentration_as_steps(first_conc=first_conc, second_conc=second_conc, dt=interval, start_time=agonist_time+(0.5*agonist_duration), duration=agonist_duration,rise_time = rise_time, decay_time = decay_time)
    
    # first, try to concat times and concs to here
    t_after_jump = np.arange(t[-1]+interval,t_final,interval)
    concs_after_jump = np.zeros(np.size(t_after_jump))
    concentrations = np.concatenate((concentrations,concs_after_jump))
    t = np.concatenate((t,t_after_jump))
    
    # above function generates concentrations only for a time-centred agonist application
    # so add in other times later to reduce computational cost

    # Pre-allocate for Q matrices and propensities, with Q for t intervals along axis 2 (wrd dimension)
    Qs = np.ndarray(shape = [np.size(Q['rates'],0),np.size(Q['rates'],0),np.size(t)])
    #propensities = np.ndarray(shape = [np.size(Q['rates'],0),np.size(Q['rates'],0),np.size(t)])
    
    # get attributes of Q matrix pertaining to concentration
    conc_fraction = Q['conc']
    conc_rates = [item for item in Q['conc-dep'].items()]
    
    # Pre-calculate Q matrices for each interval in 0,t_final
    # and also get propensities scaled to interval
    for intervalnum in np.arange(np.size(Qs,2)):
        Qs[:,:,intervalnum] = np.copy(Q['rates'])
        for item in conc_rates:
            # rescale each of the concentration-dependent rates for each time-dependent concentration
            Qs[item[0],item[1],intervalnum] = (concentrations[intervalnum]/conc_fraction)*(Qs[item[0],item[1],intervalnum])
    Qs[np.isnan(Qs)] = 0
    
    # Propensity is usually adjusted to interval length for stoichiometry-dependent reactions [i.e where the rate of overall
    # reaction is described by a forward and reverse rate]
    # Not needed here where propensity describes the relationship between dimensionless rates for a bidirected communicable reaction network
    # which does not change as a function of interval length - but does as func of conc (hence above)
    # i.e. for rates, can just take lambda of poisson = R*t, i.e. the rates per interval
    
    #pre-allocate storage for occupancy
    occupancy = np.zeros([np.size(Qs,0),np.size(t)])
    
    #take initial states for N with probability given in Q['intial states']
    initial_states = [int(i) for i in Q['initial states'].keys()]
    initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
    initial_occupancy = np.zeros(np.size(Q['rates'],axis=0))
    for item, value in enumerate(initial_states):
        initial_occupancy[value] = initial_probabilities[item]
    #populate t = 0 with initial states
    occupancy[:,0] = initial_occupancy
    
    #copy N for later
    n=copy.deepcopy(N)
    # get another Q dict for after jump using first conc rates
    S=copy.deepcopy(Q)
    S.update({'rates':Qs[:,:,0]})

    # initialise transition matrices
    transitions = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],1)])
    compar_accumulator = np.zeros([np.size(Q['rates'],0)])

    # iterate through jump
    iteration_occupancy = np.zeros([np.size(occupancy,0),np.size(np.arange(0,t_final,interval)),iterations])

    for iteration in np.arange(iterations):
        for intervalcount, timet in enumerate(t[:-1]):
            #for each state populated by receptors at intervalcount
            # = N to draw for exit that state
            prevstates = occupancy[:,intervalcount]
    
            for statefrom, N in enumerate(prevstates):
                for stateto in np.arange(np.size(Q['rates'],1)):
                    #compar_accumulator[stateto] = np.random.poisson((propensities[statefrom,stateto,intervalcount])*N)
                    if Qs[statefrom,stateto,intervalcount]>0:
                        compar_accumulator[stateto] = np.random.binomial(N,(Qs[statefrom,stateto,intervalcount]*interval)/(np.nansum(Qs[statefrom,:,intervalcount],0)*interval))
                    else:
                        compar_accumulator[stateto] = 0
                    compar_accumulator[np.isnan(compar_accumulator)] = 0
                    # because can generate negative or excessive populations of receptors,
                    # rescale draws such tht sum(draws) = N
                    # and round to nearest integer -- make sure still not generating excess
                compar_accumulator = np.floor(np.divide((np.clip(compar_accumulator,0,N)),(np.sum(np.clip(compar_accumulator,0,N))),where = compar_accumulator>0)*N)
                transitions[statefrom,:] = compar_accumulator
                
            # subtract transitions to sum(row) from prevstates elementwise
            # add to new states sum(col)
            newstates = prevstates - np.nansum(transitions,axis=1)
            newstates = newstates + np.nansum(transitions,axis=0)
            
            #update next occupancy
            occupancy[:,intervalcount+1] = newstates
        # for remaining time, perform a relaxation in the first conc
        # set initial states for the relaxation.
        if t[-1]<t_final -(agonist_time-(agonist_time-rise_time-decay_time-(0.5*agonist_duration))): # efficiency: correct for pulse translation before decide if needed
            S.update({'initial states':{}})
            for key, value in enumerate(occupancy[:,-1]/np.nansum(occupancy[:,-1])):
                if value >0:
                    S['initial states'].update({key:value})  
            S.update({'conc':first_conc})
            _,_,remaining_occs,_ = Tau_leap_Gillespie(N = n, Q = S,t_final = t_final-t[-1]-interval,interval = interval,voltage=voltage,Vrev = Vrev,iterations = 1,plot=False)
            iteration_occupancy[:,:,iteration] = np.hstack((occupancy,remaining_occs))[:,:np.size(iteration_occupancy,1)]
        else:
            iteration_occupancy[:,:,iteration] = occupancy[:,:np.size(iteration_occupancy,1)] # catch for single sample overspill
        
    # get mean occupancy for iterations
    mean_occ = np.mean(iteration_occupancy,2)
    occupancy = mean_occ # lazy catch for below
    
    #clip t and occupancy + derive time-matching pt
    t = t[t<t_final]
    occupancy = occupancy[:,:np.size(t)]
    p_t = np.divide(occupancy,np.max(np.nansum(occupancy,0)),where=occupancy>0)

    if 'conducting states' in Q.keys():
        conducting_occupancies = np.zeros([len(Q['conducting states'].keys()),np.size(occupancy,1)])
        for item, value in enumerate(Q['conducting states'].keys()): # multiply conductance of each state by occupancy and drivign force
            conducting_occupancies[item,:] = occupancy[value,:]*(((voltage-Vrev) *10**-3)*Q['conducting states'][value])
        currents = np.nansum(conducting_occupancies,0)
        currents[np.isnan(currents)] = 0
        p_t[np.isnan(p_t)]=0 # as above
        if plot:   
                plt.style.use('ggplot')
                figure,axes = plt.subplots(2,1)
                axes[0].plot(t,currents*10**12,color='black') # plot current
                axes[0].set_title("Simulated Current, N = {}, agonist pulse conc = {} M".format(N,Q['conc']))
                axes[0].set_xlabel("t (s)")
                axes[0].set_ylabel("pA")
                # plotting occupancy probabilities over time
                axes[1].set_prop_cycle(mycycle)
                for state in np.arange(np.size(p_t[:,0],axis=0)):
                    axes[1].plot(t,p_t[state,:],label="{}".format(state))
                axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format((1/interval)/1000))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
        return(t,p_t,occupancy,currents*10**12)

    else:
        return(t,p_t,occupancy)
# =============================================================================
# Adaptive Tau leaping Gillespie routines
# =============================================================================

#Can base on time for exhaustion of species as Cao, 2006...
# including information we have about the rates
def Weighted_adaptive_Tau_leap(N,Q,t_final,sampling = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True):
    """
    
    Performs a Gillespie walk with Tau leaping, where the step size (Tau) is
    determined by the rates of each transition, weighted by the 
    the fraction of species in each state.
    
    Because the state vector depends on a random draw from the poisson distribution,
    the interval sizes, and thus number of intervals, will differ between iterations.
    As such, the data is upsampled.
    
       Because of the upsampling, occupancies may be generated through interpolation that are
       not real numbers, but this isn't problematic, since:
           1. Probability of occupacny for each state cna still be calculated
           2. (Because) Occupancies at time t still sum to N
           3. The impact is minimised by averaging over iterations
        And this procedure avoids the 10-15% inaccuracy introducing by rounding during interpolation.
        
    This method means that runtime does not increase drastically with sampling rate.
    Models with large number of states will run slower, and each additional concentration-dependent
    transition increases the number of operations, so for models with large numbers of 
    concentration-dependent rates, using fixed Tau leaping is recommended. 
    Similarly, the length of simulation, t_final; will also impact performance for
    complex models.
    """
    # number of excepted occrs = 
    # draw poisson as np.random.poisson(propensities*interval)    
            
    # pre-allocate to store occupancy 
    Rates = np.copy(Q['rates'])
    Taus = np.divide(1,Rates,where= Rates>0)# pseudo time constants
    Rates[~np.isfinite(Rates)]=0
    Taus[~np.isfinite(Taus)]=np.nan
    
    #take initial states for N with probability given in Q['intial states']
    initial_states = [int(i) for i in Q['initial states'].keys()]
    initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
    initial_occupancy = np.zeros(np.size(Q['rates'],axis=0))
    for item, value in enumerate(initial_states):
        initial_occupancy[value] = initial_probabilities[item]
    
    # initialise transition matrices
    transitions = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0)])
    compar_accumulator = np.zeros([np.size(Q['rates'],0)])
    
    #pre-allocate to store (upsampled occupancy)
    sampled_occupancy = np.zeros([np.size(Q['rates'],0),np.size(np.arange(0,t_final,sampling)),iterations])
    
    for iteration in np.arange(iterations):
        t = 0
        times = []
        times.append(t)
        while t<t_final:
                #for each state populated by receptors at intervalcount
                # = N to draw for exit that state
                if t ==0:
                    occupancy = initial_occupancy    #populate t = 0 with initial states
                    if iteration == 0:
                        prevstates = np.copy(occupancy)
                    else:
                        prevstates = occupancy
                else:
                    prevstates = occupancy[-1,:]
                
                # get timestep size from weighted rates by pseudo time constantweighted interval w = (aTau^a +bTau^b...nTau^n)/a+b+...n
                # 1/~Tw = interval weighted, equivalent of which is below for rates: interval step depends on reactions leaving state i
                frcprev = prevstates/np.sum(prevstates) # fraction of N in each state
                dt = Taus*frcprev
                dt = np.nanmin(dt[dt>0])
                
                # determine which transitions occur
                for statefrom, N in enumerate(prevstates):
                    for stateto in np.arange(np.size(Q['rates'],1)):
                        compar_accumulator[stateto] = np.random.binomial(N,((Rates[statefrom,stateto])*dt)/(np.nansum(Rates[statefrom],0)*dt))
                        
                        # because can generate negative or excessive populations of receptors,
                        # rescale draws such tht sum(draws) = N
                        # and round to nearest integer -- make sure still not generating excess
                    compar_accumulator = np.floor(np.divide((np.clip(compar_accumulator,0,N)),(np.sum(np.clip(compar_accumulator,0,N))),where = compar_accumulator>0)*N)
                    transitions[statefrom,:] = compar_accumulator
                    
                #advance time
                t = t+dt
                times.append(t)
                    
                # subtract transitions to sum(row) from prevstates elementwise and add to new states sum(col)
                newstates = prevstates - np.nansum(transitions,axis=1)
                newstates = newstates + np.nansum(transitions,axis=0)
                
                #update next occupancy
                occupancy = np.vstack((occupancy,newstates))
        occupancy = occupancy.transpose()
        #upsample occupancy to uniform intervals
        # BUT this generate occupacnies that may not be real numbers
            # though occupancy of all states at time i will sum to N
        # not particularly an issue as still allows calculation of probability of occupancy of state i
        # and effect will be diminished by averaging
        for item in np.arange(np.size(occupancy,0)):
            interpolated_f = interp1d(times,occupancy[item,:],fill_value='extrapolate') 
            Tnew = np.arange(0, t_final, sampling)
            upsampled_occupancy = interpolated_f(Tnew)
            sampled_occupancy[item,:,iteration] = upsampled_occupancy

    # get mean occupancy
    mean_occ = np.mean(sampled_occupancy,2)
    occupancy = np.copy(mean_occ)
    p_t = np.divide(occupancy,np.max(np.nansum(occupancy,0)),where=occupancy>0)

    if 'conducting states' in Q.keys():
        conducting_occupancies = np.zeros([len(Q['conducting states'].keys()),np.size(occupancy,1)])
        for item, value in enumerate(Q['conducting states'].keys()): # multiply conductance of each state by occupancy and drivign force
            conducting_occupancies[item,:] = occupancy[value,:]*(((voltage-Vrev) *10**-3)*Q['conducting states'][value])
        currents = np.nansum(conducting_occupancies,0)
        currents[np.isnan(currents)] = 0
        p_t[np.isnan(p_t)]=0 # as above
        if plot:   
                plt.style.use('ggplot')
                figure,axes = plt.subplots(2,1)
                axes[0].plot(Tnew,currents*10**12,color='black') # plot current
                axes[0].set_title("Simulated Current, N = {}, agonist pulse conc = {} M".format(N,Q['conc']))
                axes[0].set_xlabel("t (s)")
                axes[0].set_ylabel("pA")
                # plotting occupancy probabilities over time
                axes[1].set_prop_cycle(mycycle)
                for state in np.arange(np.size(p_t[:,0],axis=0)):
                    axes[1].plot(Tnew,p_t[state,:],label="{}".format(state))
                axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format((1/sampling)/1000))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
        return(Tnew,p_t,occupancy,currents*10**12)

    else:
        return(Tnew,p_t,occupancy)
    
def Weighted_adaptive_agonist_application_Tau_leap(N,Q,t_final,agonist_time,agonist_duration,first_conc,second_conc,sampling = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True,rise_time = 250*10**-6,decay_time = 300*10**-6):
    """
    Performs a Gillespie walk with Tau leaping for an agonist application
    , where the step size (Tau) is determined by the rates of each transition,
    weighted by the fraction of species in each state.
    
    This method means that runtime does not increase drastically with sampling rate outside
    of the application. But, during the pulse, the chain may change rapidly, so as an insurance
    the interval is chosen during the application as either the sampling rate or the adaptive 
    interval (whichever is smaller), but after the pulse, Tau interval uses an adaptive strategy.
    
    Since runtime scales with the duration of agonist pulse, the interval size impacts performance
    through the above association.
        Models with large number of states will run slower.
        Though each additional concentration-dependent transition increases the number of operations
        so for models with large numbers of concentration-dependent rates, using fixed Tau leaping
        is recommended.
        Similarly if the pre-pulse concentration of agonist !=0, the process will be slower.
            In fact, it is recommended to use the fixed interval version when this is the case
            
    Some complex models may run faster in fixed interval simualtions (see agonist_application_tau_leap_Gillespie):
        e.g. GluMom03 runs faster wwith fixed interval
        conversely, threesQ and GlyAG are faster with the weighted (adaptive) application
    """
    # get concentrations
    concentrations,t_concs = concentration_as_steps(first_conc=first_conc, second_conc=second_conc, dt=sampling, start_time=agonist_time+(0.5*agonist_duration), duration=agonist_duration,rise_time = rise_time, decay_time = decay_time)
    # get function for interpolating concs
    interpolated_conc_f = interp1d(t_concs,concentrations,fill_value='extrapolate') 
    # to be used at each t for getting concs
    
    # get conc-dep properties amd max possible rates
    conc_fraction = Q['conc']
    conc_rates = [item for item in Q['conc-dep'].items()]

    #take initial states for N with probability given in Q['intial states']
    initial_states = [int(i) for i in Q['initial states'].keys()]
    initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
    initial_occupancy = np.zeros(np.size(Q['rates'],axis=0))
    for item, value in enumerate(initial_states):
        initial_occupancy[value] = initial_probabilities[item]
    
    # initialise transition matrices
    transitions = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0)])
    compar_accumulator = np.zeros([np.size(Q['rates'],0)])
    
    #pre-allocate to store (upsampled occupancy)
    sampled_occupancy = np.zeros([np.size(Q['rates'],0),np.size(np.arange(0,t_final,sampling)),iterations])
    
    for iteration in np.arange(iterations):
        t = 0
        times = []
        times.append(t)
        while t<t_final: # after pulse, should relax as expected
            
            # advancing to agonist_time-rise_time-decay_time+sampling is valid for = 0
            # 0 should trigger no relax
            while t<agonist_time: # while chain itself is stationary
                pre_pulse_concs = interpolated_conc_f(np.arange(0,agonist_time-rise_time-decay_time-(0.5*agonist_duration)+sampling,sampling))
                if not np.any(pre_pulse_concs): # if conc is zero before pulse
                    #advance to first non-zero concentration time
                    t = agonist_time
                    occs = np.vstack((initial_occupancy,initial_occupancy)).transpose()
                    times.append(t)
                else: # if pre-pulse conc is non-zero
                    # the conc would be fixed, so
                    # perform a relxation until chain non-stationary (i.e. pulse time)
                    
                    # first, get pre-pulse conc and then iniitalise a Q matrix to use in relaxation
                    newQ=copy.deepcopy(Q)
                    newQ_rates = newQ['rates']
                    newQ.update({'conc':first_conc})
                    conc_rates = [item for item in newQ['conc-dep'].items()]
                    for item in conc_rates:
                        newQ_rates[item[0],item[1]] = (first_conc)*(newQ_rates[item[0],item[1]])
                    newQ.update({'rates':newQ_rates})
                    _,_,occs,_  = Weighted_adaptive_Tau_leap(N = N,Q=newQ,t_final=agonist_time,sampling=sampling,voltage=voltage,Vrev=Vrev,iterations=1,plot=False)
                    times = list(np.arange(0,agonist_time,sampling))
                    t=agonist_time
                occupancy = occs
                
            while (t>=agonist_time) & (t<t_final): # when chain non-stationary
            
                interp_conc = interpolated_conc_f(t) # get conc
                prevstates = occupancy[:,-1] # get previous states
                
                # determine size of next interval
                # adjust conc-dep rates as appropriate
                Qs = np.copy(Q['rates'])
                for item in conc_rates:
                    Qs[item[0],item[1]] = (interp_conc/conc_fraction)*(Qs[item[0],item[1]])
                # caclulate timestep size from weighted rates
                # 1/~Tw = interval weighted, equivalent of which is below for rates
                #interval step depends on reactions leaving state 
                Rates = np.copy(Qs)
                Rates[~np.isfinite(Rates)] = 0 # for poisson selection and below
                Taus = np.divide(1,Rates,where=Rates>0) # pseudo time constants from maximum possible rates. Faster than 1/Rates
                frcprev = prevstates/np.sum(prevstates) # fraction of N in each state
                dt = np.multiply(Taus,frcprev,where=Taus>0) # efficiency
                if np.any(np.isfinite(dt) & (dt>0)): # catch - shouldn't be triggered
                    dt = np.nanmin(dt[dt>0])
                else:
                    dt = sampling
    
                # but if conc very low (at beginning of pulse), timestep very long
                    # unles impose some condition on non-zero concs so that dt does not step over pulse
                #for duration of the pulse, because the chain changes rapidly, use dt = sampling if dt longer
                if t<= agonist_time+rise_time+agonist_duration:
                    if dt > sampling: #
                        dt = sampling
                        
                # determine which transitions occur in next interval
                for statefrom, N in enumerate(prevstates):
                    for stateto in np.arange(np.size(Q['rates'],1)):
                        if Rates[statefrom,stateto]>0:
                            compar_accumulator[stateto] = np.random.binomial(N,(Rates[statefrom,stateto]*dt)/(np.nansum(Rates[statefrom,:],0)*dt)) # probability must be normalised, so scale to norm factor by propensity
                        else:
                            compar_accumulator[stateto] = 0
                        # because can generate negative or excessive populations of receptors,
                        # rescale draws such tht sum(draws) = N
                        # and round to nearest integer -- make sure still not generating excess
                    compar_accumulator = np.floor(np.divide((np.clip(compar_accumulator,0,N)),(np.sum(np.clip(compar_accumulator,0,N))),where = compar_accumulator>0)*N)
                    transitions[statefrom,:] = compar_accumulator
                    
                #advance time
                t = t+dt
                times.append(t)
                    
                # subtract transitions to sum(row) from prevstates elementwise
                # add to new states sum(col)
                newstates = prevstates - np.nansum(transitions,axis=1)
                newstates = newstates + np.nansum(transitions,axis=0)
                
                #update next occupancy
                occupancy = np.vstack((occupancy.transpose(),newstates)).transpose()
        #upsample occupancy to uniform intervals
        # though this generate occupacnies that may not be real numbers
            # see function docs.
        for item in np.arange(np.size(occupancy,0)):
            interpolated_f = interp1d(times,occupancy[item,:],fill_value='extrapolate') 
            Tnew = np.arange(0, t_final, sampling)
            upsampled_occupancy = interpolated_f(Tnew)
            sampled_occupancy[item,:,iteration] = upsampled_occupancy
    
    # get mean occupancy
    mean_occ = np.mean(sampled_occupancy,2)
    occupancy = np.copy(mean_occ)
    
    # clip Tnew, occupancy to t_final & get p_t:
    if Tnew[-1]>t_final:
        occupancy = occupancy[:,:np.min(np.where(Tnew>t_final))]
        Tnew = Tnew[:np.min(np.where(Tnew>t_final))]
    p_t = np.divide(occupancy,np.max(np.nansum(occupancy,0)),where=occupancy>0)    
            
    if 'conducting states' in Q.keys():
        conducting_occupancies = np.zeros([len(Q['conducting states'].keys()),np.size(occupancy,1)])
        for item, value in enumerate(Q['conducting states'].keys()): # multiply conductance of each state by occupancy and drivign force
            conducting_occupancies[item,:] = occupancy[value,:]*(((voltage-Vrev) *10**-3)*Q['conducting states'][value])
        currents = np.nansum(conducting_occupancies,0)
        currents[np.isnan(currents)] = 0
        p_t[np.isnan(p_t)]=0 # as above
        if plot:   
                plt.style.use('ggplot')
                figure,axes = plt.subplots(2,1)
                axes[0].plot(Tnew,currents*10**12,color='black') # plot current
                axes[0].set_title("Simulated Current, N = {}, agonist pulse conc = {} M".format(N,second_conc))
                axes[0].set_xlabel("t (s)")
                axes[0].set_ylabel("pA")
                # plotting occupancy probabilities over time
                axes[1].set_prop_cycle(mycycle)
                for state in np.arange(np.size(p_t[:,0],axis=0)):
                    axes[1].plot(Tnew,p_t[state,:],label="{}".format(state))
                axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format((1/sampling)/1000))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
        return(Tnew,p_t,occupancy,currents*10**12)

    else:
        return(Tnew,p_t,occupancy)
    
# =============================================================================
#  [3] Q-matrix Methods and CME simulation
# =============================================================================
def Q_relax(Q,N,t_final,voltage= -60,interval= 5e-05,Vrev= 0,plot =True, just_pt=False):
    """Simulates the change in distribution for N receptors over t=0, t_final
    for a single iteration - i.e. During a relaxation to steady-state, with Q as time-homogenous.
    Returns occupancy probability, or simulated currents (pA) and occupancies
    (see just_pt arg). For visualisation, See Graphs arg.
    Assumes constant concentration, such that p_inf constant.
    
    Q should be a Q dictionary containing the desired concentration, also containing information about conducting states, initial states etc.

    - t_final (s): total simulation time
    - voltage (mV). Default = -60.
    - interval (s): size of timestep interval
    - plot returns plots when True. When False, returns output in order:
    occupancy probability,occupancy, and currents
    - just_pt (Default=False) returns only the occupancy probabilties for each time
    when = True.
    - Vrev is the reversal potential of the channel in mV. Default -0.
    """
    t = np.arange(0,t_final,(interval))
        
    #take initial states for N with probability given in Q['intial states']
    initial_states = [int(i) for i in Q['initial states'].keys()]
    #initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
    # above changed to below changed to below
    initial_probabilities = [i*N for i in Q['initial states'].values()]
    initial_occupancy = np.zeros(np.size(Q['rates'],axis=0))
    for item, value in enumerate(initial_states):
        initial_occupancy[value] = initial_probabilities[item]
    # preallocate for p(occupancy at t) and perform spectral decomposition
    p_t = np.zeros([np.size(Q['Q'],0),np.size(t,0)]) # preallocation to store probabilities (from t =0+t to t=t_final), not initial probs
    eigvals,eigvecs,spectrals = spectral_matrices(Q['Q'],ret_eig=True) # calling embedded function to compute sorted eigenvalues,eigenvectors, and spectral matrices.
    # each component ofspectrals, k, is stored in the 3rd axis (axis =2)
    # with each time interval, p_t changes, and lit changes
    pinf = p_inf(Q['Q']) # steady-state occupancies
    pzero = np.divide(initial_occupancy,np.nansum(initial_occupancy),where=initial_occupancy>0) # occupancy probabilities for initials tates,if specified or not
    amplitude_coefficients = amplitude_coeff(spectrals, pzero) #calling embedded function to find the amplitude coefficients
    for intervalnum, intervaltime in enumerate(t):
        exp_component = np.exp(-eigvals*intervaltime) # calculating exponential terms at time
        p_jt = coefficients_to_components(exp_component, amplitude_coefficients) # calling embedded function to multiply out terms ( = probability of state j occupancy at t)
        p_t[:,intervalnum] = pinf + p_jt
    if just_pt:
        return(p_t)
    else:
        occupancy = p_t*N
        if 'conducting states' in Q.keys():
            conducting_occupancies = np.zeros([len(Q['conducting states'].keys()),np.size(occupancy,1)])
            for item, value in enumerate(Q['conducting states'].keys()): # multiply conductance of each state by occupancy and drivign force
                conducting_occupancies[item,:] = occupancy[value,:]*(((voltage-Vrev) *10**-3)*Q['conducting states'][value])
            currents = np.nansum(conducting_occupancies,0)
            currents[np.isnan(currents)] = 0
            if plot:   
                plt.style.use('ggplot')
                figure,axes = plt.subplots(2,1)
                axes[0].plot(t,currents*10**12,color='black') # plot current
                axes[0].set_title("Simulated Current, N = {}, agonist conc = {} M".format(N,Q['conc']))
                axes[0].set_xlabel("t (s)")
                axes[0].set_ylabel("pA")
                axes[1].set_prop_cycle(mycycle)
                # plotting occupancy probabilities over time
                for state in np.arange(np.size(p_t[:,0],axis=0)):
                    axes[1].plot(t,p_t[state,:],label="{}".format(state))
                axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format(1/interval))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
            return(t,p_t,occupancy,currents*10**12)
        else:
            return(t,p_t,occupancy)

def Q_agonist_application(Q,N,first_conc,second_conc,agonist_time,agonist_duration,t_final,interval = 5e-04,voltage =-60,Vrev = 0,rise_time=250*10**-6,decay_time=250*10**-6,plot = True):
    """
    Simulating a fast jump between two agonist concentrations by:
        
    A relaxation is performed in first_conc until the agonist application,
    then the concentration function erf(second_conc) is convolved with the Q
    matrix at each timestep during the jump, before a final relaxation is
    performed in first_conc for the remaining time.
    
    - Q: A Q dictionary containing a Q matrix
    - N: the number of receptors to simulate
    - first_conc (M): the concentration of the agonist before the jump
    - second_conc (M): the concentration of agonist during the jump
    - onset_time (s): the time to apply the agonist (should be > 2x duration)
    - application_duration (s): the duration in second_conc
    - record_length (s): The length of the simulated trace
    - interval (s): The size of the timestep interval
    - rise_time (s): the 10-90% rise time fo the fast jump
    - decay_time (s): the 10-90% decay time of the jump
    - voltage(mV): The voltage
    - Vrev(mV): The reversal potential of the channel 
    - plot: when True, displays plots
    - sq_pulse. If true, overrides realistic cocnentration jumps to use a square pulse.

    """
    # get t by interval and discrete concentration by timestep
    t = np.arange(0,t_final,interval)
    jump_concs,jump_times = concentration_as_steps(first_conc=first_conc,second_conc=second_conc,dt=interval,start_time=agonist_time+(0.5*agonist_duration),duration = agonist_duration,rise_time=rise_time,decay_time=decay_time)
    
    # currently deprecated
    # if sq_pulse: # square pulse
    #     concen_s = np.zeros(np.size(t))
    #     concen_s[t<agonist_time]=first_conc
    #     concen_s[(t>=agonist_time)&(t<=agonist_time+agonist_duration)]=second_conc
    #     concen_s[t>agonist_time+agonist_duration] = first_conc
    #     jump_concs = concen_s
    
    #take initial states for N with probability given in Q['intial states']
    # this won't work when probabilities are not wholen umbers. I.e.
        # when resting state not occupied with probability 1.
    initial_states = [int(i) for i in Q['initial states'].keys()]
    #initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
    # above changed to below
    initial_probabilities = [i*N for i in Q['initial states'].values()]
    
    initial_occupancy = np.zeros(np.size(Q['rates'],axis=0))
    for item, value in enumerate(initial_states):
        initial_occupancy[value] = initial_probabilities[item]
        
    #precalculate Q matrices for each timestep of the jump: 3d array
    conc_fraction = Q['conc']
    Qs = np.zeros([np.size(Q['Q'],0),np.size(Q['Q'],1),np.size(jump_times)])
    for intervalnum, value in enumerate(jump_times):
        Qs[:,:,intervalnum] = np.copy(Q['rates']) # get rates from transition matrix
        for item in Q['conc-dep'].items(): # scale the conc-dep rates to conc(t)
            Qs[item[0],item[1],intervalnum] = (jump_concs[intervalnum]/conc_fraction)*(Qs[item[0],item[1],intervalnum])
        for row in np.arange(np.size(Qs,0)): # re-establish as Q matrix
            Qs[row,row,intervalnum] = -np.nansum(Qs[row,:,intervalnum])
    Qs[~np.isfinite(Qs)]=0

    # pre-allocate occupancy for jump
    occupancy_in_jump = np.zeros([np.size(Qs,0),np.size(jump_times)])
    
    # Perform a relaxation from t = 0 until jump time in the pre-jump constant concentration of agonist
    # get t at jump
    t_at_jump = jump_times[np.min(np.where(jump_times>agonist_time))] #+ interval
    jump_indexer = np.min(np.where(jump_times>agonist_time)) #+ 1

    # relax in the pre-jump_constant_conc
    R = copy.deepcopy(Q)
    R.update({'Q':Qs[:,:,0]}) # update with the Q for pre-agonist constant conc
    R.update({'conc':first_conc}) # and the conc
    _,pre_jump_pt,occs,_ = Q_relax(Q=R,N = N,Vrev=Vrev,interval=interval,t_final=t_at_jump,voltage=voltage,plot=False,just_pt=False)
    occupancy_in_jump[:,:jump_indexer] = occs # store pre-jump occs
    
    # apply agonist during the jump
    jump_pt = np.zeros([np.size(Q['Q'],0),np.size(jump_times)-jump_indexer])
    for concentration_interval, conc in enumerate(jump_concs[jump_indexer:]):
        eigvals,eigvecs,spectrals = spectral_matrices(Qs[:,:,concentration_interval+jump_indexer],ret_eig=True) # calling embedded function to compute sorted eigenvalues,eigenvectors, and spectral matrices.
        pinf = p_inf(Qs[:,:,concentration_interval+jump_indexer])
        if concentration_interval ==0:
            pzero = (occupancy_in_jump[:,concentration_interval+jump_indexer-1])/np.nansum(occupancy_in_jump[:,concentration_interval+jump_indexer-1])
        else:
            pzero = jump_pt[:,concentration_interval-1]
        amplitude_coefficients = amplitude_coeff(spectrals, pzero) #calling embedded function to find the amplitude coefficients
        exp_component = np.exp(-eigvals*interval) # will be constant
        p_jt = coefficients_to_components(exp_component, amplitude_coefficients) # calling embedded function to multiply out terms ( = probability of state j occupancy at t)
        jump_pt[:,concentration_interval] = pinf + p_jt
    occupancy_in_jump[:,jump_indexer:] = jump_pt * N

    #after jump, relax for remaining time in the first conc
    if t_final>jump_times[-1]:
        S = copy.deepcopy(Q)
        S.update({'Q':Qs[:,:,0]})
        S.update({'conc':first_conc}) # and update the conc to first_conc
        S.update({'initial states':{}})
        for key, value in enumerate(jump_pt[:,-1]):
            if value >0:
                S['initial states'].update({key:value})
        _,post_jump_pt,post_jump_occs,_ = Q_relax(Q=S,N = N,Vrev=Vrev,interval=interval,t_final=t_final-jump_times[-1],voltage=voltage,plot=False,just_pt=False)
        post_jump_pt[np.isnan(post_jump_pt)]=0 # as above

        # concatenate all pt & occs
        occupancy =np.hstack((occupancy_in_jump,post_jump_occs))
    else:
        occupancy = occupancy_in_jump    
    # clip Tnew, occupancy to t_final & rederive p_t from occs (so time-matched)
    occupancy = occupancy[:,:np.size(t)]
    p_t = np.divide(occupancy,np.max(np.nansum(occupancy,0)),where=occupancy>0)
    
    # tidying up & plot
    if 'conducting states' in Q.keys():
        conducting_occupancies = np.zeros([len(Q['conducting states'].keys()),np.size(occupancy,1)])
        for item, value in enumerate(Q['conducting states'].keys()): # multiply conductance of each state by occupancy and drivign force
            conducting_occupancies[item,:] = occupancy[value,:]*(((voltage-Vrev) *10**-3)*Q['conducting states'][value])
        currents = np.nansum(conducting_occupancies,0)
        currents[np.isnan(currents)] = 0 # fix for not plotting remainder whne == np.nan
        if plot:   
                plt.style.use('ggplot')
                figure,axes = plt.subplots(2,1)
                axes[0].plot(t,currents*10**12,color='black') # plot current
                axes[0].set_title("Simulated Current, N = {}, agonist pulse conc = {} M".format(N,second_conc))
                axes[0].set_xlabel("t (s)")
                axes[0].set_ylabel("pA")
                axes[1].set_prop_cycle(mycycle)
                # plotting occupancy probabilities over time
                for state in np.arange(np.size(p_t[:,0],axis=0)):
                    axes[1].plot(t,p_t[state,:],label="{}".format(state))
                axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format((1/interval)/1000))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
        return(t,p_t,occupancy,currents[:-1]*10**12)

    else:
        return(t,p_t,occupancy)
# =============================================================================
#  [4] Simulation Master Function and File Handling
# =============================================================================
def simulate(func,n_sweeps,noise_sd = 4,show_progress=True,graph=True,**kwargs):
    """
    Takes Q dictionary (see example models) to simulate a record consisting of
    n_sweeps using func. The purpose is to allow simultaion of Ephys records
    through repeated, identical stimuli when optimal conditions for simulation
    of that model have been established. This is more useful for stochastic 
    simulations, where the output is different for each call
    
    Gaussian Noise is also added using a standard deviation value = noise_sd. 2pA
    is recommended.

    Recommendations for usage:
        - If using fixed interval Tau methods, ensure sampling is high enough
        that the fastest events are captured, but low enough that 'ringing',
        rapid oscillations between values, does not occur.
        - Generate realistic macropscopic behaviour as follows:
           - A single iteration of a stochastic simulation function generates a trajectory.
           - To generate a stochastic 'sweep', or single stimulus epoch, we
           average the trajectory occupancy over several iterations to generate a current
           for a single sweep. This current will display stochastic behaviour and should
           closely match the equivalent stimulus being applied in a Q-matrix (CME) type
           simulation [For proof, see Gillespie, 1977].
           - So to generate a record, we repeat the sweep generation procedure many times.
           
           - First,determine how many iterations (itn), and if using fixed interval methods -
           which sampling rate - give rise to a sweep with a current trajectory
           that broadly recapitulates the current trajectory in Q-matrix (CME) methods: these are
           Q_relax and Q_agonist_application.
           - Then set that keyword argument (kwarg) = itn
           - Then set n_sweeps = number of sweeps desired
           
    Usage:
        e.g. to store all outputs in an array called 'P1':
        for the fixed Tau interval simulation method
        and using noise standard deviation in pA = 2.0
        
        P1 = simulate(func = agonist_application_tau_leap_Gillespie,n_sweeps = 10,noise_sd = 2,N=100,Q=Q,t_final = 0.5,agonist_time = 0.1,agonist_duration = 0.1,first_conc = 0,second_conc = 10*10**-3,interval = 1e-03,voltage =-60,Vrev = 0,iterations = 10)
                  # obviously set Q as the dired Q dict before (e.g. Q = threesQ())
                  # If kwarg not assigned, will use Default.
        i.e. using the arguments for the base function as keyword arguments (kwargs),
        and parameters for the simulation, ensuring that iterations is set = itn (see above)
        as well as filling the function method (func=) to the desired method
        and n_sweeps to number of sweeps desired in the record
        
    The sweeps will be plotted with offset, and the average plotted will be averaged *currents* of all
    sweeps, where the current from each sweep is produced from average occupancy.
    The plotted averaged occupancy probability is the average of all sweep occupancy probabilities.
    
    Returns:
        a nested dict, containing:
            - t: timepoints corresponding to samples of below
            - p_t: occupancy_probabilities: a 3d np.ndarray, with dimensions 0=states,1=time,2=sweep number
            - occ_t: occupancy: a 3d np.ndarray, with dimensions 0 = states,1 = time,2 = sweep number
            - I_t: currents: a 2d np.ndarray, with dimensions 0 = time, 1 = sweep number
            - conditions: all kwargs listed (to detail simulation conditions, e.g. N) + noise_sd + n_sweeps
            
    Structure of the nested dict is easily viewable for a returned record P1 by entering P1.keys().
    Similarly, values can be accessed by P1[keyi].values()
    Values themselves may be dicts of key:values pairs (nested)
    """ 
    storage = False
    show_progress = not show_progress # added
    for item in tqdm(np.arange(n_sweeps),disable=show_progress): # progress bar updates at each sweep
        output = func(**kwargs,plot=False)
        t,pt,occ,current = output[0],output[1],output[2],output[3]
        if not storage:
            pts = np.zeros([np.size(pt,0),np.size(pt,1),np.size(np.arange(n_sweeps))])
            occs = np.zeros([np.size(pt,0),np.size(pt,1),np.size(np.arange(n_sweeps))])
            currents = np.zeros([np.size(current),np.size(np.arange(n_sweeps))])
            t = t # should be same for all (even for adaptive Tau, sampling makes np.size(t) identical)
            storage = True
        
        pts[:,:,item] = pt
        occs[:,:,item] = occ
        currents[:,item] = Add_Gaussian_noise(current,noise_sd=noise_sd) # store current with noise
        # if not show_progress:
        #     if item == n_sweeps-1:
        #         tqdm.ncols=0    
        
    # set output dicts
    record_dict = {}
    record_dict.update({'t':t})
    record_dict.update({'p_t':pts})
    record_dict.update({'occ_t':occs})
    record_dict.update({'I_t':currents})
    record_dict.update({'conditions':kwargs})
    record_dict['conditions'].update({'n_sweeps':n_sweeps})
    record_dict['conditions'].update({'noise_sd':noise_sd})
    # create plots
    plt.style.use('ggplot')
    if graph:
        figure,axes = plt.subplots(2,1)
    # avg currents and occupancy probability
        axes[0].plot(t,np.mean(currents,axis=1),color='black') # plot current
        axes[0].set_title("Mean Simulated Current, N = {}".format(record_dict['conditions']['N']))
        axes[0].set_xlabel("t (s)")
        axes[0].set_ylabel("pA")
        axes[1].set_prop_cycle(mycycle)
        for state in np.arange(np.size(pts,axis=0)):
            axes[1].plot(t,np.mean(record_dict['p_t'],axis=2)[state,:],label="{}".format(state)) # avg for all sweeps
        axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
        if 'interval' in record_dict['conditions'].keys(): # method-dependent kwarg
            axes[1].set_title(" Mean P(State Occupancy at t), {}kHZ".format((1/record_dict['conditions']['interval'])/1000))
        else:
            axes[1].set_title("Mean P(State Occupancy at t), {}kHZ".format((1/record_dict['conditions']['sampling'])/1000))
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylabel("Probability")
        plt.tight_layout()
        # plotting all sweeps with offset
        offset_factor = np.max(np.abs(np.mean(currents,axis=1)))
        fig,axs = plt.subplots(num=2)
        axs.set_prop_cycle(mycycle)
        for item in np.arange(n_sweeps):
            axs.plot(t,currents[:,item]+item*offset_factor)
            axs.annotate(text = '{}'.format(item),xy =(np.max(t)/2,np.max(np.mean(currents,axis=1)+item*offset_factor)),xycoords='data')
        axs.set_xlabel("t (s)")
        axs.set_title("Offset sweeps of record")
        plt.tight_layout()
    return(record_dict)
    
def save_sim(record_dict,path,filename):
    """
    Provided with a record dictionary, of type produced by simulate function
    saves file to a path. Path can be dragged. Filename should be string
    """
    with open(path+filename,'wb') as handle:
        pickle.dump(record_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_sim(path):
    """
    Provided a path to a saved record dictionary, loads it and returns it in the expected format
    """
    with open(path,'rb') as handle:
        record_dict = pickle.load(handle)
    return(record_dict)

def current_to_DataFrame(record_dict):
    """
    Provided with a record dictionary of type produced by simulate,
    returns a pandas DataFrame to allow harmonisation with EpyPhys

    """
    curr_as_df = pd.DataFrame(data = record_dict['I_t'],index = record_dict['t'])
    return(curr_as_df)

# =============================================================================
# [5] Built-In functions
# =============================================================================
def Q_graph(Q):
    """Plots Q as a graph format (I.e. as a network)"""
    grp = nx.MultiDiGraph(Q['rates'])
    grp.remove_edges_from(list(nx.selfloop_edges(grp))) # remove self loop edges
    nx.draw(grp,pos = nx.spring_layout(grp),with_labels=True)

def concentration_as_steps(first_conc,second_conc,dt,start_time,duration,rise_time = 400,decay_time=False):
    """Calculates the concentration of a drug at discrete time intervals,
    according to a fast jump from condition with concentration =  first-conc
    to a condition with concentration second_conc of the same drug. Returns
    concentrations and times.
    
    A single instance of concentration_as_steps 
    
    C(t) is calculated by fitting an error function to the onset/offset time
    (Sachs,1999).
    
    Args:
        - first_conc (M): The concentration of a drug in starting condition.
          can be 0.
        - second_conc (M): Concentration of a drug in the condition after a jump.
          can be 0.
        - dt (ms): the size of the time interval at which to caclulate concentration
            of the above drug 
        - start_time (ms): the time at which to apply the pulse maximum (i.e. centre of pulse)
        - duration (ms) the pulse half duration at half maximum (i.e. length of application).
        - rise_time (us) the 10-90% rise time constant. Default 400uS
        - decay_time (us) to 90-10% decay time constant. By Default = rise_time.
        
        """
    rise_time = rise_time*10**-3 # converting to mS
    if not decay_time:
        decay_time = rise_time
    else:
        decay_time = decay_time*10**-3
    centre = start_time+rise_time
    t = np.arange(0,(start_time+centre+decay_time),dt) # creating times to calculate concentration at in intervals of dt
    c_difference = -(first_conc-second_conc) # difference between conc streams used
    # conc calculated as error signal from first conc to second conc, scaled by first conc
    conc = (c_difference * 0.5 * (special.erf((t - centre + duration / 2) / rise_time) - special.erf((t - centre - duration / 2) / decay_time))) + first_conc
    return(conc,t)

def Add_Gaussian_noise(current,noise_sd = 2):
    """
    Given a current, adds noise to, and returns the current.
    
    noise_sd refers to the standard deviation of the noise to add. Default = 2 pA
    """
    samples = np.size(current)
    noise = np.random.default_rng(random.randint(0,2**32)).normal(0,noise_sd,samples)
    current = current+noise
    return(current)
    
def p_inf(Q):
    """given a matrix Q, returns the equilibrium probabilities of state occupancy"""
    # Performed according to Colquhoun and Hawkes (1995) notation
    S = np.zeros([np.size(Q,0),np.size(Q,1)+1]) # Q,col k+1 = 1
    S[:,:-1] = Q
    S[:,-1] = 1
    u = np.zeros(np.size(Q,0))
    u[:]=1 # row vector u
    SST_inv = np.linalg.pinv(np.dot(S,(S.transpose()))) #pseudo-inverse of SST, since SST is singular matrix of determinant 0
    pinf = np.dot(u,SST_inv) # equil probability.
    return(pinf)

def spectral_matrices(Q=False,eigvecs = False,ret_eig= False):
    """Returns spectral matrices from sorted eigenvectors.
    
    For speed, eigenvectors can be specified, but if False (Default)
    then are computed from supplied Q matrix.
    
    if ret_eig =True, SORTED eigenvalues and eigenvectors of -Q are also
    returned by eigvals,eigvectors,spectrals = spectral matrices()"""
    if not np.any(eigvecs): #if no eigenvalues given, get eigenvalues & vectors; and sort.
        eigvals,eigvecs = np.linalg.eig((-Q))
        order = np.argsort(eigvals)
        eigvals = np.matrix(eigvals[order])
        eigvecs = np.matrix(eigvecs[:,order]) # each col i of eigvecs corresponds to eigval[i]
    # maintaining Colquhoun and Hawkes notation
    X = np.matrix(eigvecs) #conversion to type matrix for numpy matrix multiplication
    Y= np.matrix(np.linalg.inv(eigvecs)) # "
    spectrals= np.zeros([np.size(Q,0),np.size(Q,0),np.size(Q,0)]) # preallocation to store k spectral matrices
    for state in np.arange(np.size(Q,0)):
        spectrals[:,:,state] = X[:,state]*Y[state,:]
    if ret_eig:
        return(eigvals,eigvecs,spectrals)
    else:
        return(spectrals)

def amplitude_coeff(spectral_matrices,p_zero):
    """finds amplitude coefficient of exponential components"""
    #by Colquhoun and Hawknes notation
    wijr = np.zeros(np.size(spectral_matrices,axis=2))
    wij = np.zeros([np.size(spectral_matrices,axis=2),np.size(spectral_matrices,axis=2)])
    for component in np.arange(np.size(spectral_matrices,2)): # for each component i = {0:k} 
        for state in np.arange(np.size(spectral_matrices,2)): # for each state,j = {0:k}
            for row in np.arange(np.size(p_zero)): # for each row, r= {0:k}
                wijr[row] = p_zero[row]*spectral_matrices[row,state,component] #wij = (rows[i:k])
                wij[component,state] = np.sum(wijr) # [i,j] = sum(rows[i:k])
    return(wij)

def coefficients_to_components(exponential_components,amplitude_coefficients):
    """To solve for p(t), multiplies out exponential terms by their amplitude components
    excluding the zero order exponents that describe steady-state occupancy probability
    (since amplitude_components[0,:] returns steady-state occupancy)
    
    exponential_components is expected as a 1*karray
    """
    exponential_components=exponential_components.transpose() # formatting for • product
    pj_t = np.zeros(np.size(exponential_components))
    for state in np.arange(np.size(exponential_components,0)): # for each state, j
        # take non-zero- i.e. not-steady-state components (where np.dot(amplitude_coefficients[0,j],exp_component[:,0])[0,:] gives zero order component for state j)
        pj_t[state] = np.dot(amplitude_coefficients[1:,state],exponential_components[1:,0]) # i=2:k(sum(wij)*exp(-lt))
    return(pj_t)

def logicalQ_constructor(Q):
    """Constructs logical transition matrix for use by microscopic reversibility
    setting functions, consisting of 0s for no rate, 1 for real rate to be
    constrained by MR, and 0.5 for rates to be fixed/set by user""
    Args:
        Q: a Q matrix

        When prompted to CONSTRAIN rates, enter rate to constrain as as:
            [fromstate,tostate];[fromstate,tostate]
            NB semicolon"""
    Q_logical_edges = np.zeros([np.size(Q,0),np.size(Q,0)])
    Q_logical_edges = np.where(Q>0,1,Q_logical_edges)
    print("Some rates may be constrained by data, and others can be set by microscopic reversibility")
    constrain = input("Enter edges to constrain. See help(logicalQ_constructor) >").split(";")
    if any(constrain):
        for item, value in enumerate(constrain):
            ij = np.matrix(constrain[item])
            Q_logical_edges[ij[0,0],ij[0,1]] = 0.5
    return(Q_logical_edges)

def path_rates(path,adjusted_rate_matrix):
    """Accepts path of type numpy.ndarray, where each entry is a visited node,
    and then calculates the probability of a transition from source:target
    as the product of transition probabilities between nodes of the path"""
    # flip path
    path = np.flip(path)
    edge_rates = np.zeros(np.size(path)-1)
    #taking the flipped path, get the list of transitions that occur
    for item, value in enumerate(path):
        if value != path[-1]:# if not last node on the list
            edge_rates[item] = adjusted_rate_matrix[path[item],path[item+1]]
    path_rate = np.product(edge_rates) #rate of the path from source:target = prob(T1)*prob(T2)... where T1... are all transitions in that path
    return(path_rate)

# def make_Q_reversible(Q,show=False,protected_rates = False):
#     """Taking a Q matrix and applying microscopic reversibility to it
#     using minimum spanning tree principle from Colquhoun et al., 2004
    
#     A graph is constructed, identifying the rates to constrain, and the
#     remainder are set by microscopic reversibility. For rates to be set
#     by MR, a minimum spanning tree is identified. By the principle that these
#     form independent cycles, the MR rates can then be set in any order.
    
#     Args:
#     Q: a Q matrix containing rates. It is easiest if this matrix contains
#         rates that one wants to constrain the model by. Any rates that exist
#         but should be set by MR should be any value >0. The option will be
#         given to select which rates should be constrained (see protected_rates too).
        
#     show (Default=False): 
#         When = True, a graph is produced to show the model, with each node as a state
    
#     protected_rates (Default = False). 
#             When False, the user will be asked to
#             select rates to constrain (i.e those not to be set by MR). These
#             will then form an additional output for future usage. Otherwise,
#             protected_rates should be entered in the format as the type produced
#             from this function, and they will not be output.
            
#             E.g. In first instance, use: Q,protected_rates = make_Q_reversible(Q,protected_rates=False)
#             Then using the output from above, use: Q = make_Q_reversible(Q,protected_rates=protected rates)
            
#             The user will be prompted to enter edges to constrain in format [from,to];[from2,to2] etc.
    
#     """
#     if not np.any(protected_rates):
#         Q_logical = logicalQ_constructor(Q) #calling embedded to create a logical matrix
#     else:
#         Q_logical = protected_rates
#     logic_graph = nx.MultiDiGraph(Q_logical)
#     for edge in logic_graph.edges():
#         # rates to be set by microscopic reversibility have weight 1, set have weight 0.5
#         if logic_graph.get_edge_data(edge[0],edge[1])[0]['weight'] ==1: # rate ij = rateji*(product rates of forward route/product rates of reverse route)
#             # where item[0] is source and itme[1] is target, getting shortest forward and reverse paths
#             forward_path = nx.shortest_path(logic_graph,edge[0],edge[1])
#             reverse_path = nx.shortest_path(logic_graph,edge[1],edge[0])
#             # getting product of rates along this path using embedded function
#             forward_rates_prod = path_rates(forward_path,Q)
#             reverse_rates_prod = path_rates(reverse_path,Q)
#             Q[edge[0],edge[1]] = Q[edge[1],edge[0]] * (forward_rates_prod/reverse_rates_prod)
#     if show:
#         Qgraph = nx.MultiDiGraph(Q)
#         nx.draw(Qgraph,pos = nx.spring_layout(Qgraph),with_labels=True)
#     if not np.any(protected_rates):
#         return(Q,Q_logical)
#     else:
#         return(Q)

def resample_simulation(simulation_dict,freq,Vrev=0,voltage=False,update=False,keepnoise=True):
    """
    A function primarily intended for changing sampling of an existing simulations,but
    has several other uses.
    
    Resamples a current to a desired frequency in Hz using the raw occupancies (which
    are the raw data in trajectory type simualtions). 
    Can also be used to change driving force of a simulation (if voltage is not False and is
    instead set to mV value) or if Vrev is changed.
    
    
    A simulation dict should be provided of type produced by simulate function
    that contains the key 'occ_t', which is a numpy.ndarray of shape(nstates,nsamples,nsweeps)

    If update = True (Default = False), the simualtion_dict is updated. Otherwise,
    a new dictionary is returned containing the upsampled values
    
    if keepnoise = True (default), the existing noise is layered onto the current, otherwise,
    it should be set to a value (in pA) to add onto a denoised, resampled current.
    
    This could be adapted to change N as well.
    """
    if voltage==False:
        flag = 'newdriving'
        voltage=simulation_dict['conditions']['voltage']
    pt = copy.deepcopy(simulation_dict['p_t']) # p_t is pre-normalised in trajectory type simulations (i.e. = occ/N)
    times = simulation_dict['t']
    # have to use 1d to avoid overflow in interpolation, so interpolate by looping
    Tnew = np.arange(0, simulation_dict['t'][-1], 1/freq)
    resampled_pt = np.zeros([pt.shape[0],np.size(Tnew),pt.shape[2]])
    for sweep in np.arange(pt.shape[2]):
        for state in np.arange(pt.shape[0]):
            interpolated_f = interp1d(times,pt[state,:,sweep],fill_value='extrapolate') 
            upsampled_pt = interpolated_f(Tnew)
            resampled_pt[state,:,sweep] = upsampled_pt
    # using resampled p_t, get occs and the currents
    resampled_occs = resampled_pt*simulation_dict['conditions']['N']
    resampled_currs = np.zeros([len(simulation_dict['conditions']['Q']['conducting states'].values()),np.size(Tnew),pt.shape[2]])
    for item, value in enumerate(simulation_dict['conditions']['Q']['conducting states'].items()): # multiply conductance of each state by occupancy and drivign force
        state,conductance = value
        resampled_currs[item,:,:] = resampled_occs[state,:,:]*(((voltage-Vrev) *10**-3)*conductance)
    resampled_currs = np.nansum(resampled_currs,0)
    resampled_currs = resampled_currs*10**12
    if keepnoise == True:
        for item in np.arange(np.size(resampled_currs,1)):
            resampled_currs[:,item] = Add_Gaussian_noise(resampled_currs[:,item],noise_sd=simulation_dict['conditions']['noise_sd'])
        flag2='unchanged'
    elif keepnoise>0:
        resampled_currs = Add_Gaussian_noise(resampled_currs,noise_sd=keepnoise)
    if not update:
        return(Tnew,resampled_pt,resampled_occs,resampled_currs)
    else:
        if flag == 'newdriving':
            simulation_dict['conditions'].update({'voltage':voltage})
        if flag2 !='unchanged':
            simulation_dict['conditions'].update({'noise_sd':keepnoise})

        simulation_dict.update({'t':Tnew})
        simulation_dict.update({'p_t':resampled_pt})
        simulation_dict.update({'occ_t':resampled_occs})
        simulation_dict.update({'I_t':resampled_currs})
        # does not return when updated

def resample_current(current_dataframe,target_frequency):
    """
    resamples the current directly. For resampling occupancy and associated current,
    see resample_simulation

    """
    times = current_dataframe.index
    new_times = np.arange(0,times.max(),1/target_frequency)
    resampled = np.zeros([np.size(new_times),current_dataframe.shape[1]])
    for sweep in np.arange(current_dataframe.shape[1]):
        interpolated_f = interp1d(times,current_dataframe.iloc[:,sweep].to_numpy(),fill_value='extrapolate') 
        resampled[:,sweep] = interpolated_f(new_times)
    resampled_current = pd.DataFrame(resampled,index = new_times)
    
    return(resampled_current)




# =============================================================================
# =============================================================================
# # Code for stability analysis using Monte Carlo Simulation
# =============================================================================
# =============================================================================
import EPyPhys as epp
from sklearn.metrics import r2_score
import scipy.stats as sps
def mechanistic_analysis(path,align=True):
    """
    performs stabilitly analysis for the parameter estimates

    """
    current = load_sim(path)
    sweeps = current_to_DataFrame(current)
    # first, plot mean vs ensemble for one curr
    egfig,egaxs = plt.subplots()
    sweeps.plot(color='dimgrey',legend=False,ax=egaxs)
    sweeps.mean(axis=1).plot(color='midnightblue',ax=egaxs)
    #egaxs.set_xlim(right=0.025)
    plt.grid(False)
    egaxs.set_facecolor('white')
    epp.add_scalebar(egaxs)
    plt.tight_layout()
    
    # plot mean state occupancy over time
    occfig,occaxs = plt.subplots()
    occaxs.set_prop_cycle(mycycle)
    for state in np.arange(np.shape(current['p_t'])[0]):
        occaxs.plot(current['t'],np.mean(current['p_t'][state][:,:],axis=1),label = "{}".format(state))
    occaxs.legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
    occaxs.set_xlabel("t (s)")
    occaxs.set_ylabel("Probability")
    #plt.grid(False)
    #occaxs.set_facecolor("white")
    occaxs.spines['left'].set_color('black')
    occaxs.spines['bottom'].set_color('black')
    plt.tight_layout()

    # do the optimisation - aligned and unaligned. Though when sweepnumber enough, fine
    # have to assess the noise manually for myself and jot down values
    try:
        sweepno,sweepthresh,binno,binthresh = epp.NSFA_optimal(sweeps, alignment=align,optim_binsize=True,batchsize=16)
    except ValueError:
        sweepno,sweepthresh,binno,binthresh = np.nan,np.nan,np.nan,np.nan # catch for when none identified.
    return(sweepno,sweepthresh,binno,binthresh)
def mechanistic_analysis_2(path,n_bins,n_sweeps,align=True):
    """
    performs skew analysis NSFA using optimum parameters and retrieves the 
    ground truth Po-Peak as sum(Po) where |I| is max.
    """

    current = load_sim(path)
    sweeps = current_to_DataFrame(current)
    if np.isnan(n_bins):
        n_bins = 10 # catch
    if np.isnan(n_sweeps):
        n_sweeps = sweeps.shape[1] # catch
    N,i,Popen,y_mean,x,_,_,I_skew,var_skew = epp.NSFA(sweeps.iloc[:,:n_sweeps],num_bins=n_bins,skew_analysis=True,alignment=align,return_var=True)
    # get ground truth Po
    # get where current max
    locmax = np.where(np.mean(np.abs(current['I_t']),axis=1) == np.max(np.mean(np.abs(current['I_t']),axis=1)))[0][0]
    po_all = []
    for item in current['conditions']['Q']['conducting states'].keys():
        po_all.append(np.mean(current['p_t'][item],axis=1)[locmax]) # because not necessarily max in same location. Have to find where current is max
    total_po = np.sum(po_all)
    print("ground truth Po_peak was {}".format(np.round(total_po,4)))
    
    occs = []
    weighting = []
    curr = []
    
    for item, value in enumerate(current['conditions']['Q']['conducting states'].keys()):
        occs.append(current['p_t'].mean(axis=2)[value].sum())
    for item in occs:
        weighting.append(item/np.sum(occs))
    for item, value in enumerate(current['conditions']['Q']['conducting states'].keys()):
        curr.append(weighting[item]*current['conditions']['Q']['conducting states'][value])
    weighted_mean_cond = (np.sum(curr))*10**12 # convert to pS
    
    
    return(N,i,Popen,y_mean,x,I_skew,var_skew,total_po, weighted_mean_cond)

def r2_and_connect_thresholds(path,align=True,ensemble=False):
    """
    Performs mechanistic analysis 1 and 2 and 
    Generates the returns for a model's table': i.e. ensemble vs optimised, aligned/not
    And returns associated figures
    N, i, Po, g_mean, Po, Iskew,varskew, Po ground, weighted mean conductance ground, r2, bins, bin thresh, sweeps, sweepthresh
    sweepno,sweepthres,bi

    """
    if not ensemble:
        sweepno,sweepthres,binno,binthresh = mechanistic_analysis(path,align=align)
        N,i,Popen,y_mean,x,I_skew,var_skew,total_po,weighted_mean_cond = mechanistic_analysis_2(path, binno, sweepno,align=align)
    else:
        sweepno,sweepthres,binno,binthresh = 1000,np.nan,0,np.nan
        N,i,Popen,y_mean,x,I_skew,var_skew,total_po,weighted_mean_cond = mechanistic_analysis_2(path, 0, 1000,align=align)
    # use x to calculate r2
    r2 = r2_score(x[2,:]-x[1,:],epp.noise_func(x, N, i))
    if align:
        print("alignment was performed")
    return(N,i,Popen,y_mean,total_po,weighted_mean_cond,I_skew,var_skew,r2,sweepno,sweepthres,binno,binthresh)

def triplicate_analysis(path1,path2,path3,savepath):
    
    """
    This code is very slow and produces lots of graphs! want to remove the iterative selection element
    
    usage: takes all three simulation files and produces figures and tables
    after this run multigraph_my_figs() to save the figures as a single PDF
    
    For a triplicate of simulations, runs r2_and connect_threshold for all three simulations
    ensemble/optimised/aligned/unaligned
    
    and then saves them to the base folder
    
    probably 15 min runtime or so.
    
    Also calculates SEM where appropriate

    total error not calculated: R2 is more informative of our ability to capture the data
    
    Nb, due to file name formatting differences, paths input separately.

    N,i,Popen,y_mean,total_po,I_skew,var_skew,r2,sweepno,sweepthres,binno,binthresh
    """
    values = np.zeros([12,13])
    means = np.zeros([4,13])
    sems = np.zeros([4,13])

    #ensemble, unaligned
    values[0,:] = r2_and_connect_thresholds(path1,align=False,ensemble=True)
    values[1,:] = r2_and_connect_thresholds(path2,align=False,ensemble=True)
    values[2,:] = r2_and_connect_thresholds(path3,align=False,ensemble=True)
    #optimised, unaligned - try statements for whne fails.
    try:
        values[3,:] = r2_and_connect_thresholds(path1,align=False,ensemble=False)
    except Exception:#TypeError:
         values[3,:] = np.nan
    try:
        values[4,:] = r2_and_connect_thresholds(path2,align=False,ensemble=False)
    except Exception:#TypeError:
         values[4,:] = np.nan
    try:
        values[5,:] = r2_and_connect_thresholds(path3,align=False,ensemble=False)
    except Exception:#TypeError:
        values[5,:] = np.nan
    #ensemble,aligned
    values[6,:] = r2_and_connect_thresholds(path1,align=True,ensemble=True)
    values[7,:] = r2_and_connect_thresholds(path2,align=True,ensemble=True)
    values[8,:] = r2_and_connect_thresholds(path3,align=True,ensemble=True)
    #optimised, aligned
    try:
        values[9,:] = r2_and_connect_thresholds(path1,align=True,ensemble=False)
    except Exception:#TypeError:
        values[:,9] = np.nan
    try:
        values[10,:] = r2_and_connect_thresholds(path2,align=True,ensemble=False)
    except Exception:#TypeError:
        values[10,:] = np.nan
    try:
        values[11,:] = r2_and_connect_thresholds(path3,align=True,ensemble=False)
    except Exception:#TypeError:
        values[11,:] = np.nan
    # get means for each triplicate
    means[0,:] = np.nanmean(values[:3,:],axis=0)
    means[1,:] = np.nanmean(values[3:6,:],axis=0)
    means[2,:] = np.nanmean(values[6:9,:],axis=0)
    means[3,:] = np.nanmean(values[9:,:],axis=0)

    # get SEMS
    sems[0,:] = sps.sem(values[:3,:],axis=0)
    sems[1,:] = sps.sem(values[3:6,:],axis=0)
    sems[2,:] = sps.sem(values[6:9,:],axis=0)
    sems[3,:] = sps.sem(values[9:,:],axis=0)
    
    means = pd.DataFrame(means,columns = ['N','i','Po','y_mean','Po_grnd','g_grnd','I_skew','var_skew','r2','sweepno','sweepthres','binno','binthresh'])
    SEMs = pd.DataFrame(sems,columns = ['N','i','Po','y_mean','Po_grnd','g_grnd','I_skew','var_skew','r2','sweepno','sweepthres','binno','binthresh'])

    with open( "/".join(path1.split('/')[:-1]) + "/means",'wb') as handle:
       pickle.dump(means,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open( "/".join(path1.split('/')[:-1]) + "/sems",'wb') as handle:
       pickle.dump(SEMs,handle,protocol=pickle.HIGHEST_PROTOCOL)
    
    # close the figures to avoid ball ache
    for item in plt.get_fignums():
        #if item not in figlist: # can provide a global figlist
            plt.close(item)
    # but for tiff, have to save manually. Sigh - had hoped to use a dict of below.
   # Best just to keep track of all numbers
    return(means,SEMs)

def load_meanorsem(path):
    with open(path,'rb') as handle:
        inputs = pickle.load(handle)
    return(inputs)

# =============================================================================
# =============================================================================
#  Unused, but tested code for model fitting
# nB, expect NLP style deep learning probably would work better
# and methods for hidden markov construction cna work acceptable in less complex model
# =============================================================================
# =============================================================================

def model_constraints(model,constraint_matrix, fix_relationships=False):
    """
    Takes a model and constraint matrix to apply the following constraints:
        1. Allow some rates to be set (not modified by MR)
        2. Allow some rates to be set by MR
        
    A constraint matrix should be a square k x k array for k states, with entries:
        0.5: fixed
        1: to be set by MR
    NB: concentration-dependent rates should be 'fixed'
    
    For easy creation of the constraint matrix:
        constraint_matrix = np.zeros_like(model['rates'])
        constraint_matrix.fill(0.5) - alternatively fill with 1
        constraint_matrix[edgefrom,edgeto] = 1 - to set edgefrom,edgeto to be set by MR
    """
    Q = model['Q']
    logic_graph = nx.MultiDiGraph(constraint_matrix)
    for edge in logic_graph.edges():
        # rates to be set by microscopic reversibility have weight 1, set have weight 0.5
        if logic_graph.get_edge_data(edge[0],edge[1])[0]['weight'] ==1: # rate ij = rateji*(product rates of forward route/product rates of reverse route)
            # where item[0] is source and itme[1] is target, getting shortest forward and reverse paths
            forward_path = nx.shortest_path(logic_graph,edge[0],edge[1])
            reverse_path = nx.shortest_path(logic_graph,edge[1],edge[0])
            # getting product of rates along this path using embedded function
            forward_rates_prod = path_rates(forward_path,Q)
            reverse_rates_prod = path_rates(reverse_path,Q)
            Q[edge[0],edge[1]] = Q[edge[1],edge[0]] * (np.divide(forward_rates_prod,reverse_rates_prod, where = (reverse_rates_prod>0)))
    for row in range(0,np.size(Q,axis=0)): # Q matrix to convention - used by Q matrix method / CME
            Q[row,row] = - np.nansum(Q[row])
    model.update({'Q':Q})
    rates = model['Q'].copy()
    for row in range(0,np.size(rates,axis=0)): # generator convention
            rates[row,row] = np.nan
    rates[rates==0] = np.nan
    model.update({'rates':rates})
    return()

def firefly_fit(data,Q,target_rates = False,baseline_length = 100*10**-3,parameter_spread = 1000,firefly_steps = 100,fix_relationships=False,**kwargs):
    """
    Alternative;y, see fittigration method below:
    
    For fittign stochastic models to data based on current
    
    Performs unsatisfactorily so wasnt used. Specifically, either diverged from data when the number of fireflies was low
    (bestf ly pulled towards others), or when nubmer of fleis was large, swarm behaviour was better, local minima was hit and convergence did not occur.
    
    Varying behaviour of best fly may improve.
    Alternative methods to constrain best fly were tried for best fly:
        #   constraining its motion to be no further away from the data than the mean
    
        # movign randomly.
        
        # as well as placing best fly in superposition, mvoing towards or away from average flies, evaulating which is closer to data, and then taking that as best fly.
    
    Uses qs.simulate to fit a kinetic model to some data.
    Data can either be simulated itself, or can be experimental
        but should have a baseline at start
    
    data: either dataframe or simulation_dict object (as by qs.simulate)
    Q: the model dict to use
    target_rates: the rates to vary. If False (Default), then all rates may be varied (except diag)
        Otherwise, a kxk matrix should be provided where:
            0.5 is a rate that can be varied
            1 is a rate to be set by MR
            2 is a rate that is set within Q that cannot be varied
        - For easy creation:
            target_rates = zeros_like(Q['Q'])
            and set as desired.
             
    baseline_length: length of the baseline from t=0, from which to determine how to layer noise onto the current
    parameter_spread: determiens the number of uniform samples for each rate constant. The larger thsi value is, the slower the process will take, but mimima are more likely to be identified.
    **kwargs: arguments for qs.simulate, which should attempt to match the profile of the current in data (i.e. in terms of when, and hwo long agonist_application is)
        - does not need to include 'N' since currents are 
    
    fix_relationships (Bool): Default False. When True, accesses the relationship matrix
        from the model dict and uses that to contrain relationships
        --- see GluCoo17() model as an example of the format.
        In brief:
            allows relationships to be forced between constants.
                e.g. setting rate 2,3 to a multiple of rate 1,2
            The creation is fairly labourious, but allows convergence far more readily in model fitting.
            it must be given as a k x k x 2 matrix, where dimensions 0 and 1 are one-hot encodings
            of relationships in [k,k,0], and [k,k,1] contains mutliples.
                To create k,k,1:
                    kk_1 = np.ones_like(model['Q'])
                    then assign mutliples - e.g. if rate 3,4 is 2 x rate 2,3,
                    kk_1[3,4] = 2
                    etc.
                -To create kk_0:
                    kk_0 = np.zeros_like(model['Q'])
                    and the corresponding entries for 3,4 and 2,3 are arbitraliy 469
                    kk_0[3,4], kk_0[2,3] = 469
    
    This procedure should ideally be performed multiple times, since the rates used are sampled from a uniform distribution
    and different rates might converge to different minima (local or global)
    """
    if type(data) != pd.core.frame.DataFrame:
        try:
            data = current_to_DataFrame(data)
        except Exception:
            print("data must be a pandas DataFrame or simulation dict object")
            return()
    real_t_final = copy.copy(kwargs['t_final']) # used later - might need to do with other kwargs

    # get noise amplitude to layer on
    noise_amp = data.loc[data.index<baseline_length,:].abs().mean().mean()
    # get data sampling rate and length
    data_length = data.index.max()
    data_sampling_rate = 1/data.index[1]
    #normalise data - not true normalisation as still allows current of other sign
    norm_data = data/(data.abs().max().max())
    # norm data truncated for initial optimisation
    ag_time = kwargs['agonist_time']
    norm_data_trunc = (norm_data.loc[norm_data.index])#<ag_time+(data_length/5),:])
    # want to resample the current so that timepoints correspond to sims below
        # otherwise euclidean distance is exaggerated.
    #norm_data_trunc = resample_current(norm_data_trunc,1/1e-04) 
    # resampling useful when examine means if both resampled to same frequency

    # changed from resampling current to a cut operation
    #timepoints for resample
    resample_bins = pd.cut(norm_data_trunc.index,np.arange(0,real_t_final,5e-05))
    norm_data_trunc = norm_data_trunc.groupby(resample_bins).mean()
    norm_data_trunc.index = np.arange(0,real_t_final,5e-05)[:-1]
    # post agonist current
    p_a_norm_current = norm_data_trunc[norm_data_trunc.index>=ag_time]

    model_dict= Q.copy()
    if not np.any(target_rates):
        target_rates = np.zeros_like(Q['rates'])
        target_rates.fill(0.5) # all rates may be varied
        #np.fill_diagonal(target_rates,0)
    # for all rates set as 1 (i.e. that can be varied)
        # since the fastest sensible rate constant would be 100,000 (i.e. taking 1 sample at 100kHz to complete)
        # for each rate that can vary, generate random numbers form uniform distribution (0,100000)

    # generate the 1000 rate matrices, including the random rates
        # if fix relationships, the proportion of the rates is maintained. This is more effieicnt, since it reduces random number generation.
    if not fix_relationships:
        rate_matrices = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0),parameter_spread])
        for row in np.arange(np.size(target_rates,0)):
            for column in np.arange(np.size(target_rates,1)):
                if target_rates[row,column] == 0.5: 
                    #fill with random numbers if not set
                    rate_matrices[row,column,:] = np.random.uniform(0,150000,parameter_spread)
                else: #if set, fill with the set value
                    rate_matrices[row,column,:] = Q['rates'][row,column]
    else:
        # find the number of unique relationships to vary together
        # and the nubmer of unique rates to vary individually
        if 'relationships' not in model_dict.keys():
            print("Relationships array must be encoded in model. See help(model_constraints)")
            return()
        else:
            counting = 0
            rate_matrices = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0),parameter_spread])
            kk = model_dict['relationships']
            encodings = np.unique(kk[:,:,0]) # get all unique one-hot encodings
            encodings = np.delete(encodings,0) # remove the ones that can be varied
            # get the number that can be individually varied
                # which is nubmer of zeros in kk[:,:,0] - number of zero and negatives in model['Q']
            # get number of individual constants that are additionally varied
            # together, these are the constants that vary individually
            n_to_change = len(encodings) # number of unique relationship encodings
            # do single random draw for each relationship encoding and enforce multiples
            for encoding in encodings:
                random_draw = np.random.uniform(0,150000,parameter_spread)
                relationships = np.stack((np.where(kk[:,:,0] == encoding))) # zipping where each col gives indexers
                for relation in np.arange(np.shape(relationships)[1]):
                    if target_rates[relationships[0,relation],relationships[1,relation]]==0.5: # if rate not set by user
                        rate_matrices[relationships[0,relation],relationships[1,relation],:] =  random_draw * kk[relationships[0,relation],relationships[1,relation],1]
                    else:
                        rate_matrices[relationships[0,relation],relationships[1,relation],:] = Q['rates'][relationships[0,relation],relationships[1,relation]]
            # find rates that vary individually and do necessary additional draws if they are not set
            remaining_to_vary = np.stack((np.where((kk[:,:,0] == 0)& (model_dict['Q']>0))))
            for item in np.arange(np.shape(remaining_to_vary)[1]):
                if target_rates[remaining_to_vary[0,item],remaining_to_vary[1,item]] == 0.5:
                    rate_matrices[remaining_to_vary[0,item],remaining_to_vary[1,item],:] = np.random.uniform(0,100000,parameter_spread)
                    counting = counting+1
                else:
                    rate_matrices[remaining_to_vary[0,item],remaining_to_vary[1,item],:] = Q['rates'][remaining_to_vary[0,item],remaining_to_vary[1,item]]
    # create the constraint matrix
    constraint_matrix = copy.copy(target_rates)
    # addition - so that non-real rates are 0.5 (i.e. set to value of 0)
    constraint_matrix[Q['rates']==0]=0.5
    
    constraint_matrix[constraint_matrix == 2] = 0.5
    np.fill_diagonal(constraint_matrix,0.5) # timesaver
    params_fitting = {}
    distances = [] # for stacking
    print("Placing fireflies to identify minima for {} individually varying and {} sets of proportionally varying rates".format(counting,n_to_change))
    for iteration in tqdm(np.arange(parameter_spread)):
        model_iteration = model_dict.copy()
        model_iteration.update({'Q':rate_matrices[:,:,iteration]})
        # enforces MR in Q and updates the rates to match Q, enforcing necessary conventions
        model_constraints(model_iteration, constraint_matrix)
        # simulate using that model for first 10th of time with matching noise_amplitude layered over
        # this should give some more power to rise times of the current, which tend to be faster, but also aids in efficiency
        kwargs['noise_sd'] = noise_amp
        kwargs['t_final'] = real_t_final
        kwargs['n_sweeps'] = 10
        kwargs['interval'] = 5e-05
        kwargs['N'] = 1000
        kwargs['Q'] = model_iteration
        iteration_output = simulate(show_progress=False,graph=False,**kwargs) # calling main simulate function
        #plt.close('all')
        iteration_current = current_to_DataFrame(iteration_output)
        # normalise the current
        norm_iteration_current = (iteration_current/(iteration_current.abs().max().max()))
        # resample it to match that of the truncated data
        # changed to a cut operation
        #norm_iteration_current = resample_current(norm_iteration_current,1/1e-04)
        bins_curr = pd.cut(norm_iteration_current.index,np.arange(0,real_t_final,5e-05))
        norm_iteration_current = norm_iteration_current.groupby(bins_curr).mean()
        norm_iteration_current.index = np.arange(0,real_t_final,5e-05)[:-1]
        # for rmse and L2 calculation, perform on the current AFTER baseline
            # so that are more sensitive
        p_a_iteration_current = norm_iteration_current[norm_iteration_current.index>=ag_time]
        #rmse = mean_squared_error(p_a_norm_current, p_a_iteration_current,squared=True)
        euc_distance = np.linalg.norm(p_a_norm_current.mean()-p_a_iteration_current.mean())
        #if euc_distance < 0.1:
            # a plausible current at this point, model should be less than 10% different in mean to the mean of the current during n
        that_iteration = model_iteration.copy()
        that_iteration.update({'distance':euc_distance})
        #that_iteration.update({'rmse':rmse})
        that_iteration.update({'model':iteration_output})
        params_fitting.update({str(iteration):that_iteration})
        distances.append(euc_distance)
    print("Performing firefly steps. Runtime approximately firefly_steps * time for initial placement")
    # do something more efficient for each firefly step - could be a while loop
    brightest_fly = []
    best_fly = {}
    mean_luminance =[]
    max_luminance = []
    for step in tqdm(np.arange(firefly_steps)):
        # get somenew random rate matrices by the same rules as previously
        if not fix_relationships:
            random_rates = copy.copy(rate_matrices) # each entry in dimension 2 is a firefly
            for row in np.arange(np.size(target_rates,0)):
                for column in np.arange(np.size(target_rates,1)):
                    if target_rates[row,column] == 0.5: 
                        #fill with random numbers if not set
                        random_rates[row,column,:] = np.random.uniform(0,150000,parameter_spread)
                    else: #if set, fill with the set value
                        random_rates[row,column,:] = Q['rates'][row,column]
        else:
            #instances of no specified relationships will be caught earlier
            random_rates = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0),parameter_spread]) # one for each firefly in second dimension
            kk = model_dict['relationships']
            encodings = np.unique(kk[:,:,0]) # get all unique one-hot encodings
            encodings = np.delete(encodings,0) 
            for encoding in encodings:
                random_draw = np.random.uniform(0,150000,parameter_spread)
                relationships = np.stack((np.where(kk[:,:,0] == encoding))) # zipping where each col gives indexers
                for relation in np.arange(np.shape(relationships)[1]):
                    if target_rates[relationships[0,relation],relationships[1,relation]]==0.5: # if rate not set by user
                        random_rates[relationships[0,relation],relationships[1,relation],:] =  random_draw * kk[relationships[0,relation],relationships[1,relation],1]
                    else:
                        random_rates[relationships[0,relation],relationships[1,relation],:] = Q['rates'][relationships[0,relation],relationships[1,relation]]
            # find rates that vary individually and do necessary additional draws if they are not set
            remaining_to_vary = np.stack((np.where((kk[:,:,0] == 0)& (model_dict['Q']>0))))
            for item in np.arange(np.shape(remaining_to_vary)[1]):
                if target_rates[remaining_to_vary[0,item],remaining_to_vary[1,item]] == 0.5:
                    random_rates[remaining_to_vary[0,item],remaining_to_vary[1,item],:] = np.random.uniform(0,100000,parameter_spread)
                else:
                    random_rates[remaining_to_vary[0,item],remaining_to_vary[1,item],:] = Q['rates'][remaining_to_vary[0,item],remaining_to_vary[1,item]]
            # random rates is fine

    ###### ---- firefly steps
        dists = distances
        # define luminance for each firefly as 1/1+(distance from data^2)
        luminances = 1/(1+(np.array(dists)**2)) # Tilahum and Choon-Ong point out that this is faster than using exponential term
        mean_luminance.append(luminances.mean())
        max_luminance.append(np.max(luminances))
        if fix_relationships: # annoying to have here and then repeated
            n_dimensions = len(encodings) + np.size(remaining_to_vary,1)
            relationship_rates = np.zeros([n_dimensions,parameter_spread])
    
        # for each firefly, rates for pairwise distances
        for firefly in np.arange(parameter_spread):
            if fix_relationships:
                # perform dimensionality reduction for fireflies
                # make it a vector in n_dimensions for number of unique rates
                for item, encoding in enumerate(encodings):
                    relationships = np.stack((np.where(kk[:,:,0] == encoding))) # zipping where each col gives indexers
                    relationship_rates[item,firefly] = random_rates[relationships[0,0],relationships[1,0],firefly]/(kk[relationships[0,0],relationships[0,1],1]) # get base, unmultiplied rate
                for item in np.arange(np.size(remaining_to_vary,1)):
                    relationship_rates[item+len(encodings),firefly] = random_rates[remaining_to_vary[0,item],remaining_to_vary[1,item],firefly]

                # distances remains the same, but pairwise does not
        pairwise_distances = np.zeros([parameter_spread,parameter_spread])
        #if fix_relationships:
        #    pairwise_distances = np.zeros([parameter_spread,parameter_spread])
        #    for m1,m2 in combinations(np.arange(parameter_spread),2):
                # get pseudo-symmetric matrix of distances between rate matrices
        #        pairwise_distances[m1,m2] = np.linalg.norm(relationship_rates[:,m1] - relationship_rates[:,m2])
        #        pairwise_distances[m2,m1] = np.linalg.norm(relationship_rates[:,m2] - relationship_rates[:,m1])
        #else:
        for m1,m2 in combinations(np.arange(parameter_spread),2):
            # get pseudo-symmetric matrix of distances between rate matrices
            pairwise_distances[m1,m2] = np.linalg.norm(rate_matrices[:,:,m1] - rate_matrices[:,:,m2])
            pairwise_distances[m2,m1] = np.linalg.norm(rate_matrices[:,:,m2] - rate_matrices[:,:,m1])

    # define attraction of each fly, which is proportional to the distance between them * their distance from origin (brightness)
        attraction = pairwise_distances*luminances #where attraction is luminance[item]*pairwise_distances[row,column]
        # move each firefly towards brightest it can see - except the brightest fly
        for firefly in np.arange(parameter_spread):
            if luminances[firefly] == np.max(luminances): # if closest fit / brightest
                brightest_fly.append(firefly)
                # and then have to get the correct direction to move it
                    # which annoyingly is a property of all independent rates
                        # could duplicate fly and have move both towards and away
                            # and then cull the one that is shitter when update the system
                        # i'm chill with that.
                # for best firefly, superpose it as moving in both directions
                best_fly_neg = rate_matrices[:,:,firefly] - ((np.multiply(np.random.rand(),random_rates[:,:,firefly])))
                best_fly_pos = rate_matrices[:,:,firefly] + ((np.multiply(np.random.rand(),random_rates[:,:,firefly])))
                
            else:
                signs = np.subtract(rate_matrices[:,:,np.where(attraction[firefly,:]==np.max(attraction[firefly,:]))[0][0]],rate_matrices[:,:,firefly])
                signs[signs<0] = -1
                signs[signs>0] = 1
                signs[Q['Q']==0]=0 # fix for non-real rates, hopefully
                # get direction to move it towards the brightest fly it perceives
                # move the fly towards its most attractive partner
                
                # could just have a minus term instead for the best fly
                    # make anything < 0
                    # how do we deal with that: make 1? (not zero though) because we know they are real rates
                    
                # idea would be to add the distance to all rates corresponding to that relationship
                    # e.g. + kk[1] *i , where kk1 is the indiciator
                            # where a = random (0,1) * signs, and e = random_rates
                rate_matrices[:,:,firefly] = rate_matrices[:,:,firefly] + (attraction[firefly,np.where(attraction[firefly,:]==np.max(attraction[firefly,:]))[0][0]]) + (np.multiply(np.multiply(np.random.rand(),signs),random_rates[:,:,firefly]))
                rate_matrices[:,:,firefly][rate_matrices[:,:,firefly]<0] = 1 # make 1, not zero
                rate_matrices[:,:,firefly][[Q['Q']==0]]=0 # fixing change to non-real rates
        best_fly_neg[best_fly_neg<0] =1
        best_fly_neg[[Q['Q']==0]]=0
        best_fly_pos[best_fly_pos<0] =1
        best_fly_pos[[Q['Q']==0]]=0
        # need to remember to find better of neg or pos firefly and replace it in rate matrices
            # above, do both as single style, since attraction is single value
            # but then new rates of a given firefly must be multiplied out for fixed, but not for unfixed

            # new position = old position + attraction + sign-adjusted random number * random rates
            
            
            # place limit on sum(relatinships) =  for max(kk[:,:,1] * rate = 150,000)
            # would also be good to move the best firefly only in a better direction
            
            # guess will have to have rate-by-rate refinement too.
            
            # then could do with var after to illustrate point about hwo rates woule
                # have to change / whether they could
            
         
    # re-perform the simulations
        #distances = []
        for firefly in np.arange(parameter_spread):
            if firefly == 0:
                distances = []
            model_iteration = model_dict.copy()
            model_iteration.update({'Q':rate_matrices[:,:,firefly]})
            if firefly == brightest_fly[-1]:
                #--- pos best fly
                model_iteration.update({'Q':best_fly_pos})
                # enforces MR in Q and updates the rates to match Q, enforcing necessary conventions
                model_constraints(model_iteration, constraint_matrix)
                # simulate using that model for first 10th of time with matching noise_amplitude layered over
                # this should give some more power to rise times of the current, which tend to be faster, but also aids in efficiency
                kwargs['noise_sd'] = noise_amp
                kwargs['t_final'] = real_t_final
                kwargs['n_sweeps'] = 10
                kwargs['interval'] = 5e-05
                kwargs['N'] = 1000
                kwargs['Q'] = model_iteration
                iteration_output_pos = simulate(show_progress=False,graph=False,**kwargs) # c
                #if firefly == np.where(luminances == luminances.max())[0][0]:
                #    best_fly[:,:,step] = iteration_output['conditions']['Q']['Q']
                
                iteration_current_pos = current_to_DataFrame(iteration_output)
                # normalise the current
                norm_iteration_current_pos = (iteration_current/(iteration_current.abs().max().max()))
                # resample it to match that of the truncated data
                # changed to a cut operation
                #norm_iteration_current = resample_current(norm_iteration_current,1/1e-04)
                bins_curr_pos = pd.cut(norm_iteration_current_pos.index,np.arange(0,real_t_final,5e-05))
                norm_iteration_current_pos = norm_iteration_current_pos.groupby(bins_curr_pos).mean()
                norm_iteration_current_pos.index = np.arange(0,real_t_final,5e-05)[:-1]
                # for rmse and L2 calculation, perform on the current AFTER baseline
                    # so that are more sensitive
                p_a_iteration_current_pos = norm_iteration_current_pos[norm_iteration_current_pos.index>=ag_time]
                #rmse = mean_squared_error(p_a_norm_current, p_a_iteration_current,squared=True)
                euc_distance_pos = np.linalg.norm(p_a_norm_current.mean()-p_a_iteration_current_pos.mean())
                #--- neg best fly
                model_iteration.update({'Q':best_fly_neg})
                model_constraints(model_iteration, constraint_matrix)
                # simulate using that model for first 10th of time with matching noise_amplitude layered over
                # this should give some more power to rise times of the current, which tend to be faster, but also aids in efficiency
                kwargs['noise_sd'] = noise_amp
                kwargs['t_final'] = real_t_final
                kwargs['n_sweeps'] = 10
                kwargs['interval'] = 5e-05
                kwargs['N'] = 1000
                kwargs['Q'] = model_iteration
                iteration_output = simulate(show_progress=False,graph=False,**kwargs) # c
                #if firefly == np.where(luminances == luminances.max())[0][0]:
                #    best_fly[:,:,step] = iteration_output['conditions']['Q']['Q']
                
                iteration_current = current_to_DataFrame(iteration_output)
                # normalise the current
                norm_iteration_current = (iteration_current/(iteration_current.abs().max().max()))
                # resample it to match that of the truncated data
                # changed to a cut operation
                #norm_iteration_current = resample_current(norm_iteration_current,1/1e-04)
                bins_curr = pd.cut(norm_iteration_current.index,np.arange(0,real_t_final,5e-05))
                norm_iteration_current = norm_iteration_current.groupby(bins_curr).mean()
                norm_iteration_current.index = np.arange(0,real_t_final,5e-05)[:-1]
                # for rmse and L2 calculation, perform on the current AFTER baseline
                    # so that are more sensitive
                p_a_iteration_current = norm_iteration_current[norm_iteration_current.index>=ag_time]
                #rmse = mean_squared_error(p_a_norm_current, p_a_iteration_current,squared=True)
                euc_distance_neg = np.linalg.norm(p_a_norm_current.mean()-p_a_iteration_current.mean()) 
                if euc_distance_neg<euc_distance_pos:
                    ## here, just abotu to update the distances and wdecide which is best model
                    # then need to give the instance for other flies.
                    best_fly.update({str(step):iteration_output})
                    distances.append(euc_distance)
                    rate_matrices[:,:,firefly] = best_fly_neg
                else:
                    best_fly.update({str(step):iteration_output_pos})
                    distances.append(euc_distance_pos)
                    rate_matrices[:,:,firefly] = best_fly_pos
            else:
                model_iteration = model_dict.copy()
                model_iteration.update({'Q':rate_matrices[:,:,firefly]})
                # enforces MR in Q and updates the rates to match Q, enforcing necessary conventions
                model_constraints(model_iteration, constraint_matrix)
                # simulate using that model for first 10th of time with matching noise_amplitude layered over
                # this should give some more power to rise times of the current, which tend to be faster, but also aids in efficiency
                kwargs['noise_sd'] = noise_amp
                kwargs['t_final'] = real_t_final
                kwargs['n_sweeps'] = 10
                kwargs['interval'] = 5e-05
                kwargs['N'] = 1000
                kwargs['Q'] = model_iteration
    
                iteration_output = simulate(show_progress=False,graph=False,**kwargs) # c
                # if firefly == np.where(luminances == luminances.max())[0][0]:
                #     best_fly[:,:,step] = iteration_output['conditions']['Q']['Q'][:,:,]
                
                
                iteration_current = current_to_DataFrame(iteration_output)
                # normalise the current
                norm_iteration_current = (iteration_current/(iteration_current.abs().max().max()))
                # resample it to match that of the truncated data
                # changed to a cut operation
                #norm_iteration_current = resample_current(norm_iteration_current,1/1e-04)
                bins_curr = pd.cut(norm_iteration_current.index,np.arange(0,real_t_final,5e-05))
                norm_iteration_current = norm_iteration_current.groupby(bins_curr).mean()
                norm_iteration_current.index = np.arange(0,real_t_final,5e-05)[:-1]
                # for rmse and L2 calculation, perform on the current AFTER baseline
                    # so that are more sensitive
                p_a_iteration_current = norm_iteration_current[norm_iteration_current.index>=ag_time]
                #rmse = mean_squared_error(p_a_norm_current, p_a_iteration_current,squared=True)
                euc_distance = np.linalg.norm(p_a_norm_current.mean()-p_a_iteration_current.mean())
                #update distances
                distances.append(euc_distance)
            dists = distances
            if step == firefly_steps: # following final update, calculate final
                luminances = 1/(1+(np.array(dists)**2)) # Tilahum and Choon-Ong point out that this is faster thanusing exponential term
                mean_luminance.append(luminances.mean())
                max_luminance.append(np.max(luminances))
            #update distances
            #distances.append(euc_distance)
    return(mean_luminance,max_luminance,best_fly,brightest_fly)

def fittigration(data,Q,target_rates = False,baseline_length = 100*10**-3,parameter_spread = 1000,steps = 100,fix_relationships=False,max_rate = 1.5e5,**kwargs):
    """
    
    Using stochastic simulations to fit model to data.
    Was not satisfied with results, so unused. Varying model averaging and convergence behaviour may improve.
        # or using Q matrix method - but then cannot asses variance.
    Mya perform well for simpler models (I tried with 17 dimensions - i.e. rate constants/ goruped rate constants that may change)
    
    Idea is to integrate until time t, so get a value of goodness of fit for each time point
    Then to associate each rate with which integrals it's most associated
    Which means varying one at a time, or groups
    
    Using evolutioanry process to fit rates of specified mechanism using area
    under curve at time t (where AUC obtained by trapezium rule), and minimising the 
    distance between the model fit  and the data using the differences of their AUC
    
    We thought this approach woudl allow us to account for non-stationary nature of the chain
    - i.e. that some time cosntants contribute minimally at certain I(t). And avoid sequence-dependent problems
    associated with varyign rates/grouped rates individually
    
    Microscopic reversibility is imposed, and normalised currents are used.
    
    Args::
    Data =dataframe or model dict of the data to which a model is fitted
    Q = the dictionary containiing the relationships and rates/ Q matrix of the model - structed as included examples
        # thsi can contian relationships between rates that facilitate constraints
    target_rates (optional): matrix specifyign which rates are set by user, which canbe varied, and whicch should be set by MR
    parameter_spread: the number of models to run and refine at each iteration
    steps: the number of iterations
    fix_relationships: when true and a relationships matrix provided (see GluCoo17 for example), maintains relationships between rate constants
    max_rate: the maximum rate allowed for a given rate cosntant. Here  = 150,000s^-1
    ** kwargs: fir the simulation. Should match the data in tersm fo when agonist applied, decya/rise time of stimulus, concentration of agonist etc.
    
    
    """
    if type(data) != pd.core.frame.DataFrame:
        try:
            data = current_to_DataFrame(data)
        except Exception:
            print("data must be a pandas DataFrame or simulation dict object")
            return()
    real_t_final = copy.copy(kwargs['t_final']) # used later - might need to do with other kwargs

    # get noise amplitude to layer on
    noise_amp = data.loc[data.index<baseline_length,:].abs().mean().mean()
    # get data sampling rate and length
    data_length = data.index.max()
    data_sampling_rate = 1/data.index[1]
    #normalise data - not true normalisation as still allows current of other sign
    norm_data = data/(data.abs().max().max())
    # norm data truncated for initial optimisation
    ag_time = kwargs['agonist_time']
    norm_data_trunc = (norm_data.loc[norm_data.index])#<ag_time+(data_length/5),:])
    # want to resample the current so that timepoints correspond to sims below
        # otherwise euclidean distance is exaggerated.
    #norm_data_trunc = resample_current(norm_data_trunc,1/1e-04) 
    # resampling useful when examine means if both resampled to same frequency

    # changed from resampling current to a cut operation
    #timepoints for resample
    resample_bins = pd.cut(norm_data_trunc.index,np.arange(0,real_t_final,5e-05))
    norm_data_trunc = norm_data_trunc.groupby(resample_bins).mean()
    norm_data_trunc.index = np.arange(0,real_t_final,5e-05)[:-1]
    # post agonist current
    p_a_norm_current = norm_data_trunc[norm_data_trunc.index>=ag_time]
    #integrate to time t
    data_int_norm = trapesium_t(p_a_norm_current)

    model_dict= Q.copy()
    if not np.any(target_rates):
        target_rates = np.zeros_like(Q['rates'])
        target_rates.fill(0.5) # all rates may be varied
        #np.fill_diagonal(target_rates,0)
    # for all rates set as 1 (i.e. that can be varied)
        # since the fastest sensible rate constant would be 100,000 (i.e. taking 1 sample at 100kHz to complete)
        # for each rate that can vary, generate random numbers form uniform distribution (0,100000)

    # generate the 1000 rate matrices, including the random rates
        # if fix relationships, the proportion of the rates is maintained. This is more effieicnt, since it reduces random number generation.
    if not fix_relationships:
        rate_matrices = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0),parameter_spread])
        for row in np.arange(np.size(target_rates,0)):
            for column in np.arange(np.size(target_rates,1)):
                if target_rates[row,column] == 0.5: 
                    #fill with random numbers if not set
                    rate_matrices[row,column,:] = np.random.uniform(0,max_rate,parameter_spread)
                else: #if set, fill with the set value
                    rate_matrices[row,column,:] = Q['rates'][row,column]
        # generate list of rate indexers - too add
    else:
        # find the number of unique relationships to vary together
        # and the nubmer of unique rates to vary individually
        if 'relationships' not in model_dict.keys():
            print("Relationships array must be encoded in model. See help(model_constraints)")
            return()
        else:
            counting = 0
            rate_matrices = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],0),parameter_spread])
            kk = model_dict['relationships']
            encodings = np.unique(kk[:,:,0]) # get all unique one-hot encodings
            encodings = np.delete(encodings,0) # remove the ones that can be varied
            # get the number that can be individually varied
                # which is nubmer of zeros in kk[:,:,0] - number of zero and negatives in model['Q']
            # get number of individual constants that are additionally varied
            # together, these are the constants that vary individually
            n_to_change = len(encodings) # number of unique relationship encodings
            # do single random draw for each relationship encoding and enforce multiples
            
            rate_indexers =[]
            
            for encoding in encodings:
                random_draw = np.random.uniform(0,max_rate,parameter_spread)
                relationships = np.stack((np.where(kk[:,:,0] == encoding))) # zipping where each col gives indexer
                # getting a single rate indexer for each encoding - used later
                rate_indexers.append(np.append(np.append(encoding,relationships[:,0]),kk[relationships[0,1],relationships[1,1],1]).astype(int))
                for relation in np.arange(np.shape(relationships)[1]):
                    if target_rates[relationships[0,relation],relationships[1,relation]]==0.5: # if rate not set by user
                        rate_matrices[relationships[0,relation],relationships[1,relation],:] =  random_draw * kk[relationships[0,relation],relationships[1,relation],1]
                    else:
                        rate_matrices[relationships[0,relation],relationships[1,relation],:] = Q['rates'][relationships[0,relation],relationships[1,relation]]
                    
            # find rates that vary individually and do necessary additional draws if they are not set
            remaining_to_vary = np.stack((np.where((kk[:,:,0] == 0)& (model_dict['Q']>0))))
            for item in np.arange(np.shape(remaining_to_vary)[1]):
                rate_indexers.append(np.append(np.append(0,remaining_to_vary[:,item]),1).astype(int)) # given encoding of 0 in list of rate indexers
                if target_rates[remaining_to_vary[0,item],remaining_to_vary[1,item]] == 0.5:
                    rate_matrices[remaining_to_vary[0,item],remaining_to_vary[1,item],:] = np.random.uniform(0,max_rate,parameter_spread)
                    counting = counting+1
                else:
                    rate_matrices[remaining_to_vary[0,item],remaining_to_vary[1,item],:] = Q['rates'][remaining_to_vary[0,item],remaining_to_vary[1,item]]
    # create the constraint matrix
    constraint_matrix = copy.copy(target_rates)
    # addition - so that non-real rates are 0.5 (i.e. set to value of 0)
    constraint_matrix[Q['rates']==0]=0.5
        
    constraint_matrix[constraint_matrix == 2] = 0.5
    np.fill_diagonal(constraint_matrix,0.5) # timesaver
    params_fitting = {}
    differences = np.zeros([parameter_spread,np.size(p_a_norm_current,0)])
    print("Initialising models to identify minima for {} individually varying and {} sets of proportionally varying rates".format(counting,n_to_change))
    for iteration in tqdm(np.arange(parameter_spread)):
        model_iteration = model_dict.copy()
        model_iteration.update({'Q':rate_matrices[:,:,iteration]})
        # enforces MR in Q and updates the rates to match Q, enforcing necessary conventions
        model_constraints(model_iteration, constraint_matrix)
        # simulate using that model for first 10th of time with matching noise_amplitude layered over
        # this should give some more power to rise times of the current, which tend to be faster, but also aids in efficiency
        kwargs['noise_sd'] = noise_amp
        kwargs['t_final'] = real_t_final
        kwargs['n_sweeps'] = 10
        kwargs['interval'] = 5e-05
        kwargs['N'] = 1000
        kwargs['Q'] = model_iteration
        iteration_output = simulate(show_progress=False,graph=False,**kwargs) # calling main simulate function
        #plt.close('all')
        iteration_current = current_to_DataFrame(iteration_output)
        # normalise the current to its maximum (i.e. of all sweeps)
        norm_iteration_current = (iteration_current/(iteration_current.abs().max().max()))
        # resample it to match that of the truncated data
        # changed to a cut operation
        #norm_iteration_current = resample_current(norm_iteration_current,1/1e-04)
        bins_curr = pd.cut(norm_iteration_current.index,np.arange(0,real_t_final,5e-05))
        norm_iteration_current = norm_iteration_current.groupby(bins_curr).mean()
        norm_iteration_current.index = np.arange(0,real_t_final,5e-05)[:-1]
        # for rmse and L2 calculation, perform on the current AFTER baseline
            # so that are more sensitive
        p_a_iteration_current = norm_iteration_current[norm_iteration_current.index>=ag_time]
        #rmse = mean_squared_error(p_a_norm_current, p_a_iteration_current,squared=True)
        #euc_distance = np.linalg.norm(p_a_norm_current.mean()-p_a_iteration_current.mean())
        p_a_int_norm_model = trapesium_t(p_a_iteration_current)
        p_a_int_diff = data_int_norm - p_a_int_norm_model
        #if euc_distance < 0.1:
            # a plausible current at this point, model should be less than 10% different in mean to the mean of the current during n
        that_iteration = model_iteration.copy()
        that_iteration.update({'int_diff':p_a_int_diff})
        #that_iteration.update({'distance':euc_distance})
        #that_iteration.update({'rmse':rmse})
        that_iteration.update({'model':iteration_output})
        params_fitting.update({iteration:that_iteration})
        #distances.append(euc_distance)
        differences[iteration,:] = p_a_int_diff
        params_fitting.update({iteration:that_iteration})
        
    # get the rates accounting for differences between best and not best models
    step_best = {}
    models = params_fitting
    best3 = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],1),3])
    not_best3 = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],1),parameter_spread-3])
    # preallocate new_rates
    new_rates = np.zeros([np.size(Q['rates'],0),np.size(Q['rates'],1),parameter_spread,np.size(differences,1)])
    print("Optimising {} models over {} steps".format(parameter_spread,steps))
    step_diffs = []
    for step in tqdm(np.arange(steps)):
        #step_diffs.append(np.abs(differences).mean().mean())
        step_diffs.append(np.abs(differences).mean(axis=1).min()) # tracking the time-avg fit of best model at each gen
        if step_diffs[-1] == np.min(step_diffs):
            best_model_num = np.where(np.abs(differences).mean(axis=1) == np.abs(differences).mean(axis=1).min())[0][0]
            step_best.update({step:models[best_model_num]})
        for timestep in np.arange(np.size(differences,1)):
            if not np.any(differences[:,timestep] == 0):
                best_3 = np.argsort(np.abs(differences[:,timestep]))[:3]
                # indices used for differences
                for modelnum, model in enumerate(best_3):
                    best3[:,:,modelnum] = models[model]['Q'][:,:]
                best3_avg = np.mean(best3,axis=2) # take mean of three best models ------- could be changed for sensitivity
                remaining_models = models.keys() - set(best_3)
                for item, value in enumerate(remaining_models):
                    not_best3[:,:,item] = models[value]['Q'][:,:]
                not_best3_avg = np.mean(not_best3,axis=2) #### take mean of worst --------- might be made mroe sensitive by taking next 3 or etc. Beacuse of uniform sampling, avg should be approx 1/2 parameter search space I guess
                #rate_diffs = np.subtract(best3_avg,not_best3_avg)
                # get whichever closest to the best 3 models: min (=0), max (user set), or mean (worst models)
                    # has to be done for each rate/group of rates individually - for multiplicative rates, shouldn't matter
                    # might make sense to generate indexers of unique rates to vary earlier
                    
                # establish whether reamining models, 0, or max rate is closer for each rate/group of rates using rate_indexers (see earlier)
                # multiple of relationship expresses as 4th entry
                for rate_indexer in rate_indexers:
                    rate_guess = best3_avg[rate_indexer[1],rate_indexer[2]]/rate_indexer[3] # get the rate/ rate multiple
                    remaining_guess = not_best3_avg[rate_indexer[1],rate_indexer[2]]/rate_indexer[3]
                    # 0 is min, so rate_guess - 0 = rate guess, and then to find the closest to best:
                    # if 0, then 0 is closest to best rate, 1 = remaining is closest,2= max closest
                    closest = np.array([0,remaining_guess,max_rate][np.argmin(np.array([rate_guess,rate_guess-remaining_guess,max_rate]))])
                    # draw the new random rate from gaussian with mean = mean(rate_guess,closest)
                    # and sd = np.abs(rate_guess-closest) # this could be tuned, but should be adequate... though 68% of guesses will be in this interval
                    new_rates[rate_indexer[1],rate_indexer[2],:,timestep] = np.random.normal(np.mean(np.array([rate_guess,closest])),np.abs(rate_guess-closest),parameter_spread)
                    # and populate the matrix as appropriate for multiples
                    if rate_indexer[0] !=0: # i.e. is a relationship encoding
                        relationships = np.stack((np.where(kk[:,:,0] ==rate_indexer[0]))) # zipping where each col gives indexer
                        for relation in np.arange(np.shape(relationships)[1]):
                            if target_rates[relationships[0,relation],relationships[1,relation]]==0.5: # if rate not set by user
                                new_rates[relationships[0,relation],relationships[1,relation],:,timestep] =  new_rates[rate_indexer[1],rate_indexer[2],:,timestep] * kk[relationships[0,relation],relationships[1,relation],1]
                            else:
                                new_rates[relationships[0,relation],relationships[1,relation],:,timestep] = Q['rates'][relationships[0,relation],relationships[1,relation]]

            else: # if the difference is 0 at that timestep, make that timepoint the previous model (to avoid spurious evolution)
                for model in np.arange(parameter_spread):
                    new_rates[:,:,model,timestep] = models[model]['Q'][:,:]
        # get median rates from all timepoints for each model (using mean could bias towards timepoints where the chain is stationary)
        median_new_rates = np.median(new_rates,axis=3)
        # repeat the simulations with each new model, after constraining
        new_params_fitting = {}
        new_differences = np.zeros([parameter_spread,np.size(p_a_norm_current,0)])

        for iteration in np.arange(parameter_spread):
            model_iteration = model_dict.copy()
            model_iteration.update({'Q':median_new_rates[:,:,iteration]})
            # enforces MR in Q and updates the rates to match Q, enforcing necessary conventions
            model_constraints(model_iteration, constraint_matrix)
            # simulate using that model for first 10th of time with matching noise_amplitude layered over
            # this should give some more power to rise times of the current, which tend to be faster, but also aids in efficiency
            kwargs['noise_sd'] = noise_amp
            kwargs['t_final'] = real_t_final
            kwargs['n_sweeps'] = 10
            kwargs['interval'] = 5e-05
            kwargs['N'] = 1000
            kwargs['Q'] = model_iteration
            iteration_output = simulate(show_progress=False,graph=False,**kwargs) # calling main simulate function
            #plt.close('all')
            iteration_current = current_to_DataFrame(iteration_output)
            # normalise the current to its maximum (i.e. of all sweeps)
            norm_iteration_current = (iteration_current/(iteration_current.abs().max().max()))
            # resample it to match that of the truncated data
            # changed to a cut operation
            #norm_iteration_current = resample_current(norm_iteration_current,1/1e-04)
            bins_curr = pd.cut(norm_iteration_current.index,np.arange(0,real_t_final,5e-05))
            norm_iteration_current = norm_iteration_current.groupby(bins_curr).mean()
            norm_iteration_current.index = np.arange(0,real_t_final,5e-05)[:-1]
            # for rmse and L2 calculation, perform on the current AFTER baseline
                # so that are more sensitive
            p_a_iteration_current = norm_iteration_current[norm_iteration_current.index>=ag_time]
            #rmse = mean_squared_error(p_a_norm_current, p_a_iteration_current,squared=True)
            #euc_distance = np.linalg.norm(p_a_norm_current.mean()-p_a_iteration_current.mean())
            p_a_int_norm_model = trapesium_t(p_a_iteration_current)
            p_a_int_diff = data_int_norm - p_a_int_norm_model
            #if euc_distance < 0.1:
                # a plausible current at this point, model should be less than 10% different in mean to the mean of the current during n
            that_iteration = model_iteration.copy()
            that_iteration.update({'int_diff':p_a_int_diff})
            #that_iteration.update({'distance':euc_distance})
            #that_iteration.update({'rmse':rmse})
            that_iteration.update({'model':iteration_output})
            new_params_fitting.update({iteration:that_iteration})
            #distances.append(euc_distance)
            new_differences[iteration,:] = p_a_int_diff
            new_params_fitting.update({iteration:that_iteration})
        #completing the loop
        params_fitting = new_params_fitting.copy()
        differences = new_differences
        models=params_fitting.copy()
    plt.plot(np.arange(steps),step_diffs)
    return(step_best)


def trapesium_t(current_dataframe):
    """
    Used by 'fittigration' method
    """
    x = current_dataframe.mean(axis=1).to_numpy()
    y = current_dataframe.mean(axis=1).index
    
    s = np.zeros_like(y)
    for timestep, time in enumerate(y):
        #s[timestep] = np.sum((x[:timestep] - x[0]) * (y[:timestep] + y[0]) / 2)
        s[timestep] = np.trapz(x[:timestep],y[:timestep]) ### faster 
            
    return(s)

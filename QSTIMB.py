#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benjamintagg

Q-matrix and Stochastic simulation based Ion Channel Model Builder

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
            
    [NB] Relaxations (i.e. the non-agonist application procedures)are much faster
    and should be used in first instance.
    
    [3]---- CME Simulation Methods ----
    ================================
    - Q matrix method for relaxations
    - Q matric method for agonist applications
    
    [4]------ Built-in functions -----
    ================================
    - For graphing transition or Q matrix
    - For adding gaussian noise
    - Methods for enforcing microscopic reversibility onto Q matrix (CME)
    - Realistic concentration jumps (stochastic and CME) (Credit AP Plested) 
    - Functions associated with Q matrix method calculations (CME)

"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import interp1d
import copy
import networkx as nx

# config
import matplotlib.colors as mcolors
from cycler import cycler
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colorlist = ['black','dimgrey','teal','darkturquoise', 'midnightblue','lightskyblue','steelblue','royalblue','lightsteelblue','darkorange', 'orange','darkgoldenrod','goldenrod','gold','khaki','yellow']
mycycle = cycler(color=[colors[item] for item in colorlist])
# =============================================================================
# In-built Models for testing: Q (transition matrix) construction and modification
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

def threesQ(agonist_conc =5*(10**-3)):
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
def GlyAGQ(gly_conc=5*(10**-3)):
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

def GlyLeg98Q(gly_conc = 5*10**-3):
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
    return(Q)

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
# Poisson-based Tau leaping
# =============================================================================
def Tau_leap_Gillespie(N,Q,t_final,interval = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True):
    """
    
    Classical method for Gillespie Tau (interval) leaping - please read all notes
    
    Tracks trajectories for N receptors during relaxations using a single Q matrix
    that holds transitions rates (s^-1) as i,j for transition from i to j.
    
    The relaxation is repeated for the number of times in iterations. Occupancies are
    averaged, and the current is plotted and returned if a conducitng state is specified in the model
    using the average occupancies. Thus, for a single trajectory, set iterations = 1.
    
    If conductance is not specified, mean occupacnies are returned.
    
    The size of tau (the interval) is fixed (here to 5e-05 = 1/20000 = 20kHz), and may require some trial-and-error.
    An alternative would be to perform adaptive Tau leaping (Cao,2006), but that is designed for
    reactions with rate laws, rather than transition rates.
    A value for a fixed interval can crudely be obtained by 1/np.nanmax(Q['rates']) -
    i.e.
    a timestep selected is such that only one of even the fastest reaction could occur per timestep.
    
    However, this can lead to 'ringing' of open states in relaxations.
    
    Performance is related to the number of iterations, the size of the interval (i.e. number of steps).
    Models with large numbers of states will run slower, but unlike adaptive Tau,
    having many concentration-dependent states does not majorly alter runtime.
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
                    compar_accumulator[stateto] = np.random.poisson(((Rates[statefrom,stateto])*N*interval)) # lambda = R*t
                    
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
        p_t[np.isnan(p_t)]=0 # as above
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
        return(p_t,occupancy,currents*10**12)

    else:
        return(p_t,occupancy)

def agonist_application_tau_leap_Gillespie(N,Q,t_final,agonist_time,agonist_duration,first_conc,second_conc,interval = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True,rise_time = 250*10**-6,decay_time = 300*10**-6):
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
    concentrations,t = concentration_as_steps(first_conc=first_conc, second_conc=second_conc, dt=interval, start_time=agonist_time, duration=agonist_duration,rise_time = rise_time, decay_time = decay_time)
    
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
                    compar_accumulator[stateto] = np.random.poisson((Qs[statefrom,stateto,intervalcount])*N*interval) # lambda = R*t*N

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
            _,remaining_occs,_ = Tau_leap_Gillespie(N = n, Q = S,t_final = t_final-t[-1]-interval,interval = interval,voltage=voltage,Vrev = Vrev,iterations = 1,plot=False)
            iteration_occupancy[:,:,iteration] = np.hstack((occupancy,remaining_occs))
        else:
            iteration_occupancy[:,:,iteration] = occupancy[:,:np.size(iteration_occupancy,1)] # catch for single sample overspill
        
    # get mean occupancy for iterations
    mean_occ = np.mean(iteration_occupancy,2)
    
    # adjust Tnew so that pulse is translated to correct loc
    # add in time remaining from relaxation
    Tnew = np.arange(0,t_final,interval)
    Tnew = np.concatenate((np.array([0]),Tnew[:-1]+agonist_time-(agonist_time-rise_time-decay_time-(0.5*agonist_duration))))
    occupancy = mean_occ # lazy catch for below
    
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
        p_t[np.isnan(p_t)]=0 # as above
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
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format((1/interval)/1000))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
        return(p_t,occupancy,currents*10**12)

    else:
        return(p_t,occupancy)
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
    Models with large number of states will run slower, and each additional concentration-dependent transition increases the number of operations
    so for models with large numbers of concentration-dependent rates, using fixed Tau leaping
    is recommended. Similarly, the length of simulation, t_final, will also impact performance for
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
                        compar_accumulator[stateto] = np.random.poisson(((Rates[statefrom,stateto])*N*dt)) # lambda = R*t
                        
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
        p_t[np.isnan(p_t)]=0 # as above
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
        return(p_t,occupancy,currents*10**12)

    else:
        return(p_t,occupancy)
    
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
        is recommended
    
    For efficiency, the pulse is calculated as a centered error function (that is later translated to the
    correct time). If UnboundLocalError is returned, centering the function has failed due to inadequate time
    before the maximum agonist application. Delay the agonist_time, such that
    agonist_time > 0.5* agonist_duration, and t_final, and the error should not return.
    
    t_final should be 2x duration
    """
    # get concentrations
    concentrations,t_concs = concentration_as_steps(first_conc=first_conc, second_conc=second_conc, dt=sampling, start_time=agonist_time, duration=agonist_duration,rise_time = rise_time, decay_time = decay_time)
    # get function for interpolating concs
    interpolated_conc_f = interp1d(t_concs,concentrations,fill_value='extrapolate') 
    # to be used at each t for getting concs
    
    # get conc-dep properties amd max possible rates
    conc_fraction = Q['conc']
    conc_rates = [item for item in Q['conc-dep'].items()]
    # max_rates = np.copy(Q['rates'])      DEPRECATED
    # for item in conc_rates:
    #     max_rates[item[0],item[1]] = np.max(np.array([first_conc,second_conc]))*(max_rates[item[0],item[1]])
        
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
            while t<agonist_time-rise_time-decay_time-(0.5*agonist_duration): # while chain itself is stationary
                pre_pulse_concs = interpolated_conc_f(np.arange(0,agonist_time-rise_time-decay_time-(0.5*agonist_duration)+sampling,sampling))
                if not np.any(pre_pulse_concs): # if conc is zero before pulse
                    #advance to first non-zero concentration time
                    t = agonist_time-rise_time-decay_time-(0.5*agonist_duration)+sampling
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
                    # then do the relaxation
                    t_inter = np.min(np.where(np.arange(0,agonist_time,sampling)>agonist_time-rise_time-decay_time-(0.5*agonist_duration)))/(1/sampling) + sampling
                    _,occs,_  = Weighted_adaptive_Tau_leap(N = N,Q=newQ,t_final=t_inter,sampling=sampling,voltage=voltage,Vrev=Vrev,iterations=1,plot=False)
                    times = list(np.arange(0,t_inter,sampling))
                    t = t_inter
                occupancy = occs
                
            while (t>=agonist_time-rise_time-decay_time-(0.5*agonist_duration)) & (t<t_final): # when chain non-stationary
            
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
                dt = Taus*frcprev
                if np.any(np.isfinite(dt) & (dt>0)): # catch - shouldn't be triggered
                    dt = np.nanmin(dt[dt>0])
                else:
                    dt = sampling
    
                # but if conc very low (at beginning of pulse), timestep very long
                    # unles impose some condition on non-zero concs so that dt does not step over pulse
                #for duration of the pulse, because the chain changes rapidly, use dt = sampling if dt longer
                if t<= agonist_time-rise_time-decay_time+(0.5*agonist_duration):
                    if dt > sampling: #
                        dt = sampling
            
                # determine which transitions occur in next interval
                for statefrom, N in enumerate(prevstates):
                    for stateto in np.arange(np.size(Q['rates'],1)):
                        compar_accumulator[stateto] = np.random.poisson(((Rates[statefrom,stateto])*N*dt)) # lambda = R*t
                        
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
        #times = np.concatenate((times[:2],times[2]))
        for item in np.arange(np.size(occupancy,0)):
            interpolated_f = interp1d(times,occupancy[item,:],fill_value='extrapolate') 
            Tnew = np.arange(0, t_final, sampling)
            upsampled_occupancy = interpolated_f(Tnew)
            sampled_occupancy[item,:,iteration] = upsampled_occupancy
    
    # adjust Tnew so that pulse is translated to correct loc
    Tnew = np.concatenate((np.array([0]),Tnew[:-1]+agonist_time-(agonist_time-rise_time-decay_time-(0.5*agonist_duration))))
    
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
        p_t[np.isnan(p_t)]=0 # as above
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
        return(p_t,occupancy,currents*10**12)

    else:
        return(p_t,occupancy)
# =============================================================================
#  Q-matrix Methods and CME simulation
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
    initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
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
            return(p_t,occupancy,currents*10**12)
        else:
            return(p_t,occupancy)

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

    """
    # get t by interval and discrete concentration by timestep
    t = np.arange(0,t_final,interval)
    jump_concs,jump_times = concentration_as_steps(first_conc=first_conc,second_conc=second_conc,dt=interval,start_time=agonist_time,duration = agonist_duration,rise_time=rise_time,decay_time=decay_time)
    
    #take initial states for N with probability given in Q['intial states']
    initial_states = [int(i) for i in Q['initial states'].keys()]
    initial_probabilities = [int(i)*N for i in Q['initial states'].values()]
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
    t_at_jump = jump_times[np.min(np.where(jump_times>agonist_time-rise_time-decay_time-(0.5*agonist_duration)))] + interval
    jump_indexer = np.min(np.where(jump_times>agonist_time-rise_time-decay_time-(0.5*agonist_duration))) + 1
    # relax in the pre-jump_constant_conc
    R = copy.deepcopy(Q)
    R.update({'Q':Qs[:,:,0]}) # update with the Q for pre-agonist constant conc
    R.update({'conc':first_conc}) # and the conc
    pre_jump_pt,occs,_ = Q_relax(Q=R,N = N,Vrev=Vrev,interval=interval,t_final=t_at_jump,voltage=voltage,plot=False,just_pt=False)
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
    S = copy.deepcopy(Q)
    S.update({'Q':Qs[:,:,0]})
    S.update({'conc':first_conc}) # and update the conc to first_conc
    S.update({'initial states':{}})
    for key, value in enumerate(jump_pt[:,-1]):
        if value >0:
            S['initial states'].update({key:value})
    post_jump_pt,post_jump_occs,_ = Q_relax(Q=S,N = N,Vrev=Vrev,interval=interval,t_final=t_final-jump_times[-1],voltage=voltage,plot=False,just_pt=False)
    # concatenate all pt & occs
   # p_t =np.hstack((pre_jump_pt,jump_pt,post_jump_pt)) # deprecated
    occupancy =np.hstack((occupancy_in_jump,post_jump_occs))
    
    # translate t so that jump starts at agonist time 
    t = np.arange(0,t_final,interval)
    t = np.concatenate((np.array([0]),t[:-1]+agonist_time-(agonist_time-rise_time-decay_time-(0.5*agonist_duration))))
    Tnew = t
    
    # clip Tnew, occupancy to t_final & rederive p_t from occs:
    if Tnew[-1]>t_final:
        occupancy = occupancy[:,:np.min(np.where(Tnew>t_final))]
        Tnew = Tnew[:np.min(np.where(Tnew>t_final))]
        p_t = np.divide(occupancy,np.max(np.nansum(occupancy,0)),where=occupancy>0)
    
    # tidying up
    if 'conducting states' in Q.keys():
        conducting_occupancies = np.zeros([len(Q['conducting states'].keys()),np.size(occupancy,1)])
        for item, value in enumerate(Q['conducting states'].keys()): # multiply conductance of each state by occupancy and drivign force
            conducting_occupancies[item,:] = occupancy[value,:]*(((voltage-Vrev) *10**-3)*Q['conducting states'][value])
        currents = np.nansum(conducting_occupancies,0)
        currents[np.isnan(currents)] = 0 # fix for not plotting remainder whne == np.nan
        post_jump_pt[np.isnan(post_jump_pt)]=0 # as above

        if plot:   
                plt.style.use('ggplot')
                figure,axes = plt.subplots(2,1)
                axes[0].plot(Tnew,currents*10**12,color='black') # plot current
                axes[0].set_title("Simulated Current, N = {}, agonist pulse conc = {} M".format(N,second_conc))
                axes[0].set_xlabel("t (s)")
                axes[0].set_ylabel("pA")
                axes[1].set_prop_cycle(mycycle)
                # plotting occupancy probabilities over time
                for state in np.arange(np.size(p_t[:,0],axis=0)):
                    axes[1].plot(Tnew,p_t[state,:],label="{}".format(state))
                axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
                axes[1].set_title("P(State Occupancy at t), {}kHZ".format((1/interval)/1000))
                axes[1].set_xlabel("t (s)")
                axes[1].set_ylabel("Probability")
                plt.tight_layout()
        return(p_t,occupancy,currents[:-1]*10**12)

    else:
        return(p_t,occupancy)
# =============================================================================
# Built-In functions
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
    noise = np.random.normal(0,noise_sd,samples)
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

def make_Q_reversible(Q,show=False,protected_rates = False):
    """Taking a Q matrix and applying microscopic reversibility to it
    using minimum spanning tree principle from Colquhoun et al., 2004
    
    A graph is constructed, identifying the rates to constrain, and the
    remainder are set by microscopic reversibility. For rates to be set
    by MR, a minimum spanning tree is identified. By the principle that these
    form independent cycles, the MR rates can then be set in any order.
    
    Args:
    Q: a Q matrix containing rates. It is easiest if this matrix contains
        rates that one wants to constrain the model by. Any rates that exist
        but should be set by MR should be any value >0. The option will be
        given to select which rates should be constrained (see protected_rates too).
        
    show (Default=False): 
        When = True, a graph is produced to show the model, with each node as a state
    
    protected_rates (Default = False). 
            When False, the user will be asked to
            select rates to constrain (i.e those not to be set by MR). These
            will then form an additional output for future usage. Otherwise,
            protected_rates should be entered in the format as the type produced
            from this function, and they will not be output.
            
            E.g. In first instance, use: Q,protected_rates = make_Q_reversible(Q,protected_rates=False)
            Then using the output from above, use: Q = make_Q_reversible(Q,protected_rates=protected rates)
    
    """
    if not np.any(protected_rates):
        Q_logical = logicalQ_constructor(Q) #calling embedded to create a logical matrix
    else:
        Q_logical = protected_rates
    logic_graph = nx.MultiDiGraph(Q_logical)
    for edge in logic_graph.edges():
        # rates to be set by microscopic reversibility have weight 1, set have weight 0.5
        if logic_graph.get_edge_data(edge[0],edge[1])[0]['weight'] ==1: # rate ij = rateji*(product rates of forward route/product rates of reverse route)
            # where item[0] is source and itme[1] is target, getting shortest forward and reverse paths
            forward_path = nx.shortest_path(logic_graph,edge[0],edge[1])
            reverse_path = nx.shortest_path(logic_graph,edge[1],edge[0])
            # getting product of rates along this path using embedded function
            forward_rates_prod = path_rates(forward_path,Q)
            reverse_rates_prod = path_rates(reverse_path,Q)
            Q[edge[0],edge[1]] = Q[edge[1],edge[0]] * (forward_rates_prod/reverse_rates_prod)
    if show:
        Qgraph = nx.MultiDiGraph(Q)
        nx.draw(Qgraph,pos = nx.spring_layout(Qgraph),with_labels=True)
    if not np.any(protected_rates):
        return(Q,Q_logical)
    else:
        return(Q)

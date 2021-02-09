#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:10:38 2020
@author: benjamintagg
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
-----------------------------------------------------------------
-------------------------------REQUIREMENTS----------------------
-----------------------------------------------------------------
For installation, see:
Numpy package for scientific computing: https://numpy.org/
Matplotlib 2D plotting library: https://matplotlib.org/index.html
Scipy: fitting error function: https://www.scipy.org/
Pandas:
NetworkX for analysis of networks: https://networkx.github.io/documentation/stable/install.html
Sympy for solving systems of equations: https://www.sympy.org/en/index.html

-----------------------------------------------------------------
-------------------------------Credit----------------------------
-----------------------------------------------------------------
The initial model for A1 +y-2 is taken from Coombs et al.(2017): doi: 10.1016/j.celrep.2017.07.014. 

The function concentration_as_steps is adapted from pulse_erf of SCALCS/c_jumps: https://github.com/DCPROGS/SCALCS/blob/master/scalcs/cjumps.py
Under the terms of that license:
    "You may modify your copy or copies of the Program or any portion of it,
    thus forming a work based on the Program, and copy and distribute such modifications
    or work under the terms of Section 1 above, provided that you also meet
    all of these conditions:
a) You must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.; b) You must cause any work that you distribute or publish, that in whole or in part contains or is derived from the Program or any part thereof, to be licensed as a whole at no charge to all third parties under the terms of this License. c) If the modified program normally reads commands interactively when run, you must cause it, when started running for such interactive use in the most ordinary way, to print or display an announcement including an appropriate copyright notice and a notice that there is no warranty (or else, saying that you provide a warranty) and that users may redistribute the program under these conditions, and telling the user how to view a copy of this License.  (Exception: if the Program itself is interactive but does not normally print such an announcement, your work based on the Program is not required to print an announcement.)


Thus approach is inspired by Sachs (1999): https://doi.org/10.1016/S0006-3495(99)76923-7,
first realised by Andrew Plested (https://www.leibniz-fmp.de/mnb/software), who
used an error function to approximate the concentration profile.


-----------------------------------------------------------------
-------------------------------DESCRIPTION-----------------------
-----------------------------------------------------------------
This script contains functions for a time-homogenous Markov chain model of AMPA
receptor kinetics. It is based in that originally published Robert and Howe (2003),
that was subsequently updated in in Coombs et al.(2017), Coombs et al.(2019).
As well as a simulation by distribution (Colquhoun and Hawkes, Q matrix cookbook),
a stochastic simulation is implemented in discrete time, but the trajectory of
each receptor is determined probabilistically over a random walks; 
rather than the distribution.

The stochastic simulation is enacted with the simple rule that the next transition
that occurs is random. The probabilities on which it is enacted defy the Markovian assumption
but it nevertheless demonstrates that bursting behaviour can arise from enacting randomness
Thus underlying the idea that stochastic behaviour at the level of transitions relates single
channel behaviour to the more probabilistic channel currents.
A better rule is enacted in V3.

As such, the model should be able to make suitable experimental predictions,
when sampling rate of the experimental apparatus is considered; that concern
identity of a receptor - e.g. open probability.
The number of missed events should also be reduced by considering trajectory,
rather than distirbution.
"""
######Configuration######
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from math import exp
from scipy import special # does not import with scipy by default
import networkx as nx
#########################
# Coombs (2017) scheme
# Assumes that a homogeneous population of AMPARs has 1:1 TARP stochiometry (probably hippocampal, but unlikely cerebellar (Soto,unpublished)),
# subunits that behave independently and stochastically; with equal, additive
# contribution to unitary current. As such, 
# R1-4: apo state: liganded, ungated or deactivated receptor subunit)
# O1-4: Ion channel open
# D1-4: Ion channel desensitised
# Desensitised conducting state (Coombs, 2019) only appear to exist for
# GluA2(R) homomers and has thus been excluded.

def AMPAR_generator(glu_conc = 10,k_on = False,k_off = False, alpha = False, beta = False,gamma = False, delta = False,gamma_0 = False, delta_0 = False,delta_3 = False,k_off_2 = False, k_off_3 = False,gamma_2=False,delta_2=False,delta_1 = False,real_transition_only = True):
    """Creates generator matrix (transition matrix), Q, for time-homogeneous CTMC
    AMPAR_Q takes rates (s) as inputs to return the generator array.
    If no input is provided, the default values from Coombs et al. (2017) are
    used, using the same notation. Concentration in mM.
    Diagnonal entries follow convention such that sum Q[i,:] = 0 is satisfied.
    NB: Q is constructed as a square matrix by inserting non-existant states [0,15,16].
    HOWEVER, these are removed when real_transition_only = True (Default).
    
    NB that direct transitions between open states are not considered
    
    """
    # Nb, units below have been converted to mS, such that one unit of time = 1mS
    # but are presented in format (S/1000) to maintain consistency with format in Coombs(2017,2019)
    if glu_conc !=0:
        glu_conc = glu_conc*(10**-3) # converting concentration to molar format for k_on rate calc.
    # since concentration response approx linear between EC_{min} and EC_{max}, can scale rate k_on according to conc.
    # rates converted to ms

    if not k_on:
        k_on = (1.3*(10)**7)/1000 # M^-1 s^-1 per mole per millisecond - nb next line.
    k_on = k_on*glu_conc # scaling rate constant for binding according to molar conc glutamate
    # Dimensionless constants:
    #if not orate: depricated
       # orate = k_on #s^-1
    #if not crate:
        #crate = 150000/1000 #s^-1
    if not k_off:
        k_off = (3000/1000) # s^-1
    if not alpha:
        alpha = (1000/1000) # s^-1
    if not beta:
        beta = (6000/1000) # s^-1
    if not gamma:
        gamma = (16/1000) # s^-1
    if not gamma_0:
        gamma_0 = (4.4/1000) # s^-1
    if not delta_0:
        delta_0 = (0.48/1000) # s^-1
    if not delta_1:
        delta_1 = (1200/1000)
    if not delta_2:
        delta_2 = (1300/1000)
    if not delta_3:
        delta_3 = (250/1000)# s^-1
    if not k_off_2:
        k_off_2 = (63/1000)# s^-1
    if not k_off_3:
        k_off_3 = (630/1000)# s^-1
    if not gamma_2:
        gamma_2 = (3900/1000)# s^-1
    # formal invocation of generator matrix Q as N(S) X N(S) matrix, where S = state, and Q[Si,Sj] = rate(Sij). NB, in python, Q[sij] = Q[Si-1 j -1]
    Q = np.zeros([20,20]) # for 17 states, creating 20 X 20 matrix, since most most transitions do not occur (e.g. O1->D4)
    # Rates(q) indexed in Q. NB. 0 values placed to account for transitions into and from non-existent O0
    
    # indexing rates between open transitions (O1-4) in forward and reverse directions (DEPRICATED)
    #Q[1,2],Q[2,3],Q[3,4],Q[2,1],Q[3,2],Q[4,3] = (3*orate),(2*orate),orate, crate,(2*crate),(3*crate) 
    # indexing rates for gating: occupied (R0-4) to open (O1-4) transitions, in forward and reverse directions
    Q[6,1],Q[7,2],Q[8,3],Q[9,4],Q[1,6],Q[2,7],Q[3,8],Q[4,9] = beta,(2*beta),(2*beta),(4*beta),alpha,alpha,alpha,alpha
    # indexing rates for ligand binding and unbinding (R0-R4)
    Q[5,6],Q[6,7],Q[7,8],Q[8,9],Q[6,5],Q[7,6],Q[8,7],Q[9,8] = (4*k_on),(3*k_on),(2*k_on),k_on,k_off,(2*k_off),(3*k_off),(4*k_off)
    # indexing rates for transition to and from desensitised states (D1-4)
    Q[10,5],Q[11,6],Q[12,7],Q[13,8],Q[14,9],Q[5,10],Q[6,11],Q[7,12],Q[8,13],Q[9,14] = gamma_0,gamma,gamma,gamma,gamma, (4*delta_0),delta_3,(2*delta_1),(3*delta_1),(4*delta_1)
    # indexing rates for transitions between desensitised states
    Q[10,11],Q[11,12],Q[12,13],Q[13,14],Q[11,10],Q[12,11],Q[13,12],Q[14,13] = (3*k_on),(3*k_on),(2*k_on),k_on,k_off_2,k_off_3,(2*k_off),(3*k_off)
    # indexing rates for transitions to deep desensitised states
    Q[12,17],Q[13,18],Q[14,19],Q[17,12],Q[18,13],Q[19,14] = delta_2,(2*delta_2),(3*delta_2),gamma_2,gamma_2,gamma_2
    # indexing rates for transitions between deep desensitised states
    Q[17,18],Q[18,19],Q[18,17],Q[19,18] = (2*k_on),k_on,k_off,(2*k_off)
    # by convention Q[ii] = - Sum_(jES, j!=i) *  (q(ij). i.e. Q[i,i] = - sum(Q[i,1:20])
    if real_transition_only:
        Q = np.delete(Q,[0,15,16],0)
        Q = np.delete(Q,[0,15,16],1)
    for row in range(0,np.size(Q,axis=1)):
        Q[row,row] = - np.sum(Q[row]) # replacing qii. Nb, that since Q[i,0:14] is empty, Q[0,0] = 0 is still satisfied
    return Q

def Q_constructor(N_states,give_key=False,give_rate_dict = False,show=False):
    """A general Q matrix constructor that creates a Q matrix of size N_states
    x N_states allowing rate constant mapping, and assignment of their values.
    
    Args:
        
    N_states: number of states in the model
    
    give_key (Default = False). When False, constructs a key of the Q matrix:
    
            First, a Q matrix key is constructed by entering variable names:
                Rates for a group of transitions, such as those between resting states 
                may share a rate constant, in which case this is specified here.
                    - If they share a rate constant, this should be specified
                        E.g. for transitions ij and ik sharing constant gamma, where ij is
                        twice as fast as ik, when asked to enter a variable name, for ij set
                        2*gamma; and for ik, set 1*gamma [NB, no spaces]
                    - If no rate exists for this transition, set 0
                    - If a rate exists, but you want to set it by MR, set MR. (see below*)
                    
            The key will then be returned, and can be input as give_key = key
            for a subsequent instance should the user only wish to modify the
            rate constants. The key list of rate constants will also be returned
            
           ! An example of the format required can be seen by entering variablename = A1_y2_key()
    
    give_rate_dict(Default = False): User can provide a dictionary of rate constants
                            and their value. This is useful if only one value needs to be changed.
                            If entered, the user will be asked to modify a particular constant
                            
    show (Defeault=False) When set True, an image of the model is shown as a biograph object
    
    * when a rate is set as MR, microscopic reversibility will be used ot calculate that rate.
    """
    # warning
    print('For models with large numbers of states, you may wish to manually assemble a rate dictionary')
    # constructing key if not provided
    if not np.any(give_key):
        Q_key = np.zeros([N_states,N_states])
        Q_key = Q_key.astype(np.chararray)
        print("For help with variable names, enter help(Q_constructor) in the terminal")
        rate_constant_list = []
        for state in np.arange(np.size(Q_key,0)):
            for transition in np.arange(np.size(Q_key,0)):
                key = input("Enter multiple * rate_constant for transition from {} to {} >".format(state,transition))
                Q_key[state,transition] = key    
                if "*" in key:
                    constant = key.split("*")[1]
                    if constant not in rate_constant_list: # if that rate constant not already in list, add it
                        rate_constant_list.append(constant)
                        
    # If key constructed, get rate list
    else:
        Q_key = give_key
        rate_constant_list = []
        for state in np.arange(np.size(Q_key,0)):
            for transition in np.arange(np.size(Q_key,0)):
                key = Q_key[state,transition]
                if "*" in key:
                    constant = key.split("*")[1]
                    if constant not in rate_constant_list: # if that rate constant not already in list, add it
                        rate_constant_list.append(constant)
                       
    # If rate constant dictionary not provided, create one with user-specified values
    if not give_rate_dict: # empty dicts return False
        rate_constants = {}
        print("If a rate constant is concentration-dependent, account for this in the value entered")
        for item in rate_constant_list:
            rate_constants[item] = float(input("Enter a value for rate constant {} >".format(item)))
            
    # if a rate constant dict provided, choose a value to adjust
    else: 
        rate_constants = give_rate_dict
        adjust = input("Enter rate_constant_name,new value >").split(",")
        which_constant = "{}".format(adjust[0])
        rate_constants[which_constant] = float(adjust[1])
        
    # Using key to construct Q matrix
    Q = np.zeros([np.size(Q_key,0),np.size(Q_key,0)])
    for state in np.arange(np.size(Q_key,0)):
        for transition in np.arange(np.size(Q_key,1)):
            if "*" in Q_key[state,transition]:
                multiple,constant = Q_key[state,transition].split("*")
                multiple = float(multiple)
                Q[state,transition] = multiple*(rate_constants[constant])
            elif "0" in Q_key[state,transition]:
                Q[state,transition] = 0
            elif "MR" in Q_key[state,transition]:
                Q[state,transition] = np.nan # for now, set nan
                
    for row in range(0,np.size(Q,axis=1)):
        Q[row,row] = - np.nansum(Q[row]) # replacing qii. Nb, that since Q[i,0:14] is empty, Q[0,0] = 0 is still satisfied
        
    # setting microscopically reversible rates
    protected_rates = np.zeros([np.size(Q,axis=0),np.size(Q,axis=1)])
    protected_rates = np.where(np.isnan(Q)==False,0.5,Q) # for constrained rates, set edge length  = 0.5
    protected_rates = np.where(np.isnan(protected_rates),1,protected_rates) # for rates to set by MR, = 1
    if show:
        Q = make_Q_reversible(Q,show=True,protected_rates = protected_rates) # call embedded function, with or without graphs
    else:
        Q = make_Q_reversible(Q,protected_rates = protected_rates)
    print("Q matrix has been constructed consisting of rates set by constraint and by microscopic reversibility")
    if not np.any(give_key):
        return(Q,Q_key,rate_constants)
    else:
        return(Q,rate_constants)
######Here#####
# may want to catch instances of entering mathematical operations as rate constant values - e.g. powers, division, multiplication
# after initial condition of give_key, nothing else is running
# think because i forgot to do 1*
# need agonist conc option on some rates.
# would also need to factor in that number of open states may change in general implementation
# but if open states assigned, then conductances easily entered by other functions.

def probabilistic_walk(N,t_final,voltage= -60,agonist_conc= 10,sampling= 20,Vrev= 0,Q=False,conductances=False,graphs =False, initial_states = False,just_pt=False):
    """Simulates the change in distribution for N receptors over t=0, t_final
    for a single iteration. E.g. During a relaxation to steady-state.
    Q, the transition matrix, is time-homogenous.
    Returns occupancy probability, or simulated currents (pA) and occupancies
    (see just_pt arg). For visualisation, See Graphs arg.
    Assumes constant concentration, such that p_inf constant.
    
    NB, since occupancy probability ={0,1}, occupancies !={R}. For non-constant
    concentration, call this function for each concentration step.
    
    optional arguments include:
    
    -t_final (ms)
    
    -agonist_conc (mM)
    
    - Q, a transition matrix. If False (default), the Q matrix for
    AMPAR as A1 +y-2 at 1:4 stoichiometry is obtained.
    
    
    - voltage (mV) specifies the voltage at which to calculate the currents. 
    Default = -60.
    
    - conductances (pS) allows specification of conductances for open states of Q
    as a vector.
    
    - sampling (kHz) specifies the time interval (default = 20). Ideally, this
    should be matched to experimental apparatus.
    
    - graphs returns the visualisation of the state distribution, and current
    when True (Default = False). When False, returns output in order:
    occupancy probability,occupancy, and currents
    
    - initial_states (Default = False) allows specification of a row vector,
    where each entry is the NUMBER of receptors in each initial state.
    Must have same size as the 0 axis of Q. When false, all are entered in
    resting state. Nb, for steady_state, call initial_states = p_inf(Q)
    
    - agonist_conc allows specification of agonist concentration. Default = 10(mM)
    
    - just_pt (Default=False) returns only the occupancy probabilties for each time
    when = True.
     
    - Vrev is the reversal potential of the channel in mV. Default -0.
    """
    time_intervals = np.arange(0,t_final,(1/(sampling))) # time units are in msec, sampling in kHz, so 1000(msec)/sampling*1000 = samples per sec = 1/sampling
    if not np.any(Q):
        Q = AMPAR_generator(glu_conc=agonist_conc)
    if not np.any(conductances): # but conductances not specified
        conductances = np.array([3.7,16.1,30.6,38.6]) # conductances for GluA1 + y-2 used as default (1:4 stoichiometry)
    if not np.any(initial_states):
        initial_states = np.zeros(np.size(Q,0))
        initial_states[4] = N
    p_t = np.zeros([np.size(Q,0),np.size(time_intervals,0)]) # preallocation to store probabilities (from t =0+t to t=t_final), not initial probs
    eigvals,eigvecs,spectrals = spectral_matrices(Q,ret_eig=True) # calling embedded function to compute sorted eigenvalues,eigenvectors, and spectral matrices.
    # each component ofspectrals, k, is stored in the 3rd axis (axis =2)
    # with each time interval, p_t changes, and lit changes
    pinf = p_inf(Q) # steady-state occupancies
    pzero = initial_states/np.nansum(initial_states) # occupancy probabilities for initialstates,if specified or not
    amplitude_coefficients = amplitude_coeff(spectrals, pzero) #calling embedded function to find the amplitude coefficients
    for intervalnum, intervaltime in enumerate(time_intervals):
        exp_component = np.exp(-eigvals*intervaltime) # calculating exponential terms at time
        p_jt = coefficients_to_components(exp_component, amplitude_coefficients) # calling embedded function to multiply out terms ( = probability of state j occupancy at t)
        p_t[:,intervalnum] = pinf + p_jt
    if just_pt:
        return(p_t)
    else:
        voltage = (voltage-Vrev) *10**-3 # Driving force in mV
        # N*g*(V-Vrev)*p_t
        gv = conductances * voltage
        occupancies = p_t*N
        open_states = np.copy(occupancies[:4,:])
        open_states[0,:] = open_states[0,:] * gv[0]
        open_states[1,:] = open_states[1,:] * gv[1]
        open_states[2,:] = open_states[2,:] * gv[2]
        open_states[3,:] = open_states[3,:] * gv[3]
        currents = np.sum(open_states,0)
        if not graphs:
            return(p_t,occupancies,currents)
        else:
            # if visualisation requested (graphs = True)
            plt.style.use('ggplot')
            figure,axes = plt.subplots(2,1)
            axes[0].plot(time_intervals,currents,color='black') # plot current
            axes[0].set_title("Simulated Current, N = {}, agonist conc = {} mM".format(N,agonist_conc))
            axes[0].set_xlabel("t (mS)")
            axes[0].set_ylabel("pA")
            # plotting occupancy probabilities over time
            for state in np.arange(np.size(p_t[:,0],axis=0)):
                axes[1].plot(time_intervals,p_t[state,:],label="{}".format(state))
            axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
            axes[1].set_title("P(State Occupancy at t), {}kHZ".format(sampling))
            axes[1].set_xlabel("t (mS)")
            axes[1].set_ylabel("Probability")
            plt.tight_layout()
            return(p_t,occupancies,currents)

def random_walk(N,t_final,iterations=1,voltage =-60, sampling = 20, agonist_conc = 10,Vrev = 0,Q=False,conductances=False,graphs = False, initial_states = False, simple =False):
    """Performs a random walk during relaxation to steady-state for N receptors
    over t = 0,t_final, for a specified number of iterations. Returns
    occupancy probability, state occupancies from the walk(s) and currents,
    for either a single, or the average of several random walks at intervals
    1/sampling. See Graphs arg.
    
    NB, that state occupancy ={R}, and thus occupancies are
    
    A number of additional arguments may be specified:
        
    - iterations specifies the number of random walks to perform (deafult 1).
    If iterations >1, the average for all runs is returned.
    NB that occupancy probability does not change, but occupancy and
    thus current will.
    
    - voltage in mV (default -60).
    
    -Vrev (Default =0) is the Reversal potential of the channel
    
    - sampling (kHz) specifies the time interval (default  = 20). Ideally, this
    should be matched to that of the experimental apparatus.
    
    -agonist_conc specifies the agonist concentration in mM (Default = 10)
    
    - Q gives the option to specify a Q matrix. If False, the Q matrix for
    AMPAR as A1 +y-2 at 1:4 stoichiometry is obtained.
    
    - Conductances(pS) gives the option to specify a vector of conductances.
    If False (default), conductances for A1 + y-2 (1: stoichiometry) are used
    
    - Graphs returns the visualisation of the state distribution, and current
    when True (Default = False). These will be averages when iterations >1
    
    - initial_states (Default = False) allows specification of a row vector,
    where each entry is the number of receptors in each initial state.
    
    -simple (Default = False) uses a simple rule to enact stochastic behaviour
    that is anti-Markovian, but can make a useful point about single receptors.
    For more guidance enter help(stochastic_advance). When simple = False,
    A Markovian stochastic rule is implemented that includes the probabilities
    if multiple transitions in a discrete time interval.See help(stochastic_transitions)
    """
    if not np.any(Q):
        Q = AMPAR_generator(agonist_conc)
    if not np.any(initial_states):
        initial_states = np.zeros(np.size(Q,0))
        initial_states[4] = N
    time_intervals = np.arange(0,t_final,(1/(sampling))) # time units are in msec, sampling in kHz, so 1000(msec)/sampling*1000 = samples per sec = 1/sampling
    ###### SIMPLE STOCHASTIC RULE####
    #_-----------------------------_#
    # for each time interval:
    #   take initial_states
    # run through probabilistic_walk,returning p_t only for period 1/(sampling)
    # draw N numbers and assign to statesaccording to draw>prob
    # this gives final states
    # which then used to initialise the next
    if simple:
        occupancies = np.zeros([np.size(Q,0),np.size(time_intervals,0),iterations])
        p_t = probabilistic_walk(N=N,t_final = t_final ,sampling=sampling,initial_states = initial_states,Vrev = Vrev,voltage = voltage,just_pt=True) #state occupancy probability at time t
        for iteration in np.arange(iterations):
            for interval, value in enumerate(time_intervals):
                if interval == 0: # time = 0, initial states are occupancies
                        occupancies[:,0,iteration] = initial_states
                else: # for all other time points
                    previous_states = occupancies[:,(interval-1),iteration] #take initial_states as occupancies at previous time points
                    occupancies[:,interval,iteration] = stochastic_advance(Q, p_t[:,interval], previous_states) # occupancies at end of next interval of the random walk 
        if iterations >1: # if multiple iterations performed
            occupancies = np.mean(occupancies,axis = 2) # return the average
    ##### RANDOM WALK WITH STOCHASTIC RULE#####
    #_---------------------------------------_#
    else: #for proper random walk
        # configuration of a Discrete-time time homogenous random walk
        # reconstruct Q matrix into a transition probability matrix
        transition_probabilities = Get_paths(Q,(1/sampling))
        occupancies = np.zeros([np.size(Q,0),np.size(time_intervals,0)+1,iterations])
        for iteration in np.arange(iterations):
            for intervalnum in np.arange(np.size(time_intervals,0)): #for each time interval
                if intervalnum ==0:
                    previous_states = initial_states
                    occupancies[:,intervalnum,iteration] = initial_states
                else:
                    previous_states = occupancies[:,(intervalnum - 1),iteration]
                occupancies[:,intervalnum+1,iteration] = stochastic_transitions(transition_probabilities,previous_states)
        # for more than a single iteration. For single iteration, simply makes 2D container.
        occupancies = np.mean(occupancies,axis = 2) # return the average
        #preallocating to store stochastic occupancy probability
        p_t = np.zeros([np.size(Q,0),np.size(time_intervals,0)])
        for item in np.arange(np.size(time_intervals,0)):
            p_t[:,item] = occupancies[:,item]/np.sum(occupancies[:,item])
    voltage = (voltage-Vrev) *10**-3 # Driving force in mV
    # N*g*(V-Vrev)*p_t
    if not np.any(conductances):
        conductances = np.array([3.7,16.1,30.6,38.6]) # conductances for GluA1 + y-2 used as default (1:4 stoichiometry)
    gv = conductances * voltage
    open_states = np.copy(occupancies[:4,:-1]) # to -1 to go form zero:t_final
    open_states[0,:] = open_states[0,:] * gv[0]
    open_states[1,:] = open_states[1,:] * gv[1]
    open_states[2,:] = open_states[2,:] * gv[2]
    open_states[3,:] = open_states[3,:] * gv[3]
    currents = np.sum(open_states,0)
    if not graphs:
        return(p_t,occupancies,currents)
    else:
        # if visualisation requested (graphs = True)
        plt.style.use('ggplot')
        figure,axes = plt.subplots(2,1)
        axes[0].plot(time_intervals,currents,color='black') # plot current
        axes[0].set_title("Simulated Current, N = {}, agonist conc = {} mM".format(N,agonist_conc))
        axes[0].set_xlabel("t (mS)")
        axes[0].set_ylabel("pA")
        # plotting occupancy probabilities over time
        for state in np.arange(np.size(p_t[:,0],axis=0)):
            axes[1].plot(time_intervals,p_t[state,:],label="{}".format(state))
            axes[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
            axes[1].set_title("P(State Occupancy at t), {}kHZ".format(sampling))
            axes[1].set_xlabel("t (mS)")
            axes[1].set_ylabel("Probability")
            plt.tight_layout()
        return(p_t,occupancies,currents)
        
        
#HERE#####
# so pretty cool, but for multiple transitions, might need to enact rule.
# e.g. that restricted by time constant over period time interval.
#       # easiest solution might be to say that from their previous state,
        # receptors are working towards next state, restricted by time
            #interval / sum (time constants) from prev to next state = fraction that make it
            #for state beneath, interval / sum(time_constants to transition beneath) =fraction that make it there.
            # etc, until we get some remaining in the current state
# but this defies the markovian assumption.
    # but, if multiple transitions occur in interval proportionally to interval/sum(time_constants) of the transition
        # then fastest is most likely when the interval is small - i.e. the time of the path between states is distributed how? Discrete exponential?
        # develop the maths and logic of this a bit.


def agonist_jump(N,first_conc,second_conc,onset_time,application_duration,record_length,sampling,Q = False,rise_time=400,decay_time=400,stochastic = False,graphs = False,voltage=False,Vrev=False,conductances=False):
    """Simulating a fast jump between two agonist concentrations.
    
    N: the number of receptors to simulate
    
    first_conc (mM): the concentration of the agonist before the jump
    
    second_conc (mM): the concentration of agonist during the jump
    
    onset_time (mS): the time to apply the jump
    
    application_duration (mS): the duration in second_conc
    
    record_length (mS): The length of the simulated trace
    
    sampling (kHz): sampling frequency (also determines the interval at which to
                    convert concentration to a step.)
    
    Q = A Q matrix. When False, uses rates from Coombs (2017)
    
    rise_time (uS): the 10-90% rise time fo the fast jump
    
    decay_time (uS): the 10-90% decay time of the jump
        
    stochastic (Default=False): When false, uses Colquhoun and Hawkes approach.
                                When True, performs a random walk (Non-functional in current implementation)
    
    graphs (Deafult=False): When = True, plots the current over time
    
    voltage(mV)
    
    Vrev(mV): The reversal potential of the channel
    
    conductances: By Deafult (False), uses conductances for A1 +y-2 (Coombs,2017)
        
    """
    if not Q:
        num_states = 17
    else:
        num_states= np.size(Q,1)
    dt = 1/sampling
    time_intervals = np.arange(0,record_length,dt)
    if not stochastic:
        # call embedded funciton to find occupancies until end of jump
        jump_occupancies = occupancies_in_jump(N=N,first_conc=first_conc,second_conc=second_conc,dt=dt,start_time=(onset_time+(0.5*application_duration)),duration = application_duration,rise_time=rise_time,decay_time=decay_time,num_states = num_states,Q=Q)
        # calculate how much time left
        remaining_time = (np.size(time_intervals)*dt)-(np.size(jump_occupancies,1)*dt)
        # then using last states of jump, relax to equilibrium
        #if not Q:
        #    _,remaining_occs,_ = probabilistic_walk(N,remaining_time,agonist_conc=first_conc,initial_states = jump_occupancies[:,-1])
        #else:
        _,remaining_occs,_ = probabilistic_walk(N,remaining_time,agonist_conc=first_conc,initial_states = jump_occupancies[:,-1],Q=Q)
        #concatenation
        occupancies = np.hstack((jump_occupancies,remaining_occs))
    if stochastic: # using random walk method
        print("The stochastic argument should not be used at present")
        jump_occupancies = occupancies_in_jump(N=N,first_conc=first_conc,second_conc=second_conc,dt=dt,start_time=onset_time,duration = application_duration,rise_time=rise_time,decay_time=decay_time,stochastic=True,Q=Q)
        remaining_time = (np.size(time_intervals)*dt)-(np.size(jump_occupancies,1)*dt)
        _,remaining_occs,_ = random_walk(N,remaining_time,agonist_conc=first_conc,initial_states = jump_occupancies[:,-1],Q=Q)
        occupancies = np.hstack((jump_occupancies,remaining_occs[:,:-1]))
    ###### getting curents form occupancies
    if not voltage:
        voltage = -60
    if not Vrev:
        Vrev = 0
    voltage = (voltage-Vrev) *10**-3 # Driving force in mV
    # N*g*(V-Vrev)*p_t
    if not np.any(conductances):
        conductances = np.array([3.7,16.1,30.6,38.6]) # conductances for GluA1 + y-2 used as default (1:4 stoichiometry)
    gv = conductances * voltage
    open_states = np.copy(occupancies[:4,:]) # to -1 to go form zero:t_final
    open_states[0,:] = open_states[0,:] * gv[0]
    open_states[1,:] = open_states[1,:] * gv[1]
    open_states[2,:] = open_states[2,:] * gv[2]
    open_states[3,:] = open_states[3,:] * gv[3]
    currents = np.sum(open_states,0)
    if not graphs:
        return(occupancies,currents)
    else:
                # if visualisation requested (graphs = True)
        plt.style.use('ggplot')
        figure,axes = plt.subplots(1,1)
        axes.plot(time_intervals,currents,color='black') # plot current
        axes.set_title("Simulated Current, N = {},sampling={} kHz".format(N,sampling))
        axes.set_xlabel("t (ms)")
        axes.set_ylabel("pA")
        return(occupancies,currents)

def Q_graph(Q):
    """Plots Q as a graph format (I.e. as a network)"""
    grp = nx.MultiDiGraph(Q)
    nx.draw(grp,pos = nx.spring_layout(grp),with_labels=True)

#__________EMBEDDED FUNCTIONS__________#
#______________________________________

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

def stochastic_advance(Q,occupancy_probability,initial_states):
    """embedded function that draws a random sample, and takes occupancy
    probability,initial states and transition matrix to establish which
    transitions occur; according to the rule:
    The next state for a given receptor has the lowest probability occupancy at
    the next time that exceeds a randomly drawn value when the two states are
    communicable. This violates the Markovian assumption, but reinforces the
    point that simple stochastic behaviour gives rise to the bursting
    propoerties of single channels.
    """
    net_transitions = np.zeros([np.size(initial_states,0),np.size(initial_states,0)]) #preallocation
    for state in np.where(initial_states>0)[0]: # for state i 
        # draw random sample of size = number of receptors initialised in that state
        #occupancy probability is exponentially distributed, so draw is random of the same distribution
        random_sample = np.random.exponential(scale = np.mean(occupancy_probability),size=int(initial_states[state])) 
        # where occupancy prob is a real transition describes both the probability of any transition,
        # and the probability that a transition is transition ->j
        # NB that this allows for multiple transitions.
        # preallocation:
        transitions = np.zeros(np.size(random_sample,0))
        bins = np.arange(np.size(initial_states,0))
        #
        for draw in np.arange(np.size(random_sample,0)): # for each random number
            if np.size(np.where(occupancy_probability>random_sample[0])[0],axis= 0) ==0: # if no (minimum occupancy probability > random draw)
                transitions[draw] = initial_states[state] #receptors stay in current state
            else:
                transitions[draw] = np.where(occupancy_probability == np.min(occupancy_probability[np.where(occupancy_probability>random_sample[0])]))[0] # row element of which entry of occupancy probability ==(minimum occupancy probability > random draw)
        transition_to = np.bincount(np.digitize(transitions,bins,right=True)) # entry is N receptors having transitioned to j
        transition_to.resize(np.size(Q,0))
        net_transitions[state,:] = transition_to
    next_states = np.sum(net_transitions,axis=0)
    return(next_states)

def stochastic_transitions(transition_probabilities,previous_states):
    """embedded function that takes states and a Q matrix where multiple transitions
    are considered to return occupancy at the next time point. Since transition
    probabilities are exponentially distributed, a transition is determined by:
    - drawing N random numbers (N=receptors in each state) from an exponential
    distribution of scale 1.
    - finding the transition of minimum probability that exceeds the random
      draw (for each random number) This is the transition that occurs.
    - if no number exceeds this value, or the state undergoes a multiple transition
      to its original state, then it dwells in this state.
     
    - To see how the probabilities of multiple transition are determined,
    see help(Get_paths)."""
    net_transitions = np.zeros([np.size(transition_probabilities,0),np.size(transition_probabilities,1)])
    for state in np.where(previous_states>0)[0]: #for each state populated by receptors
        number_in_state = previous_states[state] # number of receptors in that state
        # draw N samples from random exponential with scale = mean(prob[state])
        random_sample = np.random.exponential(scale = np.mean(transition_probabilities[state],axis=0),size=int(number_in_state)) #for scale, should be mean(transition_probabilities), 
        transitions = np.zeros(np.size(random_sample,0))
        bins = np.arange(np.size(previous_states,0))
        for draw in np.arange(np.size(random_sample,0)): # for each random number
            if np.size(np.where(transition_probabilities[state,:]>random_sample[draw])[0]) == 0:
                transitions[draw]==state #if no probability > draw, receptor remains in that state (NB, that with multiple transitions, may still)
            else:
                minprob = np.min(transition_probabilities[state,np.where(transition_probabilities[state,:]>random_sample[draw])[0]])
                transitions[draw] = np.where(transition_probabilities[state]==minprob)[0]
        transition_to = np.bincount(np.digitize(transitions,bins,right=True))
        transition_to.resize(np.size(transition_probabilities,0))
        net_transitions[state,:] = transition_to
    next_states = np.sum(net_transitions,axis=0)
    return(next_states)

def Get_paths(Q,dt,give_rates = False):
    """For a transition matrix, Q, finding the probability of all routes from
    state i to all other states, such that multiple transitions can occur
    in a discrete time interval.
    
    The rates are calculated to include direct rates (ij) and indirect rates.
    - All rates are uniformised to rate per time interval, dt
    
    - Indirect rates are calculated for each state by:
         - finding all paths from each state to each other state
         - finding the shortest route along each path from state i to each
           communicable state
         - taking the product of rates for each communicable state along this path
        
    - The probability of a transition, or multiple transitions from state i:j is
      calculated as Rateij/sum(rates leaving i)
    - Probability of dwelling is then 1-sum
    
    This is implemented with the network x package for python.
    
    See help(path_rates)
    
    Arguments
    -Q: A Q matrix
    -dt: the time difference of the interval
    -give_rates: (Default = False). When true, return the rates of multiple
     transitions per timestep.
    """
    ### Converting Q:
        # Scaling rates for time interval (ScQ)
    ScQ = np.copy(Q) # creating deep copy
    # scaling ScQ to rate per interval dt from mS rates
    ScQ = ScQ*(dt)
    ScQ[ScQ<=0]=0 # for now, ignoring transitions form state i -> i; and negating inf vals in empty  edges
    dt_adjusted_rates = nx.MultiDiGraph(ScQ) # creating directed graph, with directed edges specified by an entry in ij, and weight = Pij
    multiple_transition_rates = np.zeros([np.size(ScQ,0),np.size(ScQ,0)])
    # calculating rates of transition ij for each state i, and each transition j, where ij can be multiple transitions
    for state in dt_adjusted_rates.nodes(): # for each state node
        for descendent_node in np.arange((np.size(dt_adjusted_rates,0))-1,-1,-1):# for each of its communicable states (both direct and indirect)
            if descendent_node == state:
                multiple_transition_rates[state,descendent_node] = np.nan #making nan for time being
            else:
                path = np.array(nx.algorithms.shortest_path(dt_adjusted_rates,source = state,target=descendent_node)) # find the shortest path between states
                multiple_transition_rates[state,descendent_node] = path_rates(path,ScQ) # calculate the rate of this path I.e. rate of transition from each state to each other
    
    # calculating Pij = prob given receptor in state i that at next interval will be in j
    # accountign for multiple transitions
    # Pij = rate ij/sum(ratein)
    transition_probs = np.zeros([np.size(multiple_transition_rates,0),np.size(multiple_transition_rates,0)])
    for state in dt_adjusted_rates.nodes():
        sum_state = np.nansum(multiple_transition_rates[state],0)
        for transition in dt_adjusted_rates.nodes():
            # probability of each transition
            transition_probs[state,transition] = multiple_transition_rates[state,transition]/(sum_state)
        transition_probs[state,state] = 1-(np.nansum(transition_probs[state],0)) # fixing pseduo (dwell) transition rates from state into itself
    if not give_rates:
        return(transition_probs)
    else:
        return(multiple_transition_rates,transition_probs)
# 1/ rates in interval = fraction of an interval taken for a reaction to complete.
    # maybe just stick with probabilities, rather than time constant.
# rates = that many per mS
# Tau = how many per interval
# Thus, cosnider rates only where 1/rate >=1
    
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
            
        - duration (ms) the pulse half duration at half maximum (i.e. length of application)
        
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

def occupancies_in_jump(N,first_conc,second_conc,dt,start_time,duration,num_states=17,rise_time = 400,decay_time=400,initial_states = False,stochastic=False, Q = False):
    """Using concentration steps produced from a fast application
    to calculate occupancy from a Q matrix generated by from AMPAR_generator().
     
     Args:
         N: number of receptors.
         
         num_states (Default=17): the number of states in the Q matrix
         
         first_conc: can be single value, or list of values
         
         second_conc: can be single value, or list of values with the same
                      ordering as first conc of length = first conc.
        
        Q = A Q matrix (if false, uses Coombs 2017 as Default)
                          
         initial_states (Default = False): a vector of occpuancies of
                              length = num_states. If False, N receptors are
                              initialised in the resting state for the Q
                              matrix of AMPAR-generator - state 4.
                              
        stochastic (Default = False): When False, uses Colquhoun and Hawkes
                                      method. When True, performs random walks
                                      instead.                                       
        For other args, see help(concentration_as_steps)
     """
     #### Config #### 
    #rise_time = rise_time*10**-6 # depricated: enacted in concentration_as_steps
    #decay_time = decay_time*10**-6 # as above
    # get concentration steps before, and during the jump
    jump_concs,jump_times = concentration_as_steps(first_conc,second_conc,dt,start_time,duration,rise_time,decay_time)
    # preallocation to store p_t
    occs = np.zeros([num_states,np.size(jump_concs)])
    # finding occupancy probabilities (p_t) until jump occurs using concentration_as_steps
    for item, value in enumerate(jump_concs):
        # N.b, N is arbitrary, since only p_t returned.
        if item ==0:
            occs[4,item] = N # initialise receptors all in resting state
        else:
            if not stochastic:
                _,occ,_ = probabilistic_walk(N,2*dt,agonist_conc=value,initial_states = occs[:,item-1],Q = Q) # return occupancies at at next interval
                occs[:,item] = occ[:,1]  #([:,0] is t = 0).
            if stochastic:
                _,occ,_ = random_walk(N,2*dt,agonist_conc=value,initial_states = occs[:,item-1], Q = Q ) # return occupancies at at next interval
                occs[:,item] = occ[:,1]  #([:,0] is t = 0).
    return(occs)

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

def list_from_key(key):
    """embedded function to create list of rate constants from a key of type numpy.charray"""
    Q_key = key
    rate_constant_list = []
    for state in np.arange(np.size(Q_key,0)):
        for transition in np.arange(np.size(Q_key,0)):
            key = Q_key[state,transition]
            if "*" in key:
                constant = key.split("*")[1]
                if constant not in rate_constant_list: # if that rate constant not already in list, add it
                    rate_constant_list.append(constant)
    return(rate_constant_list)

def Q_from_key(Q_key,rate_constant_list,constants=False):
    """Takes a key [see A1_y2_key() for example], a list of rate constants
    [from list_from_key()] and returns Q.
    
    if constants is False, rate constants can be manually entered individually;
    though the user should account for any concentration or voltage-dependent
    effects on the values of any constant. 
    Alternatively, a custom dict can be provided, where each key is the name of
    a rate constant in the rate constant list, and each value is the associated 
    constant"""
    
    if not constants:
        rate_constants = {}
        print("If a rate constant is concentration-dependent, account for this in the value entered")
        for item in rate_constant_list:
            rate_constants[item] = float(input("Enter a value for rate constant {} >".format(item)))
    else:
        rate_constants = constants
    
    Q = np.zeros([np.size(Q_key,0),np.size(Q_key,0)])
    for state in np.arange(np.size(Q_key,0)):
        for transition in np.arange(np.size(Q_key,1)):
            if "*" in Q_key[state,transition]:
                multiple,constant = Q_key[state,transition].split("*")
                multiple = float(multiple)
                Q[state,transition] = multiple*(rate_constants[constant])
            elif "0" in Q_key[state,transition]:
                Q[state,transition] = 0
            elif "MR" in Q_key[state,transition]:
                Q[state,transition] = np.nan # for now, set nan
                
    for row in range(0,np.size(Q,axis=1)):
        Q[row,row] = - np.nansum(Q[row]) # replacing qii. Nb, that since Q[i,0:14] is empty, Q[0,0] = 0 is still satisfied
        
    # setting microscopically reversible rates
    protected_rates = np.zeros([np.size(Q,axis=0),np.size(Q,axis=1)])
    protected_rates = np.where(np.isnan(Q)==False,0.5,Q) # for constrained rates, set edge length  = 0.5
    protected_rates = np.where(np.isnan(protected_rates),1,protected_rates) # for rates to set by MR, = 1
    Q = make_Q_reversible(Q,protected_rates = protected_rates)
    return(Q)


#__________-------------------------------------------_________#    
#-------------- Rate key example for Q constructor -------------#
def A1_y2_key():
    Q = np.zeros([20,20])
    Q = Q.astype(np.chararray)
    Q[1,2],Q[2,3],Q[3,4],Q[2,1],Q[3,2],Q[4,3] = '3*k_on','2*k_on','1*k_on', 'MR','MR','MR'
    # indexing rates for gating: occupied (R0-4) to open (O1-4) transitions, in forward and reverse directions
    Q[6,1],Q[7,2],Q[8,3],Q[9,4],Q[1,6],Q[2,7],Q[3,8],Q[4,9] = '1*beta','2*beta','2*beta','4*beta','1*alpha','1*alpha','1*alpha','1*alpha'
    # indexing rates for ligand binding and unbinding (R0-R4)
    Q[5,6],Q[6,7],Q[7,8],Q[8,9],Q[6,5],Q[7,6],Q[8,7],Q[9,8] = '4*k_on','3*k_on','2*k_on','1*k_on','1*k_off','2*k_off','3*k_off','4*k_off'
    # indexing rates for transition to and from desensitised states (D1-4)
    Q[10,5],Q[11,6],Q[12,7],Q[13,8],Q[14,9],Q[5,10],Q[6,11],Q[7,12],Q[8,13],Q[9,14] = '1*gamma_0','1*gamma','1*gamma','1*gamma','1*gamma', '4*delta_0','1*delta_3','2*delta_1','3*delta_1','4*delta_1'
    # indexing rates for transitions between desensitised states
    Q[10,11],Q[11,12],Q[12,13],Q[13,14],Q[11,10],Q[12,11],Q[13,12],Q[14,13] = '3*k_on','3*k_on','2*k_on','1*k_on','1*k_off_2','1*k_off_3','2*k_off','3*k_off'
    # indexing rates for transitions to deep desensitised states
    Q[12,17],Q[13,18],Q[14,19],Q[17,12],Q[18,13],Q[19,14] = '1*delta_2','2*delta_2','3*delta_2','1*gamma_2','1*gamma_2','1*gamma_2'
    # indexing rates for transitions between deep desensitised states
    Q[17,18],Q[18,19],Q[18,17],Q[19,18] = '2*k_on','1*k_on','1*k_off','2*k_off'
    Q = np.delete(Q,[0,15,16],0)
    Q = np.delete(Q,[0,15,16],1)
    Q = np.where(Q==0,'0',Q)
    return(Q)

#_________---------------------------------------------_________#
#--------------------- Built-in Models--------------------------#
#---------_____________________________________________---------#
### consists of:
    # NASPM pore blocking:
         # Q matrix
         # Jumps into NASPM + glu at different voltages
         # " at different NASPM concs
         # entry to/exit from pore block jumps (Twomey, 2018)
    # Recovery from NBQX
        #Rosenmund, 1998; Coombs,2017
        
#_-------------------------------NASPM PORE BLOCKING------------------------_#
def NASPM_Q(glu_conc,NASPM_conc,voltage=-20,NASPM_k_on = False,NASPM_k_off=False,from_all_os=False,a=False,b=False,c=False,d=False):
    """Constructs the Q matrix for A1 + 4(y-2), using Coombs (2017)
    scheme with additional states for NASPM-blocked receptors.
    
    
    Evidence for initial states is considered from Twomey et al. (2018):
        - NASPM can only enter or exit the pore during the open state
          - Thus NASPM-blocked receptors can cycle through all other
            states.
          - Receptors can only move between blocked and unblocked cycles
            via the open states, where NASPM can only enter the fully-open pore, which
            is suggested by single exponential component (Bowie,1998; Twomey, 2018) 
            and because most receptors open fully (i.e. to O4, Coombs, 2017)
            but we also consider the alternative, which may be a necessary model compromise
        - Rates for entry and exit from NASPM-mediated pore blocked state,
          as well as IC50 values.
        - Voltage dependence of NASPM block, which is thought to occur primarily
          as a change in the dissociation, rather than association rate constant (Bowie,1998)
              - If rates NASPM_k_on and NASPM_k_off are not provided, they are calculated according to values of a,b,c,d:
              - A similar approach is adopted to Bowie et al, 1998 for introducing a voltage-dependent
                  component into the rate constant except the inclusion of permeation of the blocker through the channel
                  - we thus consider kd = kon/koff
                      kon = a*[NASPM]*exp(V/b), where a is uM^-1 s^-1, b in mV
                      koff = c*exp(V/d), where c,is dimensionless (mS^-1), d in mV
    
    Args:
        - glu_conc (mM): concentration of glutamate
        - NASPM_conc (uM): concentration of NASPM
        - voltage(mv): Default =
        - NASPM_on(ms^-1): NASPM association rate constant
        - NASPM_off (ms^-1): NASPM dissociation rate constant
        - from_all_os (Default =False): block entry occurs form all open states when True, or otherwise, only from O4
        - a: the constant associated with NASPM cocnentration
        - b: the voltage constant in exp(V/b) for NASPM association
        - c: constant for NASPM dissociation
        - d: the voltage constant in exp(V/d) for NASPM dissociation

    """
    voltage = voltage*10**-3 #converting mV to V
    if not from_all_os:
        Q = np.zeros([31,31])
        Q = Q.astype(np.chararray)
        Q[:17,:17] = A1_y2_key() # all a1 +y2 states for unblocked
        Q[18:,18:] = A1_y2_key()[4:,4:] # A1+y2 states (no open states) for blocked
        Q[4,17] = '1*NASPM_on' # single transition from O4: NASPM_blocked state
        Q[17,4] = '1*NASPM_off' # set NASPM_off by MR
    else:
        Q = np.zeros([34,34]) # transition from all open to all NASPM_blocked states
        Q = Q.astype(np.chararray)
        Q[:17,:17] = A1_y2_key()
        Q[21:,21:] = A1_y2_key()[4:,4:]
        Q[0,17],Q[1,18],Q[2,19],Q[3,20] = '4*NASPM_on','3*NASPM_on','2*NASPM_on','1*NASPM_on' # where O4:NASPMO4 is the major component
        Q[17,0],Q[18,1],Q[19,2],Q[20,3] = '1*NASPM_off','2*NASPM_off','3*NASPM_off', '4*NASPM_off'
    Q = np.where(Q==0,'0',Q)
    Q_key = np.copy(Q)
    # converting NASPM conc from uM to M
    NASPM_conc = NASPM_conc*10**-6
    # Glu_conc from mM to M
    glu_conc = glu_conc*10**-3
    #########RATE CONSTANTS in M, mS, V ########
    # if rates for entry and exit from block not provided, calculate them according to above
    # if provided, scale with concentration
    if not NASPM_k_on:
        a=a*NASPM_conc
        NASPM_k_on = a*np.exp((voltage/b))
    else:
        NASPM_k_on = NASPM_k_on*(NASPM_conc/4.8*10**-6)
    if not NASPM_k_off:
        NASPM_k_off = c*np.exp((voltage/d))
    else:
        NASPM_k_off = NASPM_k_off*(NASPM_conc/4.8*10**-6)
    # remaining constants used from Coombs(2017)
    # then construct dictionary with name pair values
    rate_constant_list = list_from_key(Q_key)
    # building the rate constant dict
    constant_dict = {}
    constant_dict['k_on'],constant_dict['k_off'],constant_dict['alpha'],constant_dict['beta'],constant_dict['gamma'],constant_dict['gamma_0'] = (1.3*10**4)*glu_conc,3,1,6,0.016,0.0044 
    constant_dict['delta_0'],constant_dict['delta_1'],constant_dict['delta_2'],constant_dict['delta_3'],constant_dict['k_off_2'],constant_dict['k_off_3'],constant_dict['gamma_2'] = 0.00048,1.2,1.3,0.25, 0.063,0.63,3.9
    constant_dict['NASPM_on'],constant_dict['NASPM_off'] = NASPM_k_on,NASPM_k_off
    # construct Q matrix, enforcing microscopic reversibility where appropriate
    NASPM_Q = Q_from_key(Q_key, rate_constant_list,constants = constant_dict)
    NASPM_Q = np.where(np.isnan(NASPM_Q)==True,0,NASPM_Q)
    return(NASPM_Q)

    
def NASPM_V_ramp(N,glu_conc,NASPM_conc,startv,endv,a,b,c,d,sampling = 20,open_states_all = False):
    """Perfoms continuous coapplication of glutamate and NASPM over a range of voltages
        at every 10mV between startv and endv
    
    Args:
        N: number of receptors
        glu_conc(mM)
        NASPM_conc(uM)
        startv (mV), the first voltage
        endv (mV), the final voltage
        a, the rate constant for the concentration-dependent component of the
            NASPM association rate
        b, the rate constant for the voltage-dependent component of the
            NASPM association rate
        c,the rate constant for the voltage-INDEPENDENT component of the
            NASPM dissociation rate
        d,the rate constant for the voltage-DEPENDENT component of the
            NASPM dissociation rate
        sampling(kHz): The sampling frequency
        open_states_all(Default=False). When True, pore blocking occurs from each
            open state. When False, only occurs from O4.
        """
    # config
    voltages = np.arange(startv,(endv+10),10) # voltages to step over
    t_intervals = np.arange(0,500,1/sampling)
    conductances = np.array([3.7,16.1,30.6,38.6]) # conductances for GluA1 + y-2 used as default (1:4 stoichiometry)
    # preallocation
    NASPM_currents = np.zeros([np.size(t_intervals),np.size(voltages)])
    #glu_currents = np.copy(NASPM_currents) DEPRICATED
    NASPM_ss = np.zeros(np.size(voltages))
    glu_ss = np.zeros(np.size(voltages))
    # plot preparation
    plt.style.use('ggplot')
    vfig,vaxs = plt.subplots(2,1,figsize = (10,10))
    vaxs[0].set_xlabel('Time (mS)')
    vaxs[0].set_ylabel('I (pA)')
    vaxs[0].set_title('Co-application of {}mM Glu + {}uM NASPM,N ={},at {}kHZ'.format(glu_conc,NASPM_conc,N,sampling))
    vaxs[1].set_xlabel('Voltage (mV)')
    vaxs[1].set_ylabel('I (pA)')
    ### performing simulation with C&H method for Glu or Glu + NASPM at each voltage step.
    for step, voltage in enumerate(voltages):
        NASPMQ = NASPM_Q(glu_conc = glu_conc, NASPM_conc = NASPM_conc,voltage = voltage,from_all_os = open_states_all,a=a,b=b,c=c,d=d)
        gluQ = AMPAR_generator(glu_conc=glu_conc) # voltage term in probabilistic walk, below
        _,_,NASPM_currents[:,step] = probabilistic_walk(N=N, t_final=500, voltage=voltage,agonist_conc = glu_conc,sampling = sampling,Q=NASPMQ)
        #_,_,glu_currents[:,step] = probabilistic_walk(N=N,t_final = 500,voltage=voltage,agonist_conc = glu_conc,sampling = sampling,Q=gluQ) #DEPRICATED
        vaxs[0].plot(t_intervals,NASPM_currents[:,step],label='{}mV'.format(voltage))
        ssN = p_inf(NASPMQ) #steady-state
        ssN = ssN*N
        NASPM_ss[step] = ss_to_curr(ssN,conductances,voltage) # using embedded to get current from steady-state occupancies
        ssG = p_inf(gluQ)
        ssG = ssG*N
        glu_ss[step] = ss_to_curr(ssG,conductances,voltage)
    vaxs[1].scatter(voltages,NASPM_ss,color='orange',marker='o',label='NASPM + Glu')
    vaxs[1].scatter(voltages,glu_ss,color='black',marker='o',label = 'Glu')
    vaxs[0].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
    vaxs[1].legend(fontsize=5)
    return()

#NB, for fitting, have two depricated glu_currents lines

### HERE, using scheme below, replot NASPM voltage ramp with jumps
# will need to think about how to get and plot the steady state val.
#       may be easier to get in analysis.

# will need to get jump concs first
# and voltage range
# then loop through voltages, rather than concs
# move voltage and conductance calculation to in loop
# need to also get glu response alone at each voltage

def NASPM_V_jumps(N,glu_conc,NASPM_conc,startv,endv,a,b,c,d,jump_duration=100,sampling = 20,start_time = 5,rise = 400,decay = 400,record_length = 500,open_states_all = False):
    """
    Parameters
    ----------
    N : TYPE
        Number of receptors to simulate
    glu_conc (mM) : TYPE
        The glutamate concentration to jump into
    NASPM_conc (uM) : TYPE
        NASPM concentration to jump into
    startv (mV) : TYPE
        The voltage to start at. Incrementally increases by 10mV until endv
    endv : TYPE
        The voltage to end at.
    a : TYPE
        NASPM rate parameter (concentration-dependent component for association)
    b : TYPE
        NASPM rate parameter (voltage-dependent component for association)
    c : TYPE
        NASPM rate parameter (dimensionless for dissociation)
    d : TYPE
        NASPM rate parameter (voltage-dependent component for dissociation)
    jump_durations (mS) : float. Value > 0
        The duration of the fast jump . The Default is 10.
    sampling (kHz): 
        Sampling frequeny The default is 20.
    open_states_all : Boolean
        Whether to allow entry to and exit from pore blocking from all open
        states, or just O4 when False. The default is False.
    start_time (mS): 
        The delay before the jump. The default is 5.
    rise(uS):
        rise time of the fast jump onset
    decay(uS):
        decay time of fast jump offset
    record_length(mS):
        The length of the 'record'

    Returns
    -------
    voltage_range,t_intervals,glu_currents, NASPM currents
    Each row of currents is a different Vh (holding voltage)
    Each column corresponds to a time point in t_intervals

    """
    #Runtime warning
    print("Runtime is slow, and increases with a higher sampling frequency and jump duration. A progress indicator will update periodically")
    ### config
    if not open_states_all:
        n_states = 31
    else:
        n_states = 34
    conductances = np.array([3.7,16.1,30.6,38.6]) # conductances for GluA1 + y-2 used as default (1:4 stoichiometry)
    ### getting the different concentrations, starting at 0
    voltage_range = np.arange(startv,endv+10,10) # 10mV steps
    # getting jump concs for NASPM and glu
    jump_concs,jump_times = concentration_as_steps(0,second_conc = NASPM_conc,dt = 1/sampling,start_time = start_time,rise_time = rise,decay_time = decay,duration = jump_duration)
    glu_jump_concs,glu_jump_times = concentration_as_steps(0,second_conc =glu_conc,dt = 1/sampling,start_time = start_time,rise_time = rise,decay_time = decay,duration = jump_duration)
    # loop: for each voltage:
        # perform a fast jump into Glu+NASPM or Glu
        # then enter relxation
    #preallocating to collect currents
    for step, voltage_h in enumerate(voltage_range):
        print("Progress at",100*(np.round(step/np.size(voltage_range),2)),"%")
        # preallocation to store jump occupancies
        occs = np.zeros([n_states,np.size(jump_times)])
        occs[4,0] = N
        glu_occs = np.zeros([n_states,np.size(jump_times)])
        glu_occs[4,0] = N
        for jumpnum, jump_concentration in enumerate(jump_concs):
            if jump_times[jumpnum] == 0:
                initial_states_NASPM = occs[:,0]
                initial_states_glu = glu_occs[:,0]
            else:
                initial_states_NASPM = occs[:,jumpnum-1]
                initial_states_glu = glu_occs[:,jumpnum-1]
            NaspmQ = NASPM_Q(glu_conc=glu_jump_concs[jumpnum], NASPM_conc = jump_concs[jumpnum],voltage = voltage_h,from_all_os = open_states_all,a=a,b=b,c=c,d=d)
            glu_Q = NASPM_Q(glu_conc=glu_jump_concs[jumpnum],NASPM_conc=0,voltage=voltage_h,from_all_os=open_states_all,a=a,b=b,c=c,d=d)
            _,occupancies,_ = probabilistic_walk(N,2/sampling,voltage=voltage_h,agonist_conc = glu_jump_concs[jumpnum],sampling = sampling,Q=NaspmQ,initial_states= initial_states_NASPM)
            _,glu_occupancies,_ = probabilistic_walk(N,2/sampling,voltage=voltage_h,agonist_conc = glu_jump_concs[jumpnum],sampling = sampling,Q=glu_Q,initial_states= initial_states_glu)
            occs[:,jumpnum] = occupancies[:,-1]
            glu_occs[:,jumpnum] = glu_occupancies[:,-1]
        # then perform relaxation for the remaining time
        NaspmQ = NASPM_Q(glu_conc = 0,NASPM_conc = 0,voltage=voltage_h,from_all_os = open_states_all,a=a,b=b,c=c,d=d)
        glu_Q = NASPM_Q(glu_conc = 0,NASPM_conc = 0,voltage=voltage_h,from_all_os = open_states_all,a=a,b=b,c=c,d=d)
        _,remaining_occs,_ = probabilistic_walk(N,t_final = (record_length-jump_times[-1]),voltage=voltage_h,agonist_conc = 0,sampling = sampling,Q=NaspmQ,initial_states=occs[:,-1])
        _,remaining_glu_occs,_ = probabilistic_walk(N,t_final = (record_length-jump_times[-1]),voltage=voltage_h,agonist_conc = 0,sampling = sampling,Q=glu_Q,initial_states=glu_occs[:,-1])
        occupancies_throughout = np.append(occs,remaining_occs,axis=1)   
        glu_occupancies_throughout = np.append(glu_occs,remaining_glu_occs,axis=1) 
        # getting current
        os_throughout = np.copy(occupancies_throughout[:4,:])
        os_glu = np.copy(glu_occupancies_throughout[:4,:])
        voltage_p = voltage_h*10**-3 # mV copy for gv. Other used in functions
        gv = conductances * voltage_p
        os_throughout[0,:] = os_throughout[0,:] *gv[0]
        os_throughout[1,:] = os_throughout[1,:] *gv[1]
        os_throughout[2,:] = os_throughout[2,:] *gv[2]
        os_throughout[3,:] = os_throughout[3,:] *gv[3]
        os_glu[0,:] = os_glu[0,:] *gv[0]
        os_glu[1,:] = os_glu[1,:] *gv[1]
        os_glu[2,:] = os_glu[2,:] *gv[2]
        os_glu[3,:] = os_glu[3,:] *gv[3]
        current = np.nansum(os_throughout,axis=0)
        glu_current = np.nansum(os_glu,axis=0)
        if step ==0:
            currents = np.zeros([np.size(voltage_range),np.size(current)])
            currents[0,:] = current
            glu_currents = np.zeros([np.size(voltage_range),np.size(glu_current)])
            glu_currents[0,:] = glu_current
        else:
            currents[step,:] = current
            glu_currents[step,:] = glu_current
    # plotting in separate loop to aid performance
    print("Plotting Graph")
    t_intervals = (np.arange(np.size(currents,axis=1)))*(1/sampling)
    plt.style.use('ggplot')
    vfig,vaxs = plt.subplots(2,1,figsize=(10,10))
    vaxs[0].set_title('Voltage Ramp:{}uM NASPM + {}mM Glu, {}kHZ'.format(NASPM_conc,glu_conc,sampling))
    vaxs[1].set_title('Voltage Ramp:{}mM Glu, {}kHZ'.format(glu_conc,sampling))
    vaxs[0].set_xlabel('t(mS)')
    vaxs[1].set_xlabel('t(mS)')
    vaxs[0].set_ylabel('I (pA)')
    vaxs[1].set_ylabel('I (pA)')
            ############
    for step, voltage_h in enumerate(voltage_range):
        vaxs[0].plot(t_intervals,currents[step,:],label = "Vh = {}mV".format(voltage_h))
        vaxs[1].plot(t_intervals,glu_currents[step,:],label = "Vh = {}mV".format(voltage_h))
    vaxs[0].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
    vaxs[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
    vfig.tight_layout()
    return(voltage_range,t_intervals,glu_currents,currents)
    #return(voltage_range,t_intervals,currents)


def NASPM_conc_response_jump(N,glu_conc,voltage,a,b,c,d,NASPM_conc_start=0.1,NASPM_conc_end=10.5,jump_duration = 10,sampling = 20,start_time = 5,rise = 400,decay = 400,record_length = 500,open_states_all = False):
    """
    Simulating currents in response to jumps into glutamate and NASPM
    Glutamate concentration is fixed, and NASPM concentration varies over
    a logarithmic range.
    
    Runtime is slow

    Parameters
    ----------
    N : int
        Number of receptors
    glu_conc (mM) : float
        Glutamate concentration during continuous application
    voltage (mV) :
        Voltage at which to perform
    a : TYPE
        NASPM rate parameter (concentration-dependent component for association)
    b : TYPE
        NASPM rate parameter (voltage-dependent component for association)
    c : TYPE
        NASPM rate parameter (dimensionless for dissociation)
    d : TYPE
        NASPM rate parameter (voltage-dependent component for dissociation)
    NASPM_conc_start (uM) : float. Value > 0
        smallest concentration of NASPM The default is 0.1. 10 concentrations
        are calculated over a logarithmic range between start and end.
    NASPM_conc_end (uM) : float. Value > 0
        largest concentration of NASPM. The default is 10.5.
    jump_durations (mS) : float. Value > 0
        The duration of the fast jump . The Default is 10.
    sampling (kHz): 
        Sampling frequeny The default is 20.
    open_states_all : Boolean
        Whether to allow entry to and exit from pore blocking from all open
        states, or just O4 when False. The default is False.
    start_time (mS): 
        The delay before the jump. The default is 5.
    rise(uS):
        rise time of the fast jump onset
    decay(uS):
        decay time of fast jump offset
    record_length(mS):
        The length of the 'record'
    

    Returns
    -------
    Graphs for current responses at each concentration in the selected range
    and a log concentration-response graph.
    
    t_intervals,concs,currents,IC50,fraction_of_peak,peaks
    Each row of currents is a different concentration of NASPM
    Each column corresponds to a time point in t_intervals
    
"""
    #Runtime warning
    print("Runtime is slow, and increases with a higher sampling frequency and jump duration. A progress indicator will update periodically")
    ### config
    if not open_states_all:
        n_states = 31
    else:
        n_states = 34
    conductances = np.array([3.7,16.1,30.6,38.6]) # conductances for GluA1 + y-2 used as default (1:4 stoichiometry)
    voltage_p = voltage*10**-3 # mV copy for gv. Other used in functions
    gv = conductances * voltage_p
    ### getting the different concentrations, starting at 0
    concs = 0
    concs = np.append(concs,np.logspace(np.log10(NASPM_conc_start),np.log10(NASPM_conc_end),10))
    
    # loop: for each concentration in the NASPM concentraiton-response curve
        # perform a fast jump
        # then enter relxation
    #preallocating to collect currents
    for step, concentration in enumerate(concs):
        print("Progress at",100*(np.round(step/np.size(concs),2)),"%")
        # getting NASPM concs
        jump_concs,jump_times = concentration_as_steps(0,second_conc = concentration,dt = 1/sampling,start_time = 5,rise_time = rise,decay_time = decay,duration = jump_duration)
        glu_jump_concs,glu_jump_times = concentration_as_steps(0,second_conc =glu_conc,dt = 1/sampling,start_time = 5,rise_time = rise,decay_time = decay,duration = jump_duration)

        # preallocation to store jump occupancies
        occs = np.zeros([n_states,np.size(jump_times)])
        occs[4,0] = N
        for jumpnum, jump_concentration in enumerate(jump_concs):
            if jump_times[jumpnum] == 0:
                initial_states = occs[:,0]
            else:
                initial_states = occs[:,jumpnum-1]
            NASPMQ = NASPM_Q(glu_conc=glu_jump_concs[jumpnum],NASPM_conc=jump_concentration,voltage=voltage,from_all_os=open_states_all,a=a,b=b,c=c,d=d)
            _,occupancies,_ = probabilistic_walk(N,2/sampling,voltage=voltage,agonist_conc = glu_jump_concs[jumpnum],sampling = sampling,Q=NASPMQ,initial_states= initial_states)
            occs[:,jumpnum] = occupancies[:,-1]
        # then perform relaxation for the remaining time
        NASPMQ = NASPM_Q(glu_conc = 0,NASPM_conc = 0,voltage=voltage,from_all_os = open_states_all,a=a,b=b,c=c,d=d)
        _,remaining_occs,_ = probabilistic_walk(N,t_final = (record_length-jump_times[-1]),voltage=voltage,agonist_conc = 0,sampling = sampling,Q=NASPMQ,initial_states=occs[:,-1])
        occupancies_throughout = np.append(occs,remaining_occs,axis=1)   
        # getting current
        os_throughout = np.copy(occupancies_throughout[:4,:])
        os_throughout[0,:] = os_throughout[0,:] *gv[0]
        os_throughout[1,:] = os_throughout[1,:] *gv[1]
        os_throughout[2,:] = os_throughout[2,:] *gv[2]
        os_throughout[3,:] = os_throughout[3,:] *gv[3]
        current = np.nansum(os_throughout,axis=0)
        if step ==0:
            currents = np.zeros([np.size(concs),np.size(current)])
            currents[0,:] = current
        else:
            currents[step,:] = current
    # getting peak and plotting graphs. Loop split to aid storage
    print("finding peaks and plotting Graphs...")
    peaks = np.zeros(np.size(concs))
    t_intervals = (np.arange(np.size(currents,axis=1)))*(1/sampling)
    plt.style.use('ggplot')
    cplot,caxs = plt.subplots(2,1,figsize=(10,10))
    caxs[0].set_title('[NASPM] (uM) + {}mM Glu, {}kHZ'.format(glu_conc,sampling))
    caxs[0].set_xlabel('t(mS)')
    caxs[0].set_ylabel('I (pA)')
    caxs[1].set_xscale('log')
    caxs[1].set_xlabel("log [NASPM]")
    caxs[1].set_ylabel('Ipeak (pA)') 
    # getting IC50
    for item, concentration in enumerate(concs):
        #find peak, if negative or positive
        if abs(np.min(currents[item,:])) > np.max(currents[item,:]):
            #inward current
            peaks[item] = np.min(currents[item,:])
        else:
            peaks[item] = np.max(currents[item,:])
        # plotting currents
        caxs[0].plot(t_intervals,currents[item,:],label = "{}uM NASPM".format(np.round(concentration,3)))
    
    # getting IC50
    EC50 = 0.5*peaks[0]#0 concentration NASPM in glu
    #IC50,fit = IC50_from_EC50(EC50,peaks[1:],concs[1:]) DEPRICATED, fitting by fraction
    fraction_of_peak = peaks[1:]/peaks[0]
    # plotting fraction of peak current
    IC50,fit = IC50_from_EC50(EC50, peaks[1:], concs[1:])
    #caxs[1].plot(concs,fit(concs),linestyle="--")---------------- DEPRICATED
    caxs[1].scatter(concs[1:],fraction_of_peak[:],marker="o")
    caxs[0].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
    caxs[1].set_title('log[NASPM]uM + {}mM Glu vs Ipeak(pA), IC50={}uM'.format(glu_conc,np.round(IC50,3)))
    cplot.tight_layout()
    return(t_intervals,concs,currents,IC50,fraction_of_peak,peaks)


def NASPM_entry_exit_block_jumps(N,glu_conc,NASPM_conc,voltage,a,b,c,d,t_final = 50000,sampling = 10,open_states_all=False,rise=400,decay=400,exit_from=False):
    """
    Performs Twomey(2010) entry to and exit from NASPM pore block protocols.
    By default, only performs entry to (see exit_from arg)
    Runtime is approximately 18 minutes per protocol on a Quad-Core Unix-based
    computer with processor operating at 1.3GHz clock speed.
    
    Performance may vary on your system. As such, A progress reporter occasionally
    reports at 0,30,60,and 88% complettion for each protocol.

    Parameters
    ----------
    N : TYPE
        Number of receptors to simulate
    glu_conc (mM) : TYPE
        Millimolar concentration of glutamate
    NASPM_conc (uM) : TYPE
        Micromolar concentration of NASPM.
    voltage (mV) : TYPE
        Voltage at which to perform the protocl.
    a : TYPE
        NASPM rate parameter (concentration-dependent component for association)
    b : TYPE
        NASPM rate parameter (voltage-dependent component for association)
    c : TYPE
        NASPM rate parameter (dimensionless for dissociation)
    d : TYPE
        NASPM rate parameter (voltage-dependent component for dissociation)
    t_final (mS) : TYPE, optional
        The record duration. The default is 50000.
    sampling (kHZ) : TYPE, optional
        Sampling frequency. The default is 10.
    open_states_all : True/False
        Determines whether to use the model for entry to blocking from all
        open states(=True), or only O4(=False). The default is False.
    rise : TYPE, optional
        rise time of the open tip junction potential. The default is 400.
    decay : TYPE, optional
        Decay time of the open-tip junction potential. The default is 400.
    exit_from : True/False
        Determines whether exit from block performed after entry to block
        protocol. The default is False.

    Returns
    -------
    if exit_from = True:
        occupancies during block entry,currents during block entry,occsupancies during bloc exit,currents during block exit
    if exit_from = False:
        occupancies during block entry,currents during block entry

    """
    #___ config
    if not open_states_all:
        n_states = 31
    else:
        n_states = 34
    conductances = np.array([3.7,16.1,30.6,38.6]) # conductances for GluA1 + y-2 used as default (1:4 stoichiometry)
    gv = conductances * voltage
    t_intervals = np.arange(0,t_final,1/sampling)
    jump_in_times = np.arange(0,t_final,1010) #times for entry to jump
    concs_during_jump,times = concentration_as_steps(0,glu_conc,1/sampling,5,10,rise_time=rise,decay_time=decay)
    jump_out_times = jump_in_times+times[-1] # getting times for exit from a jump into a relxation
    occs = np.zeros([n_states,1])
    NASPMQ_jump = np.zeros([n_states,n_states,np.size(times)])
    ####### Preparation
    #AOT Q matrices
    #Q matrix during jump
    for interval, time in enumerate(times):
        NASPMQ_jump[:,:,interval] = NASPM_Q(glu_conc = concs_during_jump[interval],NASPM_conc=NASPM_conc,voltage=voltage,a=a,b=b,c=c,d=d,from_all_os=open_states_all)
    #Q matrix during relaxation
    NASPMQ_relax = NASPM_Q(glu_conc=0,NASPM_conc=NASPM_conc,voltage=voltage,a=a,b=b,c=c,d=d,from_all_os=open_states_all)
    ## preallocation for loop
    occup_during_jump = np.zeros([n_states,np.size(times)])
    # t = zero states
    occs[4,0] = N
    for jumpnum, jump_time in enumerate(jump_in_times):
        if jumpnum%10 ==0:
            print("Entry to block progress at",(jumpnum/np.size(jump_in_times))*100,"%")
        if jumpnum ==0:
            initial_state = occs[:,0]
        else:
            initial_state = occs[:,-1]
        for interval, time in enumerate(times):
            _,occup,_ = probabilistic_walk(N,t_final = 2/sampling,voltage=voltage,agonist_conc = glu_conc,sampling = sampling,Q=NASPMQ_jump[:,:,interval],initial_states = initial_state )
            occup_during_jump[:,interval] = occup[:,-1]
            occs = np.append(occs,occup_during_jump,axis=1)
        if jumpnum <= (np.size(jump_in_times))-1:
            _,relax_occup,_ = probabilistic_walk(N,t_final = (jump_in_times[jumpnum+1]-jump_out_times[jumpnum]),voltage=voltage,agonist_conc=0,sampling = sampling,Q=NASPMQ_relax,initial_states = occup_during_jump[:,-1])
        else:
            _,relax_occup,_ = probabilistic_walk(N,t_final = (t_final-jump_out_times[jumpnum]),voltage=voltage,agonist_conc=0,sampling = sampling,Q=NASPMQ_relax,initial_states = occup_during_jump[:,-1])
        relax_occup = relax_occup[:,1:]
        occs = np.append(occs,relax_occup,axis=1)
    #### getting currents
    entry_os = occs[:4,:]
    currents = np.copy(entry_os) #deep copy
    currents[0,:] = currents[0,:]*gv[0]
    currents[1,:] = currents[1,:]*gv[1]
    currents[2,:] = currents[2,:]*gv[2]
    currents[3,:] = currents[3,:]*gv[3]
    entry_current = np.nansum(currents,axis=0)
    plt.style.use('ggplot')
    # do the same for exit
    if exit_from:
        occs_exit = np.zeros([n_states,1])
        occs_exit[:,0] = occs[:,-1] #initialise with final states of entry to block
        for interval, time in enumerate(times):
            NASPMQ_jump[:,:,interval] = NASPM_Q(glu_conc = concs_during_jump[interval],NASPM_conc=0,voltage=voltage,a=a,b=b,c=c,d=d,from_all_os=open_states_all)
        NASPMQ_relax = NASPM_Q(glu_conc=0,NASPM_conc=0,voltage=voltage,a=a,b=b,c=c,d=d,from_all_os=open_states_all)
        for jumpnum, jump_time in enumerate(jump_in_times):
            if jumpnum%10 ==0:
                print("Exit from block jumps progress at",(jumpnum/np.size(jump_in_times))*100,"%")
            if jumpnum ==0:
                    initial_state = occs_exit[:,0]
            else:
                    initial_state = occs_exit[:,-1]
            for interval, time in enumerate(times):
                _,occup,_ = probabilistic_walk(N,t_final = 2/sampling,voltage=voltage,agonist_conc = glu_conc,sampling = sampling,Q=NASPMQ_jump[:,:,interval],initial_states = initial_state )
                occup_during_jump[:,interval] = occup[:,-1]
                occs_exit = np.append(occs_exit,occup_during_jump,axis=1)
            if jumpnum <= (np.size(jump_in_times))-1:
                _,relax_occup,_ = probabilistic_walk(N,t_final = (jump_in_times[jumpnum+1]-jump_out_times[jumpnum]),voltage=voltage,agonist_conc=0,sampling = sampling,Q=NASPMQ_relax,initial_states = occup_during_jump[:,-1])
            else:
                _,relax_occup,_ = probabilistic_walk(N,t_final = (t_final-jump_out_times[jumpnum]),voltage=voltage,agonist_conc=0,sampling = sampling,Q=NASPMQ_relax,initial_states = occup_during_jump[:,-1])
            relax_occup = relax_occup[:,1:]
            occs_exit = np.append(occs_exit,relax_occup,axis=1)     
                # get exit currents
        exit_os = occs_exit[:4,:]
        exit_currents = np.copy(exit_os) #deep copy
        exit_currents[0,:] = exit_currents[0,:]*gv[0]
        exit_currents[1,:] = exit_currents[1,:]*gv[1]
        exit_currents[2,:] = exit_currents[2,:]*gv[2]
        exit_currents[3,:] = exit_currents[3,:]*gv[3]
        exit_current = np.nansum(exit_currents,axis=0)
        ## plot both entry to block and exit from currents
        jumpfig,jumpax = plt.subplots(1,2)
        jumpax[0].plot(t_intervals,entry_current)
        jumpax[0].set_xlabel("time (mS)")
        jumpax[0].set_ylabel("Current (pA)")
        jumpax[0].set_title("Entry to block, {}mM Glu, {}uM NASPM".format(glu_conc,NASPM_conc))
        #
        jumpax[1].plot(t_intervals,exit_current)
        jumpax[1].set_xlabel("time (mS)")
        jumpax[1].set_ylabel("Current (pA)")
        jumpax[1].set_title("Exit from block, {}mM Glu".format(glu_conc))
        jumpfig.tight_layout()
        return(occs,entry_current,occs_exit,exit_current)
    else:
        jumpfig,jumpax = plt.subplots(1,1)
        jumpax.plot(t_intervals,entry_current)
        jumpax.set_xlabel("time (mS)")
        jumpax.set_ylabel("Current (pA)")
        jumpax.set_title("Entry to block, {}mM Glu, {}uM NASPM".format(glu_conc,NASPM_conc))
        jumpfig.tight_layout()
        return(occs,entry_current)



def IC50_from_EC50(EC50_response,peak_currents,concs):
    """Calculates IC50 of antagonist/blocker using EC50 value from antagonist-lacking condition"""
    fit_obj = np.polyfit(peak_currents,np.log10(concs),3)
    fit = np.poly1d(fit_obj)
    IC50 = fit(EC50_response)
    fit_resp_to_conc = np.polyfit(np.log10(concs),peak_currents,3)
    concfit = np.poly1d(fit_resp_to_conc)
    return(IC50,concfit)
    
# test abcd  = 10,100,200,-20


def ss_to_curr(SS,conductances,voltage):
    """current at steady state from steady_state occupancies"""
    gv = conductances * voltage
    open_states = SS[:4]
    open_states[0] = open_states[0]*gv[0]
    open_states[1] = open_states[1]*gv[1]
    open_states[2] = open_states[2]*gv[2]
    open_states[3] = open_states[3]*gv[3]
    current = np.nansum(open_states)
    return(current)

def multiple_jump_concentrations(t_final,jump_times,first_conc,second_conc,dt,duration,rise_time,decay_time,sampling):
    """accepts numpy.array of first dimension zero, containing times of all
    of the jumps
    requires other arguments for concentration_as_steps"""
    # duplicates jump every x
    t = np.arange(0,t_final,dt)
    concentrations = np.zeros(np.size(np.arange(0,t_final,dt)))
    concs,times = concentration_as_steps(first_conc=first_conc,second_conc=second_conc,dt =dt,start_time = (jump_times[0]+(0.5*duration)+(rise_time*10**-3)),duration=duration,rise_time=rise_time,decay_time=decay_time)
    interval = times*(1/dt)
    for jump_num, jump_time in enumerate(jump_times):
        each_interval = (interval+(jump_time*(1/dt))).astype(int)
        start = np.min(each_interval)
        end = np.max(each_interval)
        concentrations[start:end] = concs[:-1]
    return(concentrations,t)


#_-----------------------------NBQX Recovery---------------------------------_#
def NBQX_Q(glu_conc,NBQX_conc,Tau_off = 200):
    """Generates Q matrix for recovery from NBQX antagonism (Rosenmund,1998; Coombs,2017)
    glutamate concentration is set by the protocol.
    NBQX on rate set by MR, but usually receptors start in NQBX- bound state anyway
    
    NBQX_conc is set by protocol(in uM)
    
    Tau off refers to the fitted Tau for the Poisson jump process. By Deafult = 200mS"""
    #NBQX conc and glu conc both assumed to be max (10mM,50uM)
    
    Q = np.zeros([21,21]) # transition from all open to all NASPM_blocked states
    Q = Q.astype(np.chararray)
    Q[:17,:17] = A1_y2_key()
    Q[0,17],Q[1,18],Q[2,19],Q[3,20] = 'MR','MR','MR','MR' # where O4:NASPMO4 is the major component
    Q[17,0],Q[18,1],Q[19,2],Q[20,3] = '1*NBQX_off','2*NBQX_off','3*NBQX_off','4*NBQX_off'
    Q = np.where(Q==0,'0',Q)
    Q_key = np.copy(Q)
    # glu_conc from 10mM to M
    glu_conc = glu_conc*10**-3
    # NBQX_conc to uM
    NBQX_conc = NBQX_conc*10**-6
    # converting NBQX conc from mM to M, and scaling rate: rate = concentration/ICmax
    NBQX_off = (1/Tau_off)*(NBQX_conc/50*10**-6)
    #########RATE CONSTANTS in M, mS ########
    # remaining constants used from Coombs(2017)
    # then construct dictionary with name pair values
    rate_constant_list = list_from_key(Q_key)
    # building the rate constant dict
    constant_dict = {}
    constant_dict['k_on'],constant_dict['k_off'],constant_dict['alpha'],constant_dict['beta'],constant_dict['gamma'],constant_dict['gamma_0'] = (1.3*10**4)*glu_conc,3,1,6,0.016,0.0044 
    constant_dict['delta_0'],constant_dict['delta_1'],constant_dict['delta_2'],constant_dict['delta_3'],constant_dict['k_off_2'],constant_dict['k_off_3'],constant_dict['gamma_2'] = 0.00048,1.2,1.3,0.25, 0.063,0.63,3.9
    # constant_dict['NBQX_on'],
    constant_dict['NBQX_off'] = NBQX_off
    # construct Q matrix, enforcing microscopic reversibility where appropriate
    NBQX_Q = Q_from_key(Q_key, rate_constant_list,constants = constant_dict)
    NBQX_Q = np.where(np.isnan(NBQX_Q)==True,0,NBQX_Q)
    return(NBQX_Q)

def NQBX_rec_jumps(N,sampling = 20,Tau_off=200,from_ss = False):
    # getting concentrations at each interval
    NBQX_concs,t = concentration_as_steps(50,0,1/sampling,1000,2000)
    glu_concs,_ = concentration_as_steps(0,10,1/sampling,1000,2000)
    # getting initial state occupancy as number of receptors
    # if from_ss is True
    if from_ss:
        initial_Q = NBQX_Q(50,0)
        initial_occ = (p_inf(initial_Q))*N
        initial_states = np.zeros(21)
        # where occupancy is real, number of receptors in that state >1
        initial_states[np.where(initial_occ>1)] = initial_occ[np.where(initial_occ>1)]
    else:
        initial_states = np.zeros(21)
        initial_states[20] = N
    occs = np.zeros([21,np.size(t)])
    currs = np.zeros(np.size(t))
    for timepoint, time in enumerate(t):
        if timepoint % 1000 == 0:
            print("Progress at",(timepoint/np.size(t))*100,"%")
        Q_NBQX = NBQX_Q(NBQX_conc = NBQX_concs[timepoint],glu_conc = glu_concs[timepoint],Tau_off=Tau_off)
        if time == 0:
            initial_occs = initial_states
        else:
            initial_occs = occs[:,timepoint-1]
        ### getting occupancies for that time point
        _,occup,currens = probabilistic_walk(N,2/sampling,sampling = sampling,initial_states = initial_occs, Q = Q_NBQX)
        occs[:,timepoint] = occup[:,-1] # occupancies at end of time point
        currs[timepoint] = currens[-1]
    p_t = occs/N
    plt.style.use('ggplot')
    NBQX_fig,NBQX_axs = plt.subplots(2,1)
    NBQX_axs[0].set_title("Recovery from NBQX at {}kHz".format(sampling))
    NBQX_axs[0].plot(t,currs)
    NBQX_axs[0].set_xlabel("t(ms)")
    NBQX_axs[0].set_ylabel("I(pA)")
    NBQX_axs[1].set_xlabel("t(ms)")
    NBQX_axs[1].set_ylabel("Probability of Occupancy")
    for item in np.arange(21):
        NBQX_axs[1].plot(t,p_t[item,:],label = item)
    NBQX_axs[1].legend(fontsize=5,loc= 6,bbox_to_anchor=(1.0,0.5))
    NBQX_fig.tight_layout()
    return(p_t,occs,currs)
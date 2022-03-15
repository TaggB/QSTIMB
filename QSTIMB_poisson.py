#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:26:44 2022

@author: benjamintagg
"""
### config
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
from scipy import special
from cycler import cycler
import matplotlib.colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colorlist = ['black','dimgrey','teal','darkturquoise', 'midnightblue','lightskyblue','steelblue','royalblue','lightsteelblue','darkorange', 'orange','darkgoldenrod','goldenrod','gold','khaki','yellow']
mycycle = cycler(color=[colors[item] for item in colorlist])
#    plt.style.use("ggplot")
#    plt.rc('axes',prop_cycle=mycycle)
print("\n \n Currently using ggplot as default plotting environment")
print("Currently using EPyPhys Color Cycle: black-grey-blues-oranges-yellows")


# =============================================================================
# Poisson-based Tau leaping
# =============================================================================


def pn_Tau_leap_Gillespie(N,Q,t_final,interval = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True):
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

def pn_agonist_application_tau_leap_Gillespie(N,Q,t_final,agonist_time,agonist_duration,first_conc,second_conc,interval = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True,rise_time = 250*10**-6,decay_time = 300*10**-6):
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
                    compar_accumulator[stateto] = np.random.poisson((Qs[statefrom,stateto,intervalcount])*N*interval) # lambda = R*t*N

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
            _,_,remaining_occs,_ = pn_Tau_leap_Gillespie(N = n, Q = S,t_final = t_final-t[-1]-interval,interval = interval,voltage=voltage,Vrev = Vrev,iterations = 1,plot=False)
            iteration_occupancy[:,:,iteration] = np.hstack((occupancy[:,:-1],remaining_occs))[:,:np.size(iteration_occupancy,1)] # catch enacted for overspill
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
def pn_Weighted_adaptive_Tau_leap(N,Q,t_final,sampling = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True):
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
    
def pn_Weighted_adaptive_agonist_application_Tau_leap(N,Q,t_final,agonist_time,agonist_duration,first_conc,second_conc,sampling = 5e-05,voltage=0,Vrev = 0,iterations = 100,plot=True,rise_time = 250*10**-6,decay_time = 300*10**-6):
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
                    _,_,occs,_  = pn_Weighted_adaptive_Tau_leap(N = N,Q=newQ,t_final=agonist_time,sampling=sampling,voltage=voltage,Vrev=Vrev,iterations=1,plot=False)
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
    
### built-in
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

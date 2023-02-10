# QSTIMB
Q-matrix and Stochastic simulation-based Ion Channel Model Builder
A Quick start guide is detailed below. An explanation of what is going on under the hood is detailed at the end.
QSTIMB primarily serves to simulate ligand-gated ion channel-coupled receptors. This can be achieved via two main routes:
-1. A chemical master equation (CME) type approach, using the Q matrix method (Colquhoun and Hawkes) - via either relaxations in constant agonist cocnentrations, or during fast alternation between any two agonist concentrations (such as in piezo-driven fast application).
-2. A stochastic approach, which implements the Gillespie algorithm to allow either relaxations or fast alternation between any two agonist concentrations (such as in piezo-driven fast application). This approach is necessary because real currents show variance in their temporal characteristics even when the stimulus is identical.
</br>
The additional purposes of the software are to serve functions for simulation routines (e.g. adding Gaussian noise, plotting models), and to be simple to use. QSTIMB is written using a functional programming framework for those unfamilair with Python. **For help with a function, type 'help(functionname)' to view its documentation.**. To this end, various example models are included, illustrting how to construct them. The adopted format for the model object is deliberately intuitive.
</br>  
**For a guide on how to use, scroll down.**
</br>  
## **Model format**
Models are initialised as dictionaries containing all of the necessary information to perform simulation stored as keys. This includes: a transition matrix, a Q matrix (which necessarily adopts the standard convention for diagonals), key:value pairs of initial states, key: value pairs of conducitng states and their conductances, and key:value pairs of transition rates which are concentration-dependent, and the concentration for the rates in that Q matrix.**It is recommended that functions are used to construct the dictionaries**.
>/br>
Several example model dictionaries are included to give an illustration of how the object should be created. As a simple case, let's consider the builtin ThreesQ model, which is Smod34 from Harveit and Veruki (2006). In this scheme, arbitrary ligand-gated ion channel-coupled receptors can exist in resting, agonist bound, or open states:

- [0] Unbound, closed state
- [1] Bound, closed state
- [2] open state
such that the model scheme is:
   [0]--[1]--[2]

'''Python
    
    def threeS(agonist_conc =5*(10**-3)):
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
'''
So first, run the line
'''python
Q = threesQ(agonist_conc =5*(10**-3)
'''
tr is a k x k matrix, where k is the number of states in the model. 
    tr[i,j] contains the rate for a transition from state i to j per second and is stored as Q['rates']. This is used for stochastic simulations. Some of these rates        are concentration-dependent. As in Q, these rates are expressed as dimensionless constants (per second) - except where cocentration-dependent rates, which are         expressed as per Mole per second
    The conc-dep key (accessed via Q['conc-dep']) tells us which rates are concentration-dependent (key = state i, value = state j). 
    
Some states - the open states - are associated with a conductance. These are detailed in the keys of Q['conducting states'], where values are their associated conductance in S. Here both conducting states have the same conductance (50 pS). 

If we access Q['conc'], we see that it is 5mM (or 5x10**-3M). This value is used for relaxations, and is used to scale to rates as appropriate during agonist applications.

Q['initial states'] contains a list of initial states - i.e. states occupied at the simulation start time. The keys are state indexes (state i), and the values are occupancy probability. It is more efficient if this contains states only with a probability >0, but it can in theory also contain states with probabiltiy 0 (i.e. it could be a list of all states). 

Finally, Q['Q'] is the same as Q['rates'], but follows convention for diagnoal entries, which is necessary for the Q matrix (CME) approaches.

## **Constructing custom models**
To create a custom model, simply follow the convention above: create a function of the type above that returns a similar dictionary object, with k x k dimension objects for the number of desired states, k.

This of course creates an arbitrary model that may or may not be biophysically realistic. For more complex options, such as enacting miscroscopic reversibility, see 'make_Q_reversible'.

This model, once defined, can then be passed to the function that performs simulations.

## **Performing Simulations**
First, define a model object. Here, I shall use the threesQ model from above, with default arguments (5 mM agonist). Note that the model dictionary cna be updated, should one wish to change the rates between simulations. A deep copy is recommended (using the .copy(deep=True) method).

'''Python

    testmodel = threesQ()
'''

A simple simulation can then be performed with the method of choice. Since the CME method is deterministic, it will always return the same simualted current and occupancies. Runtime is dependent on mdoel compelxity, but should not take more than a few seconds with <25 states. If a relaxation  to steady-state is desired, this is relatively simple. Here, we will use 50 receptors (N), at a sampling frequency of 20 kHz, for a total time of 10 ms.

'''Python

    testmodel = threesQ()
    model_outputs = Q_relax(Q=testmodel,N=50,t_final = 10*(10**-3),voltage= -60,interval= 5e-05,Vrev= 0,plot =True, just_pt=False)
'''
The simulation should be displayed and all relevant information stored in model outputs. The nature of this output, as well as means for saving and loading the data is detailed below.

We can also simulate realistic agonist applications, where the concentration of agonist is changed. This approach uses realistic concentration jumps (see raw code for details: Credit to Andrew Plested for method). We will use the same number of receptors, sampling frequency, and model. This time, the receptors are initially in no agonist (first_conc = 0). At 5 ms (agonist_time), 5 mM agonist is applied (second_conc) for 1 ms (agonist_duration). The receptors are simulated for a further 4 ms (t_final). This means that a relaxation is applied once the agonist is completely removed. The rise_time and decay_time arguments specify how fast the exchange occurs between first_conc and second_conc. By default, these are both  250 us.

'''Python

    testmodel = threesQ()
    model_outputs = Q_agonist_application(Q=testmodel,N=50,first_conc=0,second_conc=5*10**-3,agonist_time=5*(10**-3),agonist_duration = 1*(10**-3),t_final = 10*10**-3,interval = 5e-05,voltage =-60,Vrev = 0,rise_time=250*10**-6,decay_time=250*10**-6,plot = True):
'''











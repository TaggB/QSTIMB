# QSTIMB
Q-matrix and Stochastic simulation-based Ion Channel Model Builder
QSTIMB primarily serves to simulate ligand-gated ion channel-coupled receptors. This can be achieved via two main routes:
-1. A chemical master equation (CME) type approach, using the Q matrix method (Colquhoun and Hawkes) - via either relaxations in constant agonist cocnentrations, or during fast alternation between any two agonist concentrations (such as in piezo-driven fast application).
-2. A stochastic approach, which implements the Gillespie algorithm to allow either relaxations or fast alternation between any two agonist concentrations (such as in piezo-driven fast application). This approach is necessary because real currents show variance in their temporal characteristics even when the stimulus is identical.
</br>
The additional purposes of the software are to serve functions for simulation routines (e.g. adding Gaussian noise, plotting models), and to be simple to use. QSTIMB is written using a functional programming framework for those unfamilair with Python. **For help with a function, type 'help(functionname)' to view its documentation.**. To this end, various example models are included, illustrting how to construct them. The adopted format for the model object is deliberately intuitive.
</br>  
**For a guide on how to use, scroll down.**
</br>  
## **Model construction**
Models are initialised as dictionaries containing all of the necessary information to perform simulation stored as keys. This includes: a transition matrix, a Q matrix (which necessarily adopts the standard convention for diagonals), key:value pairs of initial states, key: value pairs of conducitng states and their conductances, and key:value pairs of transition rates which are concentration-dependent, and the concentration for the rates in that Q matrix.**It is recommended that functions are used to construct the dictionaries**.
>/br>
Several example model dictionaries are included to give an illustration of how the object should be created. As an example, I used the Legendre et al.,(1998) model for Glycine receptors in Zebrafish hindbrain, as cited by Harveit and Veruki, 2006.
'''python
def GlyLeg98Q(gly_conc = 5*10**-3):
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
'''
So first, run the line
'''python
Q = GlyLeg98Q(gly_conc = 5*10**-3)
"""
tr is a k x k matrix, where k is the number of states in the model. tr[i,j] contains the rate for a transition from state i to j per second and is stored as Q['rates'].This is used for stochastic simulations. As we see, some of these rates are concentration-dependent. As such, the conc-dep key (accessed via Q['conc-dep']) tells us which rates are concentration-dependent (key = state i, value = state j). Some states - the open states - are associated with a conductance. These are detailed in the keys of Q['conducting states'], where values are their associated conductance in pS. Here both conducting states have the same conductance (50pS). If we access Q['conc'], we see that it is 5mM (or 5x10**-3M). This value is used for relaxations, and is used to scale to rates as appropriate during agonist applications. Q['initial states'] contains a list of initial states - i.e. states occupied at the simulation start time. The keys are state indexes (state i), and the values are occupancy probability. It is more efficient if this contains states only with a probability >0, but it can in theory also contain states with probabiltiy 0 (i.e. it could be a list of all states). Finally, Q['Q'] is the same as Q['rates'], but follows convention for diagnoal entries, which is necessary for the Q matrix approaches.

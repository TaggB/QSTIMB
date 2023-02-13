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
</br>  
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
</br>  
    tr[i,j] contains the rate for a transition from state i to j per second and is stored as Q['rates']. This is used for stochastic simulations. Some of these rates        are concentration-dependent. As in Q, these rates are expressed as dimensionless constants (per second) - except where concentration-dependent rates, which are         expressed as per Mole per second
</br>  
    The conc-dep key (accessed via Q['conc-dep']) tells us which rates are concentration-dependent (key = state i, value = state j). 
</br>  
Some states - the open states - are associated with a conductance. These are detailed in the keys of Q['conducting states'], where values are their associated conductance in S. Here both conducting states have the same conductance (50 pS). 
</br>  
If we access Q['conc'], we see that it is 5mM (or 5x10**-3M). This value is used for relaxations, and is used to scale to rates as appropriate during agonist applications.
</br>  
Q['initial states'] contains a list of initial states - i.e. states occupied at the simulation start time. The keys are state indexes (state i), and the values are occupancy probability. It is more efficient if this contains states only with a probability >0, but it can in theory also contain states with probabiltiy 0 (i.e. it could be a list of all states). 
</br>  
Finally, Q['Q'] is the same as Q['rates'], but follows convention for diagnoal entries, which is necessary for the Q matrix (CME) approaches.
</br>  
## **Constructing custom models**
To create a custom model, simply follow the convention above: create a function of the type above that returns a similar dictionary object, with k x k dimension objects for the number of desired states, k.
</br>  
This of course creates an arbitrary model that may or may not be biophysically realistic. For more complex options, such as enacting miscroscopic reversibility, see 'make_Q_reversible'.
</br>  
This model, once defined, can then be passed to the function that performs simulations.
</br>  
## **Performing Simulations**
In this section, I will showcase the underlying functions that each simulate a single stimulus epoch. Obviously, we often require more than one, which will be covered in the next section.
</br>  
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
    model_outputs = Q_agonist_application(Q=testmodel,N=50,first_conc=0,second_conc=5*10**-3,agonist_time=5*(10**-3),agonist_duration = 1*(10**-3),t_final = 10*10**-3,interval = 5e-05,voltage =-60,Vrev = 0,rise_time=250*10**-6,decay_time=250*10**-6,plot = True)
'''
</br>  
We might want to perform stochastic simulations, which recapitulate the stochastic variation of ion channel currents. Each repeated stimulus epoch gives a different response current because which transition occurs and when it occurs generates variation in the amplitude and time course of the response.
</br>  
Performing stochastic simulations is achieved through a similar means to CME methods, but relies on different functions. There are several means to achieve this, but the validated methods are Tau_leap_Gillespie and agonist_application_tau_leap_Gillespie, which are the stochastic versions of relaxations and agonist applicatiosn respectively. This approach is quite fast per epoch.
</br>  
We can perform a stochastic relaxation with the sampe parameters as above using:
''Python

    testmodel = threesQ()
    model_outputs = Tau_leap_Gillespie(N = 50,Q=testmodel,t_final=10*10**-3,interval = 5e-05,voltage=0,Vrev = 0,iterations = 1,plot=True):
'''
The crucial argument here is iterations. This should be left as 1 ideally. If the number is increased, trajectory averaging is performed, which is not necessary, or appropriate for simulation of electrical currents.
</br>  
Similarly, we can perform a stochastic realisitic agonist application. Iterations should = 1.
''Python

    testmodel = threesQ()
    model_outputs = agonist_application_tau_leap_Gillespie(N=50,Q=test_model,t_final=10*10**-3,agonist_time=5*10**-3,agonist_duration=1*10**-3,first_conc=0,second_conc=5*10**-3,interval = 5e-05,voltage=0,Vrev = 0,iterations = 1,plot=True,rise_time = 250*10**-6,decay_time = 300*10**-6)
'''
## **Using the simulate method**
Obviously, we do not want to have to repeatedly enter the function for individual epoch simulations. This would be quite inefficient, since we would have to declare variables amny times and perform the underlying setup calculations each time. For this, the simulate function is useful. It also includes a timer and progress bar. 
</br>  
Imagine, instead of simulating a single epoch, we want to simulate 100. This would allow us to understand the average properties of the receptor population as well as the source of their underlying variation. For this, we use the simulate wrapper.
''Python

    simulation_outputs = simulate(func,n_sweeps,noise_sd = 4,show_progress=True,graph=True,**kwargs):
'''
func specifies which method to use. There isn't much point defining this to be the CME method (see above), but for stochastic methods, this can be either of the two functions described above (or the alternative weighted adaptive/poisson-based methods). 
</br>  
n_sweeps is the number of repeated, but independent simulations to perform. So for 100, this would be 100.
</br>  
noise_sd is also an argument worth mentioning. This allows digital noise to be layered onto the simulation at the level of the current (noting that the stochastic variation arises as the level of occupancy). Here this is set to 4 (pA), and refers to the standard deviation of noise to apply.
</br>  
show_progress gives a progress bar and time per run when = True.
</br>  
graph shows the end result of the simulation. For many simualtions, one might like to set this to False).
</br>  
kwargs is the crucial element of this wrapper function. For those unfamiliar with python, this refers to keyword arguments. Simply, these are the arguments you would usually provide to the function that simualtes individual epochs.
</br>  
</br>  
So let us see how this works. This time, we shall change some of the arguments:
''Python

    testmodel = threesQ()
    model_outputs = qs.simulate(qs.agonist_application_tau_leap_Gillespie,n_sweeps=100,noise_sd=0,N=500,Q=testmodel,t_final=0.15,agonist_time=0.02,agonist_duration=100*10**-3,first_conc = 0,second_conc = 5*10**-3,iterations=1,voltage=-60,interval=1e-05)
'''
Above, we perform 100 independent stochastic agonsit applications for 500 receptors, without adding any digital noise. The agonist (5 mM) is applied at t = 20 ms for 100 ms. The total time simulated is 150 ms (i.e. 30 ms after agonist removal).
## **File handling**
As well as repeating an individual epoch many times, it is good practice to repeat the simulation a number of times. We can think of the individual simulation of 100 epochs as akin to a current obtained from the recording of a single cell or excised patch. To build a data set, we should repeat this several times. Simulation objects are often large though (because of all of the information stored). Keeping them in RAM is inefficient. We might also want to save the results and access them later. For this, we cna use some basic python syntax.
''Python

    import QSTIMB as qs
    testmodel = qs.threesQ()
    for item in np.arange(0,3):
        hold = qs.simulate(qs.agonist_application_tau_leap_Gillespie,n_sweeps=100,noise_sd=0,N=500,Q=testmodel,t_final=0.15,agonist_time=0.02,agonist_duration=100*10**-3,first_conc = 0,second_conc = 5*10**-3,iterations=1,voltage=-60,interval=1e-05)
        qs.save_sim(hold,'a_filepath_to_desired_directory','name_of_simulation_{}.format(item))
'''

Firstly, note that we import the module first and give it an alias qs. So when we access the functions now, we refer to them as subfunctions of this module. i.e. qs.function()

In the above code, we use a loop to repeat the process of the simulation. For the unitiated, a loop is just a means to execute the same code several times. We repeat the entire simulation threee times. Each simulation is temporarily stored as a variable called 'hold'
</br>  
We then save the hold variable to somewhere on your computer. A neat trick for those unfamiliar with the structure of their file system is to create a file where you want to save the simulation to, and drag this into the python terminal. You should get a string of the format 'mycomputer/User/some_directory/some_file'. The length of that string will change dependent on the complexity of your file directory. Copy and paste that string in place of 'a_filepath_to_desired_directory'. Decide on a anme for the simulation, which should also be a string. Leave the {}, which is useful since it provides the value of 'item' from the loop, such that the frist run will be saved as 'name_of_simulation_0', and subsequent runs will have different (unique names). This prevents overwriting existing files with the same name upon subsequent iterations.
</br>  
</br>  
Great, so we have saved three iterations of a simulation. Now we can load them.
''Python

      simulation_dictionary= qs.load_sim(path_to_simulation)
'''
Where path_to_simulation is simply the concatenation of the direcotry and filename above (again, we cna drag this in by dragging the desired simulation file into the terminal and copy/pasting.
## **Simulation file format**
Now we have a simulation - perhaps a single, or multiple epochs - either run alone, or using the simulate function, we can access the object. For the pruposes of this, We wil assume that we are loading a simulation as above.
</br>
''Python

      simulation_dictionary= qs.load_sim(path_to_simulation)
'''
The structure of the simulation dictionary is fairly intuitive, and can be appreciated by reading the keys of the dictionary
''Python

      simulation_dictionary.keys()
'''
which returns:
''Python

      dict_keys(['t', 'p_t', 'occ_t', 'I_t', 'conditions'])
'''
t is simply the time intervals (samples) - i.e. the timepoints at which the simulation attributes correspond to.
</br>
p_t is the occupancy probability at a given value of t.
   I.e. values are between 0 and 1, and at a given time point, sum to 1.0 across the values for all states.
   It can be accessed as simulation_dictionary['p_t']
   It has dimensions k X t X r, where k is the number of model states, t is the number of time points (equal in length to 't'), and r is the number of epochs in the simulation ( = n_sweeps, in the case above = 100)
   p_t is created by dividing occ_t/N
</br>
occ_t is the real occupancy of each state. 
   It can be accessed as simulation_dictionary['occ_t']
   Its values are between 0 and N at each time point.
   occ_t also has k X t X r shape.
</br>
I_t is the current
   It is produced in the simualtion by multiplying the conductances of all open states by their associated conductance, and then taking the sum at each tiem point.
   It can be accessed as simulation_dictionary['I_t']
   It has shape t x r
</br>
conditions is a dictionary of the model object used for the simulation, By now, it should have familiar structure.
   It can be accessed as simulation_dictionary['conditions']
   nested keys can be viewed by    It can be accessed as simulation_dictionary['conditions'].keys().
</br>
</br>
The above information should be sufficient to perform plotting, averaging and further analysis.

We can also easily convert the current to a standard pandas DataFrame format, to allow subsequent analysis using EPyPhys or other custom routines. 
''Python

      simulated_current = qs.current_to_DataFrame(simulation_dictionary)
'''
In this object, which I have decided to call 'simulated_current', each sweep is a column of the DataFrame, associated with a timestamped index. This allows us to use pandas built-in methods. Other packages exist should the user wish to convert to other formats (e.g. ABF or pclamp files).
</br>
In EPyPhys, we can do all sorts of analysis on this simulated current - e.g. noise analysis (of many varieties), fitting time constants etc.
For now, I shall display an example case using code only from the QSTIMB module.

</br>

## **Example usage case**
Currents arise from the same population of receptors with stochastic variation. We might wish to understand how particular kinetics within a mechanism change a receptor's responsiveness or kinetics.
</br>
Commonly, this is performed using noise analysis (or fluctuation analysis). Here, we will use a realistic agonist application and actually apply non-stationary current fluctuation analysis - see Sigg (1994) or Harveit and Veruki (2006,2007) for explanation.
</br>
</br>
Then, I decided to see how changing the number of epochs used changed the estimates of N, weighted signle channel conductance, and peak open probability. To do this, I used Monte Carlo simulation (sampling repeititively without replacement). Fortunately, this code is built into QSTIMB, and could be modified as desired.

''Python
   
      triplicate_analysis(path1,path2,path3)
      
'''
This function is a wrapper for several other functions.
</br>
It takes three paths for three independent, but identical simulations. Each simulation had 1000 epochs.
</br>
Under the hood, it is calling two functions:
</br>
The first function (EpyPhys.NSFA_optimal) repeatedly performs monte carlo noise analysis.
   First, it samples a number of epochs from the total of 1000 without replacement.
   Noise analysis is performed on these sweeps
      This is repeated 16 times for each number of sweeps (batch size = 16)
         E.g. for a sample size of 20, noise analysis is performed 16 times on independently, randomly sampled sweeps.
      An average is taken for the key parameters N, i, and Po; as well as for skew of the unbinned current amplitude and variance.
</br>
Eventually, this means that for each samples size (e.g. 10, 20, 50, 100, 200, 500, 1000 sweeps/epochs), we generate a value for each of the parameters that should be  indpendent of the identity of sampled sweeps/epochs. This allows us to see how the value of estimates change as the number of sweeps evolves. As such, we can exmaine the certainty of estimates - over what range do they vary?
</br>
The second function retrieves ground truth values, and also calculates the coefficient of determination from the 'most certain' fit.
   
## **Potential usage case**
If we wanted to fit a model, usually this is performed by fitting raw currents with (deterministic simulations). This requires some knowledge of conducting state open times etc. Choosing which model is the most appropriate is often quite arbitrary. How non-conducitng 'hidden states' are arranged is quite arbitrary since they may only subtly affect the macroscopic current.
</br>
One way we can constrain this is by examining whether a particular mechanism (with its reatew constants) accounts for the variance we observe. This allows us to consider not only whether a mechanism accounts for one current, but all potential currents we may observe in a recording.
</br>
So fitting any mechanism can be broken down into the following problems:
1. Generating a mechanism that is microscopically reversible - some rates protected, or fixed, such that they do not vary
2. Comparing its (simulated) current waveform and variance to real data
3. Optimising the mechanism until it describes all cases
</br>
There are a few nuances to this, but some starter code is built into QSTIMB.
The first option I attempted is coded as firefly_fit.
</br>
This function takes a real current, a putative mechanism, and optimises transition rates to minimise the error between simulated and real data. The optimisation is performed using a firefly algorithm (with superposition). The firefly algorithm is seemingly an attractive emans for performing fitting, since mathematically, we should move closer to the best fit in an N-dimensional space. The reader can examine the literature elsewhere for why this is the case mathematiclaly, but analagously:
1. We have a point source: this is the N-dimensional space occupied by the real data (at a given timepoint). This point source emits light of a vlue that decreases with distance.
2. We have a number of (randomly placed) fireflies that luminesce according to their proximity to the point souce. This luminescence also decreases over distance.
3. Fireflies moved to the brightest light source they can see (i.e. the value of some transition rates change)
</br>
Superposition allows the best fly to move in an arbitrary direction to see whether it gets closer or further form the point source (i.e. which direction it must move to decrease error of the fit to real data).
</br>
Several problems arise for models with a large number of rate constants:
1. Most of the flies are only as good as the best fly - this is a core criticism of firefly algorithms, that can make them no better than swarm algorithms.
2. Which rate to vary in a high dimensional space?
</br>
An alternative approach is coded in the function fittigration.
This takes the same inputs as the firefly approach, but instead uses the current integral for optimisation with an evolutionary approach. This also did not perform satisfactorily.
</br>
In fact, all open rates were left undetermined in these approaches (because of the purpose of my particular test cases), but if they were known/fitted to some values, either approach may become tractable for model fitting. It is also faster than regenerating Q matrices and performing CME simulations.

















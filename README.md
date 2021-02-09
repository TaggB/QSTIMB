# QSTIMB
Q-matrix and Stochastic simulation based Ion Channel Model Builder
QSTIMB contains an implementation of (and functions comprising) the Colquhoun and Hawkes generator (or Q matrix) approach to modelling ion channel relaxations using discrete-time markov chains, as well as performing realistic concentration jumps, voltage protocols etc **BUT** it also contains a number of means to perform simulations stochastically (discrete adjusted to continuous time Markov chains), rather than the aforementioned probabilistic approach. Why? Because real currents show variance in their profile between sweeps. It also allows creating of custom models and contains some examples for various scenarios. The software is written with a functional programming framework in mind to readily allow use of existing functions for those unfamilair with Python. **For help with a function, type 'help(functionname)' to view its documentation.**
</br>  
**For a guide on how to use, scroll down.**
</br>    
One could conceivably fit a model using a Q-matrix based approach, and add an infinite number of states until it 'fits' the data. This is of course, highly informative, and gives us a sense of the average probability. But a macroscopic current does not alwasy conform to the average - they are just too hipster. As such, the tools within allow an implementation of the Gillespie algoirthm, and bounded methods for using it to understand how diverse waveforms can give rise to the average. For example, what happens when some channels deactivate early? And importantly, how do we understand the channel open probability? It was this last question that mainly motivated creation of this software, since rates are often set to conform to a measured open probability, but for an open probability of 0.7, it is not the case that 70% of channels ALWAYS respond. And this method necessarily confuses channel bursting and single open events. In the Gillespie algorithm, not so, the open probability is not set, and through multiple simulations, should conform to a measured value.
</br>  
## How to use
**Constructing Q or Generator Matrices**  
Several functions are provided to aid in the creation of Q amtrices for non-python users, but none of them are as easy as creating a kxk numpy array where k = number of states.
'Q = np.zeros([k,k])'  
and then indexing rates manually as Q[0:k,0] = rates from state 0 to all other states
</br>  
However, several functions allow a user to be guided in Q matrix creation.  
'Q = logicalQconstructor()' allows the user to construct a Q matrix, setting rates as desired or through microscopic reversibility.  
And an intermediate solution allows construction of a Q matrix from a key:
'Q_from_key()' I recommend reading the function's documentation, or viewing the code for A1-Y2_key() to understand how to use it.
</br> 
We can then parse Q into 'make_Q_reversible' as 
'Q = make_Q_reversible(Q)' to enforce microscopic reversibility by the Colquhoun and Plested method
</br>  
We can view this Q matrix using Q_graph:
'Q_graph(Q)'
</br>  
**Simulating Ion Channels via the Q matrix method**  
A sample Q matrix can be returned via:
'Q = AMPAR_generator()' which contains rates used for GluA1 homomers in the presence of TARP g-2 (Coombs,2017)
</br>
Current relaxations can be obtained from probabilistic_walk, which uses the standard Q matrix method of obtaining spectral matrices and their coefficients (these functions are also implemented should the user want to use them as standalone, detailed below)  
'p_t,occupancies,currents = probabilistic_walk(N,t_final,voltage= -60,agonist_conc= 10,sampling= 20,Vrev= 0,Q=False,conductances=False,graphs =False, initial_states = False,just_pt=False)'  
</br>
As we can see, there are many optional arguments to specify, which are detailed in the documentation. Of note, N sets the number of receptors to simulate, and when graphs = True, and beautoful visualisation is returned.
</br> 
More interestingly along the lines of a classical approach is the 'agonist_jump()' function, which performs realistic concentration jumps, such as those performed during fast application/ piezo-driven application experiments. The method for this approach is taken from Andrew Plested (full credit details in the code file), and works by calculating the concentration at time steps, which resembles an erf type function. It's probably best if you don't play with the stochastic = True argument. It requires a lot of processing power, and may be non-functional at time of writing.  Setting graphs = True will again give you nice visualisations.
</br>
'occupancies,currents = (N,first_conc,second_conc,onset_time,application_duration,record_length,sampling,Q = False,rise_time=400,decay_time=400,stochastic = False,graphs = False,voltage=False,Vrev=False,conductances=False)'.
</br>  
**Tools for custom simulation scenarios. See Q matrix cookbook (Colquhoun and Hawkes) and raw code of, e.g. probabilistic walk or agonist jump, for usage guidelines**  
As mentioned above, all of the functions to perform relaxations or concentration jumps - or relaly any other scenario you can think of are written into this software
</br>  
'pinf = p_inf(Q)' returns the equilibrium occupancies from a Q matrix  
</br>  
'spectals = spectral_matrices(Q)' returns the spectral matrices obtained using Moore-Penrose pseudoinverse. The optional ret_eig argument also returns sorted eignevalues and eigenvectors.  
</br>  
'ampllitude_coeff = amplitude_coeff(spectral_matrices,p0)' returns the amplitude coefficients
</br>
'coefficients_to_components()' solves for p(t), the probability of state occupancy at t by multiplying out expoenntial and amplitude terms.
</br>  
'concentration_as_steps()' and 'occupancies_in_jump()' solve for concentrations and state occupancies during a realistic concentration jump
</br>  
**Example simulation types**  
</br>
By default, 'probabilistic_walk()' and 'agonist_jump()' use the A1_y2 rates from Coombs (2017) to perform an agonist jump. Set graphs = True and try it out.
</br>
Other more complex examples include pore-blocking simualtions under various conditions, currently using rates for PhTx from Bowie, Mayer, and Lange, but confusingly names 'NASPM'. Feel free to try them or, or compare the code to more basic scenarios. Because these are implemented as both realistic voltage and concentration jumps, p(t) must be recalculated frequently, and so they cna run slow on a slow processor. E.g. NASPM_v_jumps runs in approx 10 mins on 2GhZ processor.
</br>  
e.g. 'NASPM_V_ramp()' or 'NASPM_V_jumps()' or 'NASPM_conc_response_jumps()'
</br>  


# Stochastic Simulations
As well as addressing the classical Q matrix approach, I also include tools for stochastic simulation. If this doesn't interest you, then don't worry. 
These use a modified Gillespie algorithm to perform a random walk, which is originally notoriously slow, but incredibly cool! If you want a flavour of how they work, try  
'random_walk(1,1,1,graphs = True)' to see the rapid openings for a single GluA1 channel over the space of 1ms. Do it again. Do you see that the opening patterns are different to before? Like a real, stochastic ion channel? 
</br>  
Now, I said that these simualtions can be slow. If we increase N, the runtime is only affected a modest amount. If we increase time to 100ms:  
'random_walk(1,10,1,graphs = True)', or iterations it only increases slightly. **BUT** this only makes a useful point about stochastic simulation, since this is slightly anti-Markovian. See help(stochastic_advance()) to understand why (also explained a bit below, see stochastic toolkit).  
</br>  
So now, you are a bit angry about my disingenuous proclaimation that we can simply, and stochastically simulate ion channels. Well, we could argue that we actually just did to  amore than acceptable level, but if you want to do a 100% real stochastic simualtion, then be my guest. But have a strong processor and a good level of patience. It took me 10 mins to simulate a single channel for 10ms. By nature it would perform exponentially more slowly with increased N, number of states, and time.
</br  
But there is a better way, that doesn't require you to fry your computer.  
</br>  
Simply set the simple argument to false and produce a real solution to the stochastic simulation of a single channel.  
'random_walk(1,1,1,graphs = True,simple=False)'  
And we can see that this runs quickly, as before. 
</br>  
This is cool. Why? Well, this makes a discrete time adjustedment to the continuous time Gillespie algorithm. In short, it calculates the probability of any multiple transition occurring during a discrete time interval (correct for continuous time) by changing the transition matrix to a network, performing a depth first search of all transitions, and calculating the product probability of each multiple transition from a given state occurring. This is then corrected for a 'sampling rate' or dt to scale the probabilities. This sort of mimics the missed events scenario in real electrophysiology: we may miss events at 20kHz, but they can be corrected for, and we wouldn't know. As such, averaging many of these iterations with the correct set of rate constants should give rise to the same p(t) as the Q matrix method - HUZZAH!
</br>  
Of course, increasing N increases the run time, and so does increasing t_final. But not impossibly. And it's still cool. Feel free to try your own discrete time correction.
</br>  
**Tools for your own stochastic simulation**  
I would recommend understanding how random_walk occurs and how it uses the following functions if you want to experiment with your own:
'stochastic_advance()'
'stochastic_transitions()'
'get_paths()'
'path_rates()'
But the basic process is that we feed previous occupancies of each state to a for loop, which then calculates the **transition** probabilities. A random number is picked for each receptor from an exponential distribution. If the probability of the minimum probability transition exceeds the random number, that transition occurs. If no transition probability is greater than the random number, the receptor soujourns in that state.
</br>  
These transition probabilities are calculated from get paths, which makes all states of the Markov chain communicable within a single time step, dt (or sampling frequency) and calculates the probability of occurrence for each transition from each state to each other occurring via all possible routes (a path). If this sounds like no mean feat, that's because it is not. But it runs rapdily thanks to a good old depth-first search algorithm.
</br>
And that's all for now. Model fitting to records will probably be implemented at some point in the future... but for now, enjoy.
Please submit any issues encountered.

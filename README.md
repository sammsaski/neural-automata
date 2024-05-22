# Neural Automata

We propose a novel model of computation with neural automata. Like classic automata, this model of computation contains a set of states and descriptions of the transitions between these states. Unlike classic automata, every transition of the system is described by a neural network.

# Examples
To run the `simple_math` example,
- Ensure that the virtual environment is running
- Then, from either the root directory or `scripts` directory, execute the following command
```sh 
bash /path/to/run_simple_math.sh
```


# Model
Neutron
- Modes
  - Name
  - Neural Network
- Transitions
  - Nodes
  - Neural Network
 - States

1. Maintain an active state
    - state variables
2. Be able to change modes
    - neural controller
3. Be able to update state
    - state variables
    - transitions

# Automata
The automata is comprised of a set of modes, a set of transitions, and a set of state variables. Additionally, the automata can take inputs and give outputs.

# Mode
A mode is a state variable of the automata that dictates which transitions can be taken. From specific modes, only certain transitions are defined. In the context of the automata, we turn these modes into a set and input them when constructing the automata.

# Transition
A transition is an action taken during one timestep of the automata. In this case, we assume that every timestep takes one transition of the automata. Again, we create a set of transitions and provide them as input when building the automata.

# State
The set of state variables to maintain during the lifetime of the automata.


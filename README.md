# Neurosymbolic Automata

We propose a novel model of computation with neurosymbolic automata. Like classic automata, this model of computation contains a set of states and descriptions of the transitions between these states. Unlike classic automata, every transition of the system is described by a neural network.

# NeuS'25

To reproduce the results in the NeuS'25 paper, see the `examples/regex` and `examples/simple_math_vlm_comp` folders. Each of these folders will contain the data, models, and scripts necessary to running the experiments.

There is a push-button script in the `scripts` directory that will rerun the regex and arithmetic evaluation experiments which are examined in the paper. Note that, as it stands, the current code will append to the existing results files, so the results directories should be deleted and recreated in each experiment directory (e.g. `examples/regex/results` and `examples/simple_math_vlm_comp/results`) before running.
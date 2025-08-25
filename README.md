# Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs)

Repository for the paper [Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs)](https://neus-2025.github.io/files/papers/paper_34.pdf) published in [NeuS'25](https://neus-2025.github.io/index.html).

# NeuS'25

To reproduce the results in the NeuS'25 paper, see the `examples/regex` and `examples/simple_math_vlm_comp` folders. Each of these folders will contain the data, models, and scripts necessary to running the experiments. The results shown in the paper are also represented in the `results` subdirectory of each.

There is a push-button script in the `scripts` directory that will rerun the regex and arithmetic evaluation experiments which are examined in the paper. Note that, as it stands, the current code will append to the existing results files, so the results directories should be deleted and recreated in each experiment directory (e.g. `examples/regex/results` and `examples/simple_math_vlm_comp/results`) before running.

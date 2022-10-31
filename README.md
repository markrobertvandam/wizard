# MSc Thesis by Mark van Dam, MSc Artificial Intelligence at the University of Groningen

## Content
The master branch contains the most up-to-date implementation of the MCTS-based algorithm which was used for the final results in the thesis.
The DQN branch contains the implementation that uses a DQN playing agent.

## How-to
To run the code, you will have to download the requirements, either using the requirements.txt file or by downloading the modules TensorFlow/2.5.0-fosscuda-2020b and matplotlib/3.2.1-foss-
2020a-Python-3.8.2 from the HPC. 

### Training
For training runs, use `python3 main.py`

### Test
For test runs, use `python3 learned_comparison.py`

### Plotting
To plot results, use `python3 plot_accuracy.py`

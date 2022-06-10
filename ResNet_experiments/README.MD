# ResNet Experiments

Run ```generate_slurm_runs.py``` to generate 500 ```.sh``` files that each runs a combination of 5 seeds and 100 different regularization values between ```[1e-1, 1e-8]```.

The log of each experiment will be saved as a ```.npy``` file. The plotting script will then read all the individual logs and plot the results averaged over 5 seeds. Averaged files are already available in this directory, and hence, there is no need to run all the 500 experiments if you want only to plot the results.

Make sure to change ```<user>``` with your username in ```generate_slurm_runs.py```.
You may also change the amount of label noise, the width of the network, and other hyperparameters.

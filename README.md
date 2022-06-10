# Insights for Epoch-Wise Double Descent

This repository contains the official implementation of the following ICML 2022 paper:

> Mohammad Pezeshki, Amartya Mitra, Yoshua Bengio, Guillaume Lajoie
> [Multi-scale Feature Learning Dynamics: Insights for Double Descent](https://mohammadpz.github.io/DD.html)


### Requirements:
To ensure reproducibility, we publish the code and expected results of every experiment.
We claim that all figures presented in the manuscript can be reproduced using the following requirements:
```
Python 3.7.10
PyTorch 1.4.0
torchvision 0.5.0
tqdm
matplotlib 3.4.3
```

---

## Reproducibility:

### ResNet experiments on CIFAR-10
ResNet experiments on CIFAR-10 took 12000 GPU hours on Nvidia V100. The code to manage experiments using the  ```slurm``` resource management tool is provided in the README available in the ```ResNet_experiments``` folder.

### To reproduce each figure of the manuscript


```python fig1.py```:

 The generalization error as the training time proceeds. (left): The case where only the fast-learning feature or slow-learning feature are trained. (right): The case where both features are trained with \kappa=100.
![fig](/expected_results/fig1.png)


```python fig3.py```:

Top: Analytical results of Eqs. 9, 10 compared to gradient descent dynamics.
Bottom: Analytical results of scalar Eqs. 14, 15 compared to ridge regression dynamics.
![fig](/expected_results/fig3.png)

```python fig4.py```:

Left: Phase diagram of the generalization error as a function of R(t) and Q(t). The trajectories describe the evolution of R(t) and Q(t) as training proceeds. Each trajectory corresponds to a different  $\kappa$, the condition number of the modulation matrix where it describes the ratio of the rates at which two sets of features are learned.
Right: The corresponding generalization curves for different plotted over the training time axis.
![fig](/expected_results/fig4.png)

```python fig5_ab.py```:

Heat-map of empirical generalization error (0-1 classification error) for the ResNet-18 trained on CIFAR-10 with $15 % label noise. The X-axis denotes the regularization strength, and Y-axis represents the training time.
![fig](/expected_results/fig5_ab.png)

```python fig5_cd.py```:

The same plot with the analytical results of the teacher-student. We observe a qualitative comparison between the ResNet-18 results and our analytical results.
![fig](/expected_results/fig5_cd.png)


```python fig6.py```:

The effect of regularizing the quantity Q on mitigating the double descent curve.
![fig](/expected_results/fig6.png)


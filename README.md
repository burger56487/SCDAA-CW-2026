# SCDAA Coursework 2025–26 — Code Repository

This repository contains the Python implementation accompanying our submitted report for the **Stochastic Control and Dynamic Asset Allocation (SCDAA)** coursework.

The project implements numerical algorithms for solving stochastic control problems using:

* Linear Quadratic Regulator (LQR) theory
* Monte-Carlo simulation
* Supervised neural network learning
* Deep Galerkin Method (DGM)
* Policy Iteration Algorithm (PIA)

All experiments reproduce the numerical results, tables, and figures presented in the submitted report.

---

## Project Overview

The coursework studies numerical methods for solving stochastic control problems arising in dynamic asset allocation.

The implementation includes:

* Analytical **LQR benchmark solutions**
* **Monte-Carlo validation** of numerical schemes
* **Supervised neural networks** for approximating value functions and optimal controls
* **Deep Galerkin Method (DGM)** for solving the associated Hamilton–Jacobi–Bellman PDE
* **Policy Iteration Algorithms (PIA)** combined with deep learning for control improvement

The repository is organized so that each exercise corresponds to one independent Python script.

---

## Quick Start

### 1. Install Dependencies

Only libraries permitted in the coursework specification are used.

```bash
pip install numpy scipy matplotlib torch
```

Python **3.9 or newer** is recommended.

If a CUDA-enabled GPU is available, PyTorch will automatically utilize it.
However, all experiments can also be executed on a standard CPU.

---

### 2. Run All Experiments

All figures and numerical results in the report can be reproduced by running:

```bash
python ex1_lqr_mc.py
python ex2_supervised.py
python ex3_dgm.py
python ex4_pia.py
```

Generated plots will be saved automatically in the `plots/` directory.

---

## Repository Structure

```
SCDAA-CW-2026/
│
├── SCDAA.pdf                   # Submitted coursework report
│
├── ex1_lqr_mc.py               # Exercise 1: LQR solution & Monte-Carlo validation
├── ex2_supervised.py           # Exercise 2: Supervised learning of value & control
├── ex3_dgm.py                  # Exercise 3: Deep Galerkin Method for PDE
├── ex4_pia.py                  # Exercise 4: Policy Iteration with DGM
│
├── plots/                      # Figures generated during execution
│
└── README.md
```

---

## Experiments

### Exercise 1 — LQR Solution & Monte-Carlo Validation

Solves the Riccati equation for the analytical LQR solution and verifies numerical convergence using Monte-Carlo simulation.

Run:

```bash
python ex1_lqr_mc.py
```

Outputs:

* Error metrics for time discretisation and sample size
* Convergence plots stored in
  `plots/convergence_plot.png`

---

### Exercise 2 — Supervised Learning

Trains neural networks to approximate:

* the **value function**
* the **optimal control policy**

Networks used:

* `NetDGM` (value function approximation)
* `NetFFN` (control policy approximation)

Run:

```bash
python ex2_supervised.py
```

Outputs:

* MSE training logs
* `plots/ex2_supervised_loss.png`

---

### Exercise 3 — Deep Galerkin Method

Solves the linear PDE associated with a fixed control policy using the **Deep Galerkin Method**.

Run:

```bash
python ex3_dgm.py
```

Outputs:

* DGM loss evolution
* `plots/ex3_dgm_results.png`

---

### Exercise 4 — Policy Iteration Algorithm

Implements iterative **policy evaluation and policy improvement** using DGM approximations.

Run:

```bash
python ex4_pia.py
```

Outputs:

* policy convergence logs
* `plots/ex4_policy_iteration.png`

---

## Reproducibility

All numerical results and figures reported in the coursework report were generated using the scripts in this repository.

The marker can reproduce all experiments using:

```bash
pip install numpy scipy matplotlib torch

python ex1_lqr_mc.py
python ex2_supervised.py
python ex3_dgm.py
python ex4_pia.py
```

No manual parameter tuning or intervention is required.

---

## Implementation Details

Key implementation aspects include:

* **Automatic differentiation** in PyTorch to compute gradients and Hessians required by the Deep Galerkin Method (`create_graph=True`).

* **Dynamic domain sampling** at every training epoch to improve global generalization.

* **Decoupled computational graphs (`.detach()`)** in the Policy Iteration Algorithm to ensure stable alternating optimisation.

* **Explicit Euler–Maruyama discretisation** for Monte-Carlo simulation of stochastic differential equations.

* **Log–log convergence plots** verifying theoretical convergence rates.

* **Analytical LQR solutions** used as ground-truth validation benchmarks.

---

## Expected Runtime

Approximate runtime on a standard laptop CPU:

| Task                   | Time         |
| ---------------------- | ------------ |
| Monte Carlo validation | 1–3 minutes  |
| Supervised learning    | 2–4 minutes  |
| DGM training           | 5–8 minutes  |
| Policy iteration       | 8–12 minutes |

---

## Group Members & Contributions

As required by the coursework specification, this project was completed collaboratively by three students. The workload was shared equally and agreed upon by all members.

| Name        | Student Number | Contribution |
| ----------- | -------------- | ------------ |
| Yiwen Wan   | s2769905       | 33.3%        |
| Jixuan He   | s2778923       | 33.3%        |
| Jiajun Zhao | s2882066       | 33.4%        |

---

## Contact

If any issues arise when running the code, please contact any group member listed above.

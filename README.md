# SCDAA Coursework 2025–26 — Code Repository

This repository contains the Python implementation accompanying our submitted report for the **Stochastic Control and Dynamic Asset Allocation (SCDAA)** coursework.

The project implements numerical methods for solving a two-dimensional Linear Quadratic Regulator (LQR) problem using:

- analytical Riccati-ODE benchmark solutions,
- Monte Carlo simulation,
- supervised neural network learning,
- the Deep Galerkin Method (DGM),
- policy iteration with neural-network approximation.

The scripts in this repository reproduce the numerical figures and tabulated results reported in the final coursework report. As required by the coursework brief, the repository includes a `README.md` explaining how to run the code to reproduce the reported outputs. :contentReference[oaicite:1]{index=1}

---

## Project Overview

The coursework studies numerical methods for stochastic control problems arising from dynamic asset allocation.

The implementation includes:

- **Exercise 1:** analytical LQR benchmark via the Riccati ODE and Monte Carlo validation,
- **Exercise 2:** supervised learning of the value function and optimal Markov control,
- **Exercise 3:** Deep Galerkin Method for a linear PDE under a fixed control,
- **Exercise 4:** policy iteration combining DGM-based policy evaluation with neural-network policy improvement.

Each exercise is implemented in a separate Python script.

---

## Requirements

Only the libraries permitted by the coursework brief are used:

- `numpy`
- `scipy`
- `matplotlib`
- `torch`

Python **3.9 or newer** is recommended.

All scripts can be run on CPU. If a CUDA-enabled GPU is available and correctly configured in PyTorch, it may be used automatically.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy scipy matplotlib torch
'''
### 2. Run All Experiments

All figures and numerical results in the report can be reproduced by running:

```bash
python experiments/ex1_lqr_mc.py
python experiments/ex2_supervised.py
python experiments/ex3_dgm.py
python experiments/ex4_policy_iteration.py
```

Generated plots will be saved automatically in the `plots/` directory.

---

## Repository Structure

```
SCDAA-CW-2026/
│
├── experiments/
│   ├── ex1_lqr_mc.py
│   ├── ex2_supervised.py
│   ├── ex3_dgm.py
│   └── ex4_policy_iteration.py
│
├── plots/
│   ├── convergence_plot.png
│   ├── ex2_supervised_loss.png
│   ├── ex3_dgm_results.png
│   └── ex4_policy_iteration.png
│
└── README.md
```

---

## Experiments

### Exercise 1 — LQR Solution & Monte-Carlo Validation

Solves the Riccati equation for the analytical LQR solution and verifies numerical convergence using Monte-Carlo simulation.

Run:

```bash
python experiments/ex1_lqr_mc.py
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
python experiments/ex2_supervised.py
```

Outputs:

* MSE training logs
* `plots/ex2_supervised_loss.png`

---

### Exercise 3 — Deep Galerkin Method

Solves the linear PDE associated with a fixed control policy using the **Deep Galerkin Method**.

Run:

```bash
python experiments/ex3_dgm.py
```

Outputs:

* DGM loss evolution
* `plots/ex3_dgm_results.png`

---

### Exercise 4 — Policy Iteration Algorithm

Implements iterative **policy evaluation and policy improvement** using DGM approximations.

Run:

```bash
python experiments/ex4_policy_iteration.py
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

python experiments/ex1_lqr_mc.py
python experiments/ex2_supervised.py
python experiments/ex3_dgm.py
python experiments/ex4_policy_iteration.py
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

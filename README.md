# SCDAA Coursework 2025–26 — Code Repository

This repository contains the Python implementation accompanying our submitted PDF report for the **Stochastic Control and Dynamic Asset Allocation (SCDAA)** coursework.

The project implements numerical algorithms for solving stochastic control problems using:

* Linear Quadratic Regulator (LQR) theory,
* Monte-Carlo simulation,
* Supervised neural network learning,
* Deep Galerkin Method (DGM),
* Policy Iteration Algorithm (PIA).

All experiments reproduce the numerical results and figures presented in the report.

---

## Group Members & Contributions

As required by the coursework specification, this project was completed in a group of three students.

The workload was shared equally and agreed upon by all members.

| Name     | Student Number | Contribution |
| -------- | -------------- | ------------ |
| Member 1 | XXXXXXXX       | 33.3%        |
| Member 2 | XXXXXXXX       | 33.3%        |
| Member 3 | XXXXXXXX       | 33.4%        |

---

## Dependencies & Environment Setup

Only libraries explicitly permitted in the coursework instructions are used:

* `numpy`
* `scipy`
* `matplotlib`
* `torch`

Install dependencies using:

```bash
pip install numpy scipy matplotlib torch
```

Python ≥ 3.9 is recommended.

---

## Repository Structure

```
SCDAA-CW-2026/
│
├── report/
│   └── SCDAA_Report.pdf
│
├── src/
│   ├── lqr.py                  # Exercise 1.1
│   ├── mc_simulation.py        # Exercise 1.2
│   ├── networks.py             # Neural network definitions
│   ├── supervised_value.py     # Exercise 2.1
│   ├── supervised_control.py   # Exercise 2.2
│   ├── dgm_pde.py              # Exercise 3.1
│   └── policy_iteration.py     # Exercise 4.1
│
├── plots/                      # Automatically generated figures
│
├── main.py                     # Runs all experiments
└── README.md
```

---

## Running the Code

All results shown in the report can be reproduced from this repository.

### Run Entire Coursework (Recommended)

```bash
python main.py
```

This sequentially executes all exercises and automatically generates figures in the `/plots` directory.

---

## Exercise-by-Exercise Execution

### Exercise 1 — LQR Solution & Monte Carlo Validation

Solves the Riccati equation and validates convergence using Monte-Carlo simulation.

```bash
python src/mc_simulation.py
```

Generated outputs:

* Time discretisation convergence plot
* Monte-Carlo sampling convergence plot

---

### Exercise 2 — Supervised Learning

#### Value Function Approximation

```bash
python src/supervised_value.py
```

#### Optimal Control Approximation

```bash
python src/supervised_control.py
```

Outputs:

* Neural network training loss plots

---

### Exercise 3 — Deep Galerkin Method

Solves the linear PDE using the Deep Galerkin Method.

```bash
python src/dgm_pde.py
```

Outputs:

* DGM training loss
* Error comparison with Monte-Carlo benchmark

---

### Exercise 4 — Policy Iteration with DGM

Implements policy evaluation and improvement steps iteratively.

```bash
python src/policy_iteration.py
```

Outputs:

* Policy convergence plots
* Value function comparison with analytical LQR solution

---

## Reproducibility

All numerical results, tables, and figures included in the submitted report were generated using the scripts contained in this repository.

The marker can reproduce every figure by:

```bash
pip install -r requirements.txt
python main.py
```

No manual intervention or parameter tuning is required.

---

## Implementation Notes

* Automatic differentiation in PyTorch is used to compute gradients and Hessians required by the Deep Galerkin Method.
* Monte-Carlo simulations follow Euler–Maruyama discretisation.
* Log–log convergence plots are provided to verify theoretical convergence rates.
* Analytical LQR solutions are used as ground truth validation throughout.

---

## Expected Runtime

Approximate runtime on a standard laptop CPU:

| Task                | Time          |
| ------------------- | ------------- |
| Monte Carlo checks  | 2–5 minutes   |
| Supervised learning | 3–5 minutes   |
| DGM training        | 5–10 minutes  |
| Policy iteration    | 10–15 minutes |

---

## Contact

If any issues arise when running the code, please contact any group member listed above.


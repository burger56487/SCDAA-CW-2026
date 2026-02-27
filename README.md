# SCDAA Coursework 2025-26 - Code Repository

[cite_start]This repository contains the Python codebase to accompany our PDF report for the Stochastic Control and Dynamic Asset Allocation (SCDAA) coursework[cite: 1, 11]. [cite_start]It implements numerical algorithms for solving stochastic control problems using the Deep Galerkin Method (DGM) and Policy Iteration[cite: 4].

## Group Members & Contributions
[cite_start]As requested, our group consists of three members[cite: 16]. [cite_start]The contributions are split equally (33.3% each) as agreed upon by all members[cite: 22].
* **Member 1:** [Your Name], [Your Student Number], Contribution: 33.3%
* **Member 2:** [Name 2], [Student Number 2], Contribution: 33.3%
* **Member 3:** [Name 3], [Student Number 3], Contribution: 33.4%

## Dependencies & Environment Setup
[cite_start]To strictly adhere to the coursework guidelines, this project utilizes **only** the permitted libraries. No other external packages are required. 

Please ensure your Python environment has the following installed:
* `numpy`
* `scipy`
* `matplotlib`
* `torch`

You can install all dependencies via pip:
`pip install numpy scipy matplotlib torch`

## Repository Structure
[cite_start]The codebase is divided into four main execution scripts, corresponding to the four primary exercises outlined in the assignment[cite: 6, 7, 9, 10]:
* `ex1_mc.py`: Implementation of the LQR solver (Riccati ODE) and Monte Carlo validation checks.
* `ex2_supervised.py`: Supervised learning implementation using neural networks to approximate the value function and Markov control.
* `ex3_dgm.py`: Implementation of the Deep Galerkin Method to solve the linearised Bellman PDE.
* `ex4_policy_iteration.py`: The Policy Iteration algorithm combining neural network evaluation and Hamiltonian minimization.

## Instructions to Reproduce Results (PDF Graphics)
All scripts are designed to be run entirely out-of-the-box. Due to headless environment considerations, the scripts do not block execution with pop-up windows. Instead, **they automatically generate and save high-resolution `.png` files** directly to the root directory. 

Run the following commands in your terminal to reproduce the exact graphics and terminal outputs referenced in our submitted PDF:

### 1. Linear Quadratic Regulator & MC Checks (Exercise 1)
[cite_start]To generate the log-log convergence plots for time discretisation and Monte Carlo sampling[cite: 52, 53]:
```bash
python ex1_mc.py
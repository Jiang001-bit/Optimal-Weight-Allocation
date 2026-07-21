# Optimal-Weight-Allocation

This repository provides MATLAB code for the heuristic greedy algorithm used in the paper to tune diffusion time scales in networks.

The goal is to increase the Fiedler value \(\lambda_2\) of a graph Laplacian, thereby reducing the diffusion time scale \(\tau = 1/\lambda_2\), by adding edge weights with low total cost.

Files

- `heuristic_karate.m`  
  Demonstrates the heuristic greedy algorithm in Zachary's Karate Club network.

- `heuristic_vs_cvx.m`  
  Compares the heuristic algorithm with a CVX-based convex optimization baseline on ER, WS, and BA networks.

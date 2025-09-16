# field_ring_optimizer/main.py

import torch
import numpy as np
from pinn_solver import OptimizationSolver

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    # Initial guess for the geometric parameters
    initial_params = {
        's1': 1.5, 's2': 1.5, 's3': 1.5, 's4': 1.5, 's5': 1.5, 's6': 1.5,
        's7': 2.1, 's8': 2.5, 's9': 1.0, 's10': 1.0, 's11': 1.0, 's12': 1.5,
        'w2': 3.0, 'w3': 3.0, 'w4': 3.0, 'w5': 3.0, 'w6': 3.0, 'w7': 3.0,
        'w8': 1.5, 'w9': 1.5, 'w10': 1.5, 'w11': 1.5, 'w12': 1.5, 'w13': 1.5
    }
    
    # Initialize the solver with the parameters
    solver = OptimizationSolver(initial_params=initial_params, results_dir="optimization_results")
    
    # Start the training and optimization process
    solver.train(epochs=200, print_every=100)

if __name__ == "__main__":
    main()
# field_ring_optimizer/pinn_solver.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from collections import defaultdict

from geometry import PolyMesh
from constants import L_REF, V_REF, POISSON_PARAM, TARGET_FIELD_MV_PER_CM, MIN_GEOMETRY_VALUE, V_PER_UM_TO_MV_PER_CM
import plotting

class PINN(nn.Module):
    """
    The Physics-Informed Neural Network model.
    It predicts potential V(x, y) and computes its derivatives (E-field, Laplacian).
    """
    def __init__(self, hidden_layers=[128, 128, 128], activation=nn.Tanh()):
        super().__init__()
        layers = [nn.Linear(2, hidden_layers[0]), activation]
        for i in range(len(hidden_layers) - 1):
            layers.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1]), activation])
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-dimensionalize inputs
        return self.net(x / L_REF)

    def dimensional_potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x) * V_REF

    def compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        u = self.forward(x)
        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        
        laplacian = 0
        for i in range(x.shape[1]):
            grad_i = grad_u[:, i:i+1]
            lap_i = torch.autograd.grad(grad_i.sum(), x, create_graph=True, allow_unused=True)[0]
            if lap_i is not None:
                laplacian += lap_i[:, i:i+1]
        
        # Return dimensional laplacian
        return laplacian / (L_REF**2)

    def electric_field_mag(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        phi = self.dimensional_potential(x)
        # E = -∇φ
        grad_phi = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        E_v_per_um = -grad_phi
        # Return magnitude in MV/cm
        return torch.norm(E_v_per_um, dim=1, keepdim=True) * V_PER_UM_TO_MV_PER_CM

class OptimizationSolver:
    """
    Orchestrates the entire optimization process, including managing the geometry,
    the PINN, the losses, and the training loop.
    """
    def __init__(self, initial_params: dict, results_dir="results"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.poly_mesh = PolyMesh(initial_params).to(self.device)
        self.pinn = PINN().to(self.device)
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.loss_names = [
            'pde', 'boundary', 'field_target', 'geom_constraint', 'min_value', 'field_upper_boundary'
        ]
        self.log_lambdas = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(0.0)) for name in self.loss_names}
        ).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.poly_mesh.parameters(), 'lr': 1e-3},
            {'params': self.pinn.parameters(), 'lr': 1e-3},
            {'params': self.log_lambdas.parameters(), 'lr': 1e-3}
        ], amsgrad=True)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        self.history = defaultdict(list)
        self.initial_losses = None

    def _compute_losses(self):
        """Computes and returns a dictionary of all loss components."""
        vertices = self.poly_mesh.get_vertices()
        
        # 1. PDE Loss
        interior, exterior = self.poly_mesh.sample_points(n_interior=4000, n_exterior=2000)
        lap_interior = self.pinn.compute_laplacian(interior)
        lap_exterior = self.pinn.compute_laplacian(exterior)
        pde_residual_in = lap_interior + POISSON_PARAM
        pde_residual_out = lap_exterior # For vacuum, rho = 0
        loss_pde = torch.mean(pde_residual_in**2) + torch.mean(pde_residual_out**2)

        # 2. Boundary Condition Loss
        lower_b, upper_b = self.poly_mesh.get_boundary_segments(vertices)
        potential_lower = self.pinn(lower_b) # Should be 0
        potential_upper = self.pinn(upper_b) # Should be 1 (non-dimensional)
        loss_boundary = torch.mean(potential_lower**2) + torch.mean((potential_upper - 1.0)**2)

        # 3. Field Target Loss (at specific corners)
        target_indices = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
        valid_indices = [i for i in target_indices if i < len(vertices)]
        target_points = vertices[valid_indices] + torch.tensor([-0.1, 0.1], device=self.device)
        E_mag = self.pinn.electric_field_mag(target_points)
        loss_field_target = torch.mean((E_mag - TARGET_FIELD_MV_PER_CM)**2)

        # 4. Upper Boundary Field Loss (should be zero)
        _, upper_b_verts = self.poly_mesh.get_boundary_segments(vertices)
        upper_boundary_points = self._sample_line_segments(upper_b_verts, num_points=10)
        E_mag_upper = self.pinn.electric_field_mag(upper_boundary_points)
        loss_field_upper = torch.mean(E_mag_upper**2)

        # 5. Geometry Constraint Losses
        p = self.poly_mesh.params
        s_sum = 5.0 + p['s9'] + p['s10'] + p['s11'] + p['s12']
        loss_geom_constraint = (s_sum - 10.0)**2
        loss_min_value = sum(torch.relu(MIN_GEOMETRY_VALUE - param)**2 for param in p.values())
        
        return {
            'pde': loss_pde,
            'boundary': loss_boundary,
            'field_target': loss_field_target,
            'geom_constraint': loss_geom_constraint,
            'min_value': loss_min_value,
            'field_upper_boundary': loss_field_upper,
        }

    def _sample_line_segments(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """Helper to sample points along a chain of line segments."""
        points = []
        for i in range(len(vertices) - 1):
            start, end = vertices[i], vertices[i+1]
            alphas = torch.linspace(0, 1, num_points, device=self.device).view(-1, 1)
            segment_points = start + alphas * (end - start)
            points.append(segment_points)
        return torch.cat(points, dim=0)

    def train(self, epochs=2000, print_every=100):
        print("Computing initial losses for weighting...")
        initial_losses_raw = self._compute_losses()
        self.initial_losses = {k: v.item() + 1e-8 for k, v in initial_losses_raw.items()}
        print("Initial losses:", self.initial_losses)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            current_losses = self._compute_losses()
            
            # Auto-weighted loss calculation
            task_loss = 0
            reg_loss = 0
            for name in self.loss_names:
                normalized_loss = current_losses[name] / self.initial_losses[name]
                weight = torch.exp(-self.log_lambdas[name])
                task_loss += weight * normalized_loss
                reg_loss += self.log_lambdas[name]
            
            total_loss = task_loss + reg_loss
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.poly_mesh.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
            
            self.optimizer.step()
            
            if (epoch + 1) % 500 == 0: self.scheduler.step()
            
            self._log_history(epoch, current_losses, total_loss, task_loss)
            
            if epoch % print_every == 0 or epoch == epochs - 1:
                self._print_status(epoch, epochs)
                plotting.visualize_progress(
                    epoch, self.history, self.poly_mesh.get_vertices(), self.results_dir
                )
        
        print("\n--- Training Complete ---")
        plotting.plot_parameter_history(self.history, self.poly_mesh.params, self.results_dir)
        self.save_history_to_csv()

    # Helper methods for logging and printing
    def _log_history(self, epoch, losses, total_loss, task_loss):
        self.history['epoch'].append(epoch)
        self.history['total_loss'].append(total_loss.item())
        self.history['task_loss'].append(task_loss.item())
        for name, loss_val in losses.items(): self.history[f'{name}_loss'].append(loss_val.item())
        for name, param in self.poly_mesh.params.items(): self.history[name].append(param.item())
        for name, log_lambda in self.log_lambdas.items(): self.history[f'weight_{name}'].append(torch.exp(-log_lambda).item())

    def _print_status(self, epoch, epochs):
        task_loss = self.history['task_loss'][-1]
        print(f"\n--- Epoch {epoch}/{epochs-1} --- LR: {self.scheduler.get_last_lr()[0]:.1e} | Task Loss: {task_loss:.4f}")
        for name in self.loss_names:
            loss = self.history[f'{name}_loss'][-1]
            weight = self.history[f'weight_{name}'][-1]
            print(f"  - {name.replace('_', ' ').capitalize():<22}: {loss:<12.4g} (Weight: {weight:.3f})")

    def save_history_to_csv(self, filename="training_history.csv"):
        history_df = pd.DataFrame(self.history)
        filepath = os.path.join(self.results_dir, filename)
        history_df.to_csv(filepath, index=False)
        print(f"Training history saved to {filepath}")
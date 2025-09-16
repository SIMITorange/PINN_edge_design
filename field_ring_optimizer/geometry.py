# field_ring_optimizer/geometry.py

import torch
import torch.nn as nn
import numpy as np
import deepxde as dde
from scipy.stats.qmc import LatinHypercube
from matplotlib.path import Path

class PolyMesh(nn.Module):
    """
    Manages the parameterized geometry of the field termination rings.
    Generates vertices and samples points using advanced methods.
    """
    def __init__(self, initial_params: dict):
        super().__init__()
        self.params = nn.ParameterDict()
        for key, value in initial_params.items():
            self.params[key] = nn.Parameter(torch.tensor(float(value), dtype=torch.float32))

    @property
    def device(self):
        return next(self.parameters()).device

    def get_vertices(self) -> torch.Tensor:
        """Computes the polygon vertices based on current parameters."""
        # This logic is complex and specific to the problem, so it remains unchanged.
        device = self.device
        p = self.params
        W_jun = torch.tensor(10.0, device=device)
        verts = [torch.tensor([[0., 1.], [W_jun, 1.], [W_jun, 0.]], device=device)]
        x_current = W_jun
        for i in range(1, 13):
            s_i, w_i = p[f's{i}'], p[f'w{i+1}']
            x_current = x_current + s_i
            verts.append(torch.stack([x_current, torch.tensor(0.0, device=device)]).unsqueeze(0))
            verts.append(torch.stack([x_current, torch.tensor(1.0, device=device)]).unsqueeze(0))
            x_current = x_current + w_i
            verts.append(torch.stack([x_current, torch.tensor(1.0, device=device)]).unsqueeze(0))
            verts.append(torch.stack([x_current, torch.tensor(0.0, device=device)]).unsqueeze(0))
        x_end_plus_5 = x_current + 5.0
        five_t, ten_t = torch.tensor(5.0, device=device), torch.tensor(10.0, device=device)
        p54 = torch.stack([x_end_plus_5 - five_t - p['w13'], five_t])
        p55 = torch.stack([p54[0] - p['w12'] - p['s11'], p54[1] + p['s12']])
        p56 = torch.stack([p55[0] - p['w11'] - p['s10'], p55[1] + p['s11']])
        p57 = torch.stack([p56[0] - p['w10'] - p['s9'], p56[1] + p['s10']])
        p58 = torch.stack([p57[0] - p['w9'] - p['s8'], p57[1] + p['s9']])
        end_verts = torch.stack([
            torch.stack([x_end_plus_5, torch.tensor(0.0, device=device)]),
            torch.stack([x_end_plus_5, torch.tensor(1.0, device=device)]),
            p54, p55, p56, p57, p58,
            torch.stack([W_jun, ten_t]),
            torch.tensor([0.0, 10.0], device=device),
            torch.tensor([0.0, 1.0], device=device)
        ])
        verts.append(end_verts)
        return torch.cat(verts, dim=0)

    def get_dde_geometry(self):
        """Returns a deepxde.geometry.Polygon object for the current shape."""
        vertices_np = self.get_vertices().detach().cpu().numpy()
        return dde.geometry.Polygon(vertices_np)

    def get_boundary_segments(self, vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits vertices into lower and upper boundaries."""
        lower_mask = vertices[:, 1] < 0.5
        upper_mask = ~lower_mask
        # Ensure at least one point in each for stability
        if not torch.any(lower_mask): lower_mask[2] = True
        if not torch.any(upper_mask): upper_mask[0] = True
        return vertices[lower_mask], vertices[upper_mask]

    def sample_points(self, n_interior: int, n_exterior: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates high-quality interior and exterior points for PDE loss calculation.
        Uses Latin Hypercube Sampling (LHS) for better space coverage.
        """
        vertices = self.get_vertices().detach()
        geom = self.get_dde_geometry()
        
        # Bounding box for sampling
        x_min, y_min = vertices.min(dim=0).values.cpu().numpy()
        x_max, y_max = vertices.max(dim=0).values.cpu().numpy()
        
        # 1. Sample interior points using DeepXDE's superior random sampling
        interior_points = torch.from_numpy(geom.random_points(n_interior)).float().to(self.device)

        # 2. Sample exterior points using LHS in a larger bounding box
        bbox_margin_x = (x_max - x_min) * 0.1
        bbox_margin_y = (y_max - y_min) * 0.2
        
        sampler = LatinHypercube(d=2)
        sample = sampler.random(n=n_exterior) # Generates points in [0, 1]^2
        
        # Scale samples to the exterior bounding box
        low_bounds = [x_min - bbox_margin_x, y_min - bbox_margin_y]
        up_bounds = [x_max + bbox_margin_x, y_max + bbox_margin_y]
        
        scaled_points = torch.tensor(
            sample * (np.array(up_bounds) - np.array(low_bounds)) + np.array(low_bounds),
            dtype=torch.float32, device=self.device
        )
        
        # Filter out points that accidentally fell inside the polygon
        path = Path(vertices.cpu().numpy())
        is_inside_mask = torch.from_numpy(path.contains_points(scaled_points.cpu().numpy())).to(self.device)
        exterior_points = scaled_points[~is_inside_mask]

        return interior_points, exterior_points
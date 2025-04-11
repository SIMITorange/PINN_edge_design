import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Unit conversion and normalization constants
um_to_cm = 1e-4          # Convert micrometers to centimeters
V_per_um_to_MV_per_cm = 1e-2  # Convert V/μm to MV/cm (1 V/μm = 0.01 MV/cm)

# Dimensionless constants
L_ref = 10.0             # Reference length in μm (e.g., junction width)
V_ref = 2000.0           # Reference voltage in V (max potential difference)
epsilon_r = 9            # Relative permittivity (e.g., GaN)
epsilon_0 = 8.85e-14     # Vacuum permittivity in F/cm
rho_inside = 1e18 * 1.602e-19  # Charge density in C/cm³ (1e18 cm⁻³ * 1.6e-19 C)

class PolyMesh(nn.Module):
    """Class to handle the field termination structure with learnable vertices."""
    def __init__(self, initial_params):
        super().__init__()
        # Initialize learnable parameters for 12 secondary junctions
        self.s1 = nn.Parameter(torch.tensor(initial_params['s1'], dtype=torch.float32))
        self.s2 = nn.Parameter(torch.tensor(initial_params['s2'], dtype=torch.float32))
        self.s3 = nn.Parameter(torch.tensor(initial_params['s3'], dtype=torch.float32))
        self.s4 = nn.Parameter(torch.tensor(initial_params['s4'], dtype=torch.float32))
        self.s5 = nn.Parameter(torch.tensor(initial_params['s5'], dtype=torch.float32))
        self.s6 = nn.Parameter(torch.tensor(initial_params['s6'], dtype=torch.float32))
        self.s7 = nn.Parameter(torch.tensor(initial_params['s7'], dtype=torch.float32))
        self.s8 = nn.Parameter(torch.tensor(initial_params['s8'], dtype=torch.float32))
        self.s9 = nn.Parameter(torch.tensor(initial_params['s9'], dtype=torch.float32))
        self.s10 = nn.Parameter(torch.tensor(initial_params['s10'], dtype=torch.float32))
        self.s11 = nn.Parameter(torch.tensor(initial_params['s11'], dtype=torch.float32))
        self.s12 = nn.Parameter(torch.tensor(initial_params['s12'], dtype=torch.float32))

        self.w2 = nn.Parameter(torch.tensor(initial_params['w2'], dtype=torch.float32))
        self.w3 = nn.Parameter(torch.tensor(initial_params['w3'], dtype=torch.float32))
        self.w4 = nn.Parameter(torch.tensor(initial_params['w4'], dtype=torch.float32))
        self.w5 = nn.Parameter(torch.tensor(initial_params['w5'], dtype=torch.float32))
        self.w6 = nn.Parameter(torch.tensor(initial_params['w6'], dtype=torch.float32))
        self.w7 = nn.Parameter(torch.tensor(initial_params['w7'], dtype=torch.float32))
        self.w8 = nn.Parameter(torch.tensor(initial_params['w8'], dtype=torch.float32))
        self.w9 = nn.Parameter(torch.tensor(initial_params['w9'], dtype=torch.float32))
        self.w10 = nn.Parameter(torch.tensor(initial_params['w10'], dtype=torch.float32))
        self.w11 = nn.Parameter(torch.tensor(initial_params['w11'], dtype=torch.float32))
        self.w12 = nn.Parameter(torch.tensor(initial_params['w12'], dtype=torch.float32))
        self.w13 = nn.Parameter(torch.tensor(initial_params['w13'], dtype=torch.float32))

    def get_vertices(self):
        """Calculate vertex coordinates based on current parameters (units: μm)"""
        W_jun = 10
        vertices = [
            [0., 1.],      # p1
            [W_jun, 1.],   # p2
            [W_jun, 0.],   # p3
        ]

        x_current = W_jun
        y_current = 0

        # Process each of the 12 sub-stems
        for i in range(12):
            s_i = getattr(self, f's{i+1}')
            w_i_plus_1 = getattr(self, f'w{i+2}')  # w2 to w13

            # Vertical segment (s_i)
            x_current += s_i
            vertices.append([x_current.item(), y_current])
            vertices.append([x_current.item(), 1.0])

            # Horizontal segment (w_i_plus_1)
            x_current += w_i_plus_1
            vertices.append([x_current.item(), 1.0])
            vertices.append([x_current.item(), 0.0])

        # Add end structure points
        x_end_plus_5 = x_current + 5
        vertices.extend([
            [x_end_plus_5.item(), 0.],  # p20
            [x_end_plus_5.item(), 1.],  # p21
            [x_end_plus_5 - 5 - self.w13, 5.],         # p54
            [x_end_plus_5 - 5 - self.w13 - self.w12 - self.s11, 5 + self.s12],  # p55
            [x_end_plus_5 - 5 - self.w13 - self.w12 - self.s11 - self.w11 - self.s10, 5 + self.s12 + self.s11],  # p56
            [x_end_plus_5 - 5 - self.w13 - self.w12 - self.s11 - self.w11 - self.s10 - self.w10 - self.s9,
             5 + self.s12 + self.s11 + self.s10],  # p57
            [x_end_plus_5 - 5 - self.w13 - self.w12 - self.s11 - self.w11 - self.s10 - self.w10 - self.s9 - self.w9 - self.s8,
             5 + self.s12 + self.s11 + self.s10 + self.s9],  # p58
            [W_jun, 10.],      # p59
            [0., 10.],         # p60
            [0., 1.]           # p61 (closing point)
        ])
        return torch.tensor(vertices, dtype=torch.float32)

    def generate_mesh_points(self, n_points_per_region=100):
        """Generate points in each region parameterized by s and w."""
        W_jun = 10
        points = []
        x_prev = W_jun

        for i in range(12):
            s_i = getattr(self, f's{i+1}')
            w_i_plus_1 = getattr(self, f'w{i+2}')

            # Vertical region
            x_vert = x_prev + s_i * torch.rand(n_points_per_region)
            y_vert = torch.rand(n_points_per_region)
            points.append(torch.stack([x_vert, y_vert], dim=1))

            # Horizontal region
            x_horz = x_prev + s_i + w_i_plus_1 * torch.rand(n_points_per_region)
            y_horz = torch.rand(n_points_per_region)
            points.append(torch.stack([x_horz, y_horz], dim=1))

            x_prev += s_i + w_i_plus_1

        return torch.cat(points, dim=0)

    def get_boundary_segments(self):
        """Return boundary segments for different voltage conditions."""
        vertices = self.get_vertices()
        return vertices[:3], vertices[-10:-1]

    def is_inside(self, points):
        """Check if points are inside the polygon using matplotlib Path."""
        vertices = self.get_vertices().detach().cpu().numpy()
        path = Path(vertices)
        points_np = points.detach().cpu().numpy()
        inside = path.contains_points(points_np)
        return torch.tensor(inside, dtype=torch.bool, device=points.device)

class PINN(nn.Module):
    """Physics-Informed Neural Network with dimensionless formulation."""
    def __init__(self, hidden_layers=[128, 128, 128], activation=nn.Tanh()):
        super().__init__()
        layers = [nn.Linear(2, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.layers = nn.ModuleList(layers)
        self.activation = activation

        # Reference parameters
        self.L_ref = torch.tensor([L_ref], dtype=torch.float32)
        self.V_ref = torch.tensor([V_ref], dtype=torch.float32)
        self.Vbv_normalized = torch.tensor([1.0], dtype=torch.float32)
        self.Vbv = torch.tensor([V_ref], dtype=torch.float32)

    def forward(self, x):
        x_normalized = x / self.L_ref
        y = x_normalized
        for layer in self.layers[:-1]:
            y = self.activation(layer(y))
        return self.layers[-1](y)

    def dimensional_potential(self, x):
        return self(x) * self.V_ref

    def compute_gradients(self, x):
        x_normalized = x / self.L_ref
        x_normalized.requires_grad_(True)
        u_star = self(x_normalized)
        grad_u_star = torch.autograd.grad(u_star.sum(), x_normalized, create_graph=True)[0]
        grad2_u_star_xx = torch.autograd.grad(grad_u_star[:,0].sum(), x_normalized, create_graph=True)[0][:,0:1]
        grad2_u_star_yy = torch.autograd.grad(grad_u_star[:,1].sum(), x_normalized, create_graph=True)[0][:,1:2]
        return grad_u_star, grad2_u_star_xx + grad2_u_star_yy

    def electric_field(self, x):
        x_cm = x * um_to_cm
        x_cm.requires_grad_(True)
        phi = self.dimensional_potential(x_cm / um_to_cm)
        E = -torch.autograd.grad(phi.sum(), x_cm, create_graph=True)[0]
        return torch.norm(E, dim=1) * 1e-6  # Convert V/cm to MV/cm

class FieldTerminationSolver:
    def __init__(self, initial_params):
        self.poly_mesh = PolyMesh(initial_params)
        self.pinn = PINN()

        # Initialize learnable weight parameters
        self.log_lambda_pde = torch.nn.Parameter(torch.tensor(0.0))
        self.log_lambda_boundary = torch.nn.Parameter(torch.tensor(0.0))
        self.log_lambda_field = torch.nn.Parameter(torch.tensor(0.0))
        self.log_lambda_constraint = torch.nn.Parameter(torch.tensor(0.0))
        self.log_lambda_min_constraint = torch.nn.Parameter(torch.tensor(0.0))

        self.optimizer = optim.Adam([
            {'params': self.poly_mesh.parameters(), 'lr': 0.1},
            {'params': self.pinn.parameters(), 'lr': 0.1},
            {'params': [self.log_lambda_pde, self.log_lambda_boundary,
                        self.log_lambda_field, self.log_lambda_constraint,
                        self.log_lambda_min_constraint], 'lr': 0.01}
        ])

        self.target_field = 4.0
        self.loss_history = []
        self.min_value = 1.0  # Minimum value constraint for s and w parameters: 1um

        # Pre-calculate Poisson parameter
        L_ref_cm = L_ref * um_to_cm
        self.poisson_param_inside = ((L_ref_cm**2) * rho_inside) / (epsilon_r * epsilon_0 * V_ref)

    def compute_loss(self):
        interior_points = self.poly_mesh.generate_mesh_points(n_points_per_region=200)
        vertices = self.poly_mesh.get_vertices()
        x_min, x_max = torch.min(vertices[:,0]).item(), torch.max(vertices[:,0]).item()
        y_min, y_max = 0.0, 10.0

        n_exterior = 2000
        exterior_points = torch.stack([
            torch.rand(n_exterior) * (x_max + 20),
            torch.rand(n_exterior) * (y_max + 5)
        ], dim=1)

        x_pde = torch.cat([interior_points, exterior_points], dim=0).requires_grad_(True)
        inside_mask = self.poly_mesh.is_inside(x_pde)
        poisson_param = torch.where(inside_mask, self.poisson_param_inside, 0.0)

        _, laplacian_star = self.pinn.compute_gradients(x_pde)
        pde_loss = torch.mean((laplacian_star + poisson_param) ** 2)

        lower_boundary, upper_boundary = self.poly_mesh.get_boundary_segments()
        u_lower = self.pinn(lower_boundary)
        u_upper = self.pinn(upper_boundary)
        boundary_loss = torch.mean(u_lower ** 2) + torch.mean((u_upper - self.pinn.Vbv_normalized) ** 2)

        target_indices = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
        target_points = torch.stack([vertices[i] + torch.tensor([-0.1, 0.1]) for i in target_indices])
        E_mag = self.pinn.electric_field(target_points)
        field_loss = torch.mean((E_mag - self.target_field) ** 2)

        s_sum = 5 + self.poly_mesh.s9 + self.poly_mesh.s10 + self.poly_mesh.s11 + self.poly_mesh.s12
        constraint_loss = (s_sum - 10) ** 2

        # Add minimum value constraint for all s and w parameters  s,w最小值设置为1um
        min_constraint_loss = 0.0
        for i in range(1, 13):
            s_param = getattr(self.poly_mesh, f's{i}')
            min_constraint_loss += torch.relu(self.min_value - s_param) ** 2

        for i in range(2, 14):
            w_param = getattr(self.poly_mesh, f'w{i}')
            min_constraint_loss += torch.relu(self.min_value - w_param) ** 2

        total_loss = (
            torch.exp(-self.log_lambda_pde) * pde_loss +
            torch.exp(-self.log_lambda_boundary) * boundary_loss +
            torch.exp(-self.log_lambda_field) * field_loss +
            torch.exp(-self.log_lambda_constraint) * constraint_loss +
            torch.exp(-self.log_lambda_min_constraint) * min_constraint_loss +
            (self.log_lambda_pde + self.log_lambda_boundary +
             self.log_lambda_field + self.log_lambda_constraint +
             self.log_lambda_min_constraint)  # Regularization term
        )

        return {
            'total': total_loss,
            'pde': pde_loss,
            'boundary': boundary_loss,
            'field': field_loss,
            'constraint': constraint_loss,
            'min_constraint': min_constraint_loss,
            'target_points': target_points,
            'e_fields': E_mag
        }

    def train_step(self):
        self.optimizer.zero_grad()
        losses = self.compute_loss()
        losses['total'].backward()
        self.optimizer.step()
        return losses

    def train(self, epochs=1000):
        for epoch in range(epochs):
            losses = self.train_step()
            self.loss_history.append(losses['total'].item())

            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}:")
                print(f"Total Loss: {losses['total'].item():.4f}")
                print(f"PDE Loss: {losses['pde'].item():.4f}, Weight: {torch.exp(-self.log_lambda_pde).item():.4f}")
                print(f"Boundary Loss: {losses['boundary'].item():.4f}, Weight: {torch.exp(-self.log_lambda_boundary).item():.4f}")
                print(f"Field Loss: {losses['field'].item():.4f}, Weight: {torch.exp(-self.log_lambda_field).item():.4f}")
                print(f"Constraint Loss: {losses['constraint'].item():.4f}, Weight: {torch.exp(-self.log_lambda_constraint).item():.4f}")
                print(f"Min Value Constraint Loss: {losses['min_constraint'].item():.4f}, Weight: {torch.exp(-self.log_lambda_min_constraint).item():.4f}")

                # Display constraint status
                s_sum = 5 + self.poly_mesh.s9.item() + self.poly_mesh.s10.item() + self.poly_mesh.s11.item() + self.poly_mesh.s12.item()
                print(f"Current s9+s10+s11+s12+5 = {s_sum:.2f} μm (≤10)")

                # Check if any parameters are below minimum
                below_min = []
                for i in range(1, 13):
                    s_param = getattr(self.poly_mesh, f's{i}').item()
                    if s_param < self.min_value:
                        below_min.append(f"s{i}: {s_param:.4f}")

                for i in range(2, 14):
                    w_param = getattr(self.poly_mesh, f'w{i}').item()
                    if w_param < self.min_value:
                        below_min.append(f"w{i}: {w_param:.4f}")

                if below_min:
                    print("\nParameters below minimum value (1.0):")
                    for param in below_min:
                        print(param)

                # Print electric field values at target points
                target_points = losses['target_points'].detach()
                e_fields = losses['e_fields'].detach()
                print("\nElectric Field at Target Points (MV/cm):")
                for i, (point, field) in enumerate(zip(target_points, e_fields)):
                    print(f"Point {i+1} ({point[0].item():.2f}, {point[1].item():.2f}): {field.item():.4f} MV/cm")

                print("\nOptimized Parameters:")
                # Print s1 - s12
                for i in range(1, 13):
                    param_name = f"s{i}"
                    try:
                        value = getattr(self.poly_mesh, param_name).item()
                        print(f"{param_name}: {value:.4f} μm")
                    except AttributeError:
                        print(f"Attribute {param_name} not found.")

                # Print w2 - w13
                for i in range(2, 14):
                    param_name = f"w{i}"
                    try:
                        value = getattr(self.poly_mesh, param_name).item()
                        print(f"{param_name}: {value:.4f} μm")
                    except AttributeError:
                        print(f"Attribute {param_name} not found.")

                # Print Vbv
                print(f"Vbv: {self.pinn.Vbv.item():.2f} V\n")

                self.visualize()

    def visualize(self):
        vertices = self.poly_mesh.get_vertices().detach().numpy()
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.plot(vertices[:,0], vertices[:,1], 'ko-')
        plt.title("Structure")
        plt.subplot(122)
        plt.semilogy(self.loss_history)
        plt.title("Loss Curve")
        plt.tight_layout()
        plt.show()


# Initial parameters - all set to be >=1
initial_params = {
    's1':1.5, 's2':1.5, 's3':1.5, 's4':1.5, 's5':1.5, 's6':1.5,
    's7':2.1, 's8':2.5, 's9':3.0, 's10':3.5, 's11':4.0, 's12':4.5,
    'w2':3.0, 'w3':3.0, 'w4':3.0, 'w5':3.0, 'w6':3.0, 'w7':3.0,
    'w8':1.5, 'w9':1.5, 'w10':1.5, 'w11':1.5, 'w12':1.5, 'w13':1.5
}

if __name__ == "__main__":
    solver = FieldTerminationSolver(initial_params)
    solver.train(epochs=2000)

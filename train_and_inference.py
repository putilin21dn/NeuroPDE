import matplotlib.pyplot as plt
import torch
from models import DynamicLossWeights, loss_function
import numpy as np


def plot_graphic(model, exact_solution, a, b, c, x_l, x_r, device, name_model):
    """Plot comparison between model predictions and exact solution.
    
    Args:
        model: Neural network model
        exact_solution: Function computing exact solution
        a, b, c: Equation parameters
        x_l, x_r: Left and right boundaries
        device: Computing device
        name_model: Name of the model for plot labels
    """
    # Test grid
    x_test = torch.linspace(x_l, x_r, 100).view(-1, 1).to(device)
    t_test = torch.linspace(0, 5, 100).view(-1, 1).to(device)
    x_mesh, t_mesh = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing="ij")

    x_t_input = torch.cat([x_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)], dim=1)
    x_vals = x_t_input[:, 0].unsqueeze(1).to(device)
    t_vals = x_t_input[:, 1].unsqueeze(1).to(device)

    # Model prediction
    u_pred_ = model(x_vals, t_vals).detach().cpu().numpy().reshape(x_mesh.shape)
    u_true_ = exact_solution(x_mesh.cpu(), t_mesh.cpu(), a, b, c)

    # 3D plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh.cpu(), t_mesh.cpu(), u_pred_, alpha=0.7, label=name_model, cmap='viridis')
    ax.plot_surface(x_mesh.cpu(), t_mesh.cpu(), u_true_.detach().cpu().numpy(), alpha=0.5, label="analytic", cmap='inferno')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.legend()
    ax.set_zlabel('u(x,t)')
    ax.set_title(f"Comparison: {name_model} vs analytical solution")
    plt.show()

    # Time slices plot
    t_values = np.arange(0, 4, 0.5)
    plt.figure(figsize=(10, 6))
    x_test = torch.linspace(x_l, x_r, 100).view(-1, 1).to(device)

    for t_val in t_values:
        t_tensor = torch.full_like(x_test, t_val).to(device)
        u_pred = model(x_test, t_tensor).detach().cpu().numpy()
        u_exact = exact_solution(x_test.cpu(), t_tensor.cpu(), a, b, c)

        plt.plot(x_test.cpu(), u_pred, '--', label=f'{name_model} t={t_val}')
        plt.plot(x_test.cpu(), u_exact.detach().cpu().numpy(), '-', label=f'Analytic t={t_val}')

    plt.title('Comparison by time slices')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return u_true_, u_pred_


def plot_history(loss, name_model):
    """Plot training loss history.
    
    Args:
        loss: List of loss values
        name_model: Name of the model for plot title
    """
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(len(loss)), loss)
    plt.title(name_model)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def train(model, x_interior, t_interior, x_boundary_0, t_boundary, x_boundary_pi, t_boundary_pi, 
          x_initial, t_initial, boundary_funcs, initial_func, exact_solution, equation="heat", 
          a=1, b=0, c=0, epochs=5000, device="cuda", x_l=0, x_r=np.pi, forcing_fn=None, 
          name_model="PINN"):
    """Train the model.
    
    Args:
        model: Neural network model
        x_interior, t_interior: Interior points
        x_boundary_0, t_boundary: Left boundary points
        x_boundary_pi, t_boundary_pi: Right boundary points
        x_initial, t_initial: Initial condition points
        boundary_funcs: Boundary condition functions
        initial_func: Initial condition function
        exact_solution: Function computing exact solution
        equation: Type of equation ("heat" or other)
        a, b, c: Equation parameters
        epochs: Number of training epochs
        device: Computing device
        x_l, x_r: Left and right boundaries
        forcing_fn: Forcing function
        name_model: Name of the model
    """
    loss_weights = DynamicLossWeights()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_weights.parameters()), lr=1e-3)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_f = loss_function(model, x_interior, t_interior,
                             x_boundary_0, t_boundary,
                             x_boundary_pi, t_boundary_pi,
                             x_initial, t_initial,
                             boundary_funcs, initial_func,
                             equation=equation, a=a, b=b, c=c, forcing_fn=forcing_fn)

        loss = loss_weights(loss_f[0], loss_f[1], loss_f[2])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plot_history(losses, name_model)
    u_true, u_pred = plot_graphic(model, exact_solution, a, b, c, x_l, x_r, device, name_model)
    return u_true, u_pred


def test_models(models, x_interior, t_interior, x_boundary_0, t_boundary, x_boundary_pi, t_boundary_pi,
                x_initial, t_initial, boundary_funcs, initial_func, exact_solution, equation="heat",
                a=1, b=0, c=0, x_l=0, x_r=np.pi, forcing_fn=None):
    """Test multiple models.
    
    Args:
        models: List of models to test
        x_interior, t_interior: Interior points
        x_boundary_0, t_boundary: Left boundary points
        x_boundary_pi, t_boundary_pi: Right boundary points
        x_initial, t_initial: Initial condition points
        boundary_funcs: Boundary condition functions
        initial_func: Initial condition function
        exact_solution: Function computing exact solution
        equation: Type of equation ("heat" or other)
        a, b, c: Equation parameters
        x_l, x_r: Left and right boundaries
        forcing_fn: Forcing function
    """
    name_models = ["PINN"]
    results = {}
    for i, model in enumerate(models):
        name_model = name_models[i]
        u_true, u_pred = train(model, x_interior, t_interior, x_boundary_0, t_boundary,
                             x_boundary_pi, t_boundary_pi, x_initial, t_initial,
                             boundary_funcs, initial_func, exact_solution,
                             equation=equation, a=a, b=b, c=c, x_l=x_l, x_r=x_r,
                             name_model=name_model, forcing_fn=forcing_fn)
        results[name_model] = (u_true, u_pred)
    return results

import torch
import torch.nn as nn


class PINN(nn.Module):
    """Physics-Informed Neural Network (PINN) implementation.
    
    A neural network architecture that incorporates physical laws into the learning process.
    """
    def __init__(
        self,
        input_dim=2,
        hidden_dim=200,
        output_dim=1,
        num_hidden_layers=3,
        activation_fn=torch.tanh
    ):
        """Initialize PINN model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Number of neurons in hidden layers
            output_dim (int): Dimension of output
            num_hidden_layers (int): Number of hidden layers
            activation_fn (callable): Activation function to use
        """
        super(PINN, self).__init__()
        self.activation_fn = activation_fn

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        """Forward pass of the network.
        
        Args:
            x (torch.Tensor): Spatial coordinates
            t (torch.Tensor): Temporal coordinates
            
        Returns:
            torch.Tensor: Network output
        """
        inputs = torch.cat([x, t], dim=1)
        x = self.activation_fn(self.input_layer(inputs))
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)
        return x


class DGM_Layer(nn.Module):
    """Deep Galerkin Method (DGM) layer implementation.
    
    A specialized layer architecture for solving PDEs using deep learning.
    """
    def __init__(self, input_dim, hidden_dim):
        """Initialize DGM layer.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Number of neurons in hidden layer
        """
        super(DGM_Layer, self).__init__()
        self.Z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.G = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.R = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.H = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, S):
        """Forward pass of the DGM layer.
        
        Args:
            x (torch.Tensor): Input features
            S (torch.Tensor): State from previous layer
            
        Returns:
            torch.Tensor: Updated state
        """
        input_and_state = torch.cat([x, S], dim=1)
        Z = torch.tanh(self.Z(input_and_state))
        G = torch.sigmoid(self.G(input_and_state))
        R = torch.sigmoid(self.R(input_and_state))
        H = torch.tanh(self.H(torch.cat([x, S * R], dim=1)))
        return (1 - G) * H + Z * S


class RNN_PINN(nn.Module):
    """Recurrent Neural Network Physics-Informed Neural Network (RNN-PINN).
    
    Combines RNN architecture with physics-informed learning capabilities.
    """
    def __init__(self, input_dim=2, hidden_dim=64, rnn_hidden=128, output_dim=1, rnn_type="gru"):
        """Initialize RNN-PINN model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Number of neurons in MLP layers
            rnn_hidden (int): Number of neurons in RNN layer
            output_dim (int): Dimension of output
            rnn_type (str): Type of RNN to use ('gru' or 'lstm')
        """
        super(RNN_PINN, self).__init__()

        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        if rnn_type == "gru":
            self.rnn = nn.GRU(hidden_dim, rnn_hidden, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_dim, rnn_hidden, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

        self.output_mlp = nn.Sequential(
            nn.Linear(rnn_hidden, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, t):
        """Forward pass of the network.
        
        Args:
            x (torch.Tensor): Spatial coordinates of shape (N, 1)
            t (torch.Tensor): Temporal coordinates of shape (N, 1)
            
        Returns:
            torch.Tensor: Network output u(x, t) of shape (N, 1)
        """
        xt = torch.cat([x, t], dim=1)  # (N, 2)
        x_encoded = self.input_mlp(xt)  # (N, hidden_dim)

        # Add dummy time dimension for RNN
        x_seq = x_encoded.unsqueeze(1)  

        rnn_out, _ = self.rnn(x_seq) 
        out = self.output_mlp(rnn_out.squeeze(1))  

        return out



def compute_physics_loss(model, x_interior, t_interior, equation="heat", a=1.0, b=1.0, c=-0.5, forcing_fn=None):
    """Compute physics-informed loss for different PDE types.
    
    Args:
        model: Neural network model
        x_interior (torch.Tensor): Interior spatial points
        t_interior (torch.Tensor): Interior temporal points
        equation (str): Type of PDE ('heat', 'wave', or 'adv_diff_react')
        a (float): Diffusion coefficient
        b (float): Advection coefficient
        c (float): Reaction coefficient
        forcing_fn (callable, optional): Forcing function f(x,t)
        
    Returns:
        torch.Tensor: Mean squared physics loss
    """
    with torch.backends.cudnn.flags(enabled=False):
        u_pred = model(x_interior, t_interior)

    u_t = torch.autograd.grad(u_pred, t_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_interior, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    if equation == "heat":
        rhs = a * u_xx
        if forcing_fn is not None:
            rhs += forcing_fn(x_interior, t_interior)
        physics_loss = u_t - rhs

    elif equation == "wave":
        u_tt = torch.autograd.grad(u_t, t_interior, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True)[0]
        rhs = a * u_xx + b * u_t
        if forcing_fn is not None:
            rhs += forcing_fn(x_interior, t_interior)
        physics_loss = u_tt - rhs

    elif equation == "adv_diff_react":
        rhs = a * u_xx + b * u_x + c * u_pred
        if forcing_fn is not None:
            rhs += forcing_fn(x_interior, t_interior)
        physics_loss = u_t - rhs

    else:
        raise ValueError(f"Unknown equation type: {equation}")

    return torch.mean(physics_loss ** 2)


def compute_initial_condition_loss(model, x_initial, t_initial, initial_func):
    """Compute loss for initial condition.
    
    Args:
        model: Neural network model
        x_initial (torch.Tensor): Initial spatial points
        t_initial (torch.Tensor): Initial temporal points
        initial_func (callable): Initial condition function u(x,0)
        
    Returns:
        torch.Tensor: Mean squared initial condition loss
    """
    with torch.backends.cudnn.flags(enabled=False):
        u_pred_initial = model(x_initial, t_initial)

    u_true_initial = initial_func(x_initial)
    return torch.mean((u_pred_initial - u_true_initial) ** 2)


def compute_boundary_loss(model, x_boundary_l, t_boundary_l, x_boundary_r, t_boundary_r, boundary_funcs):
    """Compute loss for boundary conditions.
    
    Args:
        model: Neural network model
        x_boundary_l (torch.Tensor): Left boundary spatial points
        t_boundary_l (torch.Tensor): Left boundary temporal points
        x_boundary_r (torch.Tensor): Right boundary spatial points
        t_boundary_r (torch.Tensor): Right boundary temporal points
        boundary_funcs (dict): Dictionary of boundary condition functions
        
    Returns:
        torch.Tensor: Sum of boundary condition losses
    """
    losses = []

    for label, (x, t) in zip(["x_l", "x_r"], [(x_boundary_l, t_boundary_l), (x_boundary_r, t_boundary_r)]):
        bc = boundary_funcs[label]
        bc_type = bc["type"]
        target = bc["value"](t)
        with torch.backends.cudnn.flags(enabled=False):
            u_pred = model(x, t)

        if bc_type == "dirichlet":
            loss = torch.mean((u_pred - target) ** 2)

        elif bc_type == "neumann":
            u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            loss = torch.mean((u_x - target) ** 2)

        elif bc_type == "robin":
            u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            combo = u_x - u_pred
            loss = torch.mean((combo - target) ** 2)

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        losses.append(loss)

    return sum(losses)


class DynamicLossWeights(nn.Module):
    """Dynamic loss weighting module for multi-task learning.
    
    Implements learnable weights for physics, boundary and initial condition losses.
    """
    def __init__(self):
        """Initialize learnable loss weights."""
        super().__init__()
        self.log_sigma_physics = nn.Parameter(torch.tensor(0.0).clamp(min=-5.0))
        self.log_sigma_boundary = nn.Parameter(torch.tensor(0.0).clamp(min=-5.0))
        self.log_sigma_initial = nn.Parameter(torch.tensor(0.0).clamp(min=-5.0))

    def forward(self, physics_loss, boundary_loss, initial_loss):
        """Compute weighted sum of losses.
        
        Args:
            physics_loss (torch.Tensor): Physics-informed loss
            boundary_loss (torch.Tensor): Boundary condition loss
            initial_loss (torch.Tensor): Initial condition loss
            
        Returns:
            torch.Tensor: Weighted sum of all losses
        """
        loss = (
            0.5 * torch.exp(-2 * self.log_sigma_physics) * physics_loss + self.log_sigma_physics +
            0.5 * torch.exp(-2 * self.log_sigma_boundary) * boundary_loss + self.log_sigma_boundary +
            0.5 * torch.exp(-2 * self.log_sigma_initial) * initial_loss + self.log_sigma_initial
        )
        return loss

def loss_function(model, x_interior, t_interior, x_boundary_l, t_boundary, x_boundary_r, t_boundary_r,
                  x_initial, t_initial, boundary_funcs, initial_func, equation="wave", a=1.0, b=1.0, c=-0.5, forcing_fn=None):
    physics_loss = compute_physics_loss(model=model, x_interior=x_interior, t_interior=t_interior, equation=equation, a=a, b=b, c=c, forcing_fn=forcing_fn)
    boundary_loss = compute_boundary_loss(model, x_boundary_l, t_boundary, x_boundary_r, t_boundary_r, boundary_funcs)
    initial_loss = compute_initial_condition_loss(model, x_initial, t_initial, initial_func)
    # return physics_loss + 10*boundary_loss + 10 * initial_loss
    return physics_loss,boundary_loss,initial_loss
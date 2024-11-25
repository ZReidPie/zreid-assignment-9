import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
import seaborn as sns

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Enhanced activation functions with better numerical stability
def sigmoid(x):
    # Clip values to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Xavier/Glorot initialization for better convergence
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))
        
        # Store training history
        self.loss_history = []
        self.accuracy_history = []
        
        # Select activation function
        if activation == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2
    
    def compute_loss(self, y_true, y_pred):
        # Binary cross-entropy loss with numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_accuracy(self, y_true, y_pred):
        return np.mean((y_pred > 0.5) == y_true)

    def backward(self, X, y):
        m = X.shape[0]
        
        # Compute gradients
        dZ2 = self.A2 - y
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent updates with learning rate decay
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        # Store gradients for visualization
        self.gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }

def generate_data(n_samples=100, noise=0.1):
    np.random.seed(0)
    # Generate circular data with noise
    r = np.random.uniform(0, 2, n_samples)
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    X = np.column_stack([
        r * np.cos(theta) + noise * np.random.randn(n_samples),
        r * np.sin(theta) + noise * np.random.randn(n_samples)
    ])
    y = (r > 1.2).astype(int).reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, scatter_points=None):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    
    # Set titles and labels
    ax_input.set_title('Input Space and Decision Boundary')
    ax_hidden.set_title('Hidden Layer Representation')
    ax_gradient.set_title('Network Architecture and Gradients')
    
    # Perform training steps
    for _ in range(10):
        predictions = mlp.forward(X)
        mlp.backward(X, y)
    
    # Plot hidden space representation
    hidden_features = mlp.A1
    scatter = ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1],
                              hidden_features[:, 2], c=y.ravel(),
                              cmap='coolwarm', alpha=0.7)
    ax_hidden.view_init(elev=20, azim=frame % 360)  # Rotate view
    
    # Plot decision boundary in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid).reshape(xx.shape)
    
    # Plot contour with custom colormap
    contour = ax_input.contourf(xx, yy, Z, levels=np.linspace(0, 1, 20),
                               cmap='coolwarm', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(),
                    cmap='coolwarm', edgecolors='k')
    
    # Visualize network architecture and gradients
    nodes_pos = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.5, 0.0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1.0),
        'y': (1.0, 0.5)
    }
    
    # Plot nodes
    for node, pos in nodes_pos.items():
        ax_gradient.scatter(*pos, s=300, c='blue', alpha=0.6)
        ax_gradient.annotate(node, pos, xytext=(10, 5),
                           textcoords='offset points')
    
    # Plot connections with gradient-based thickness
    max_gradient = max(np.abs(mlp.gradients['dW1']).max(),
                      np.abs(mlp.gradients['dW2']).max())
    
    # Plot connections between layers
    for i in range(2):  # Input layer
        for j in range(3):  # Hidden layer
            gradient = np.abs(mlp.gradients['dW1'][i, j])
            width = 1 + 5 * (gradient / max_gradient)
            ax_gradient.plot([nodes_pos[f'x{i+1}'][0], nodes_pos[f'h{j+1}'][0]],
                           [nodes_pos[f'x{i+1}'][1], nodes_pos[f'h{j+1}'][1]],
                           'purple', alpha=0.6, linewidth=width)
    
    for i in range(3):  # Hidden layer to output
        gradient = np.abs(mlp.gradients['dW2'][i, 0])
        width = 1 + 5 * (gradient / max_gradient)
        ax_gradient.plot([nodes_pos[f'h{i+1}'][0], nodes_pos['y'][0]],
                        [nodes_pos[f'h{i+1}'][1], nodes_pos['y'][1]],
                        'purple', alpha=0.6, linewidth=width)
    
    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)

def visualize(activation, lr, step_num):
    # Generate dataset
    X, y = generate_data(n_samples=150, noise=0.1)
    
    # Initialize model
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1,
              lr=lr, activation=activation)
    
    # Setup visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    
    # Create animation
    num_frames = max(1, step_num // 10)
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input,
                ax_hidden=ax_hidden, ax_gradient=ax_gradient,
                X=X, y=y),
        frames=num_frames,
        repeat=False
    )
    
    # Save animation
    ani.save(os.path.join(result_dir, "visualize.gif"),
             writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    # Test different activation functions
    for activation in ['tanh', 'relu', 'sigmoid']:
        visualize(activation=activation, lr=0.1, step_num=1000)
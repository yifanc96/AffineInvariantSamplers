import numpy as np
from typing import Callable, Tuple, Optional
import warnings

class NUTSSampler:
    """
    No-U-Turn Sampler (NUTS) implementation based on Algorithm 3.
    
    This is an efficient Hamiltonian Monte Carlo sampler that automatically
    tunes the trajectory length to avoid the U-turn condition.
    """
    
    def __init__(self, 
                 log_prob_fn: Callable[[np.ndarray], float],
                 grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 step_size: float = 0.1,
                 max_treedepth: int = 10):
        """
        Initialize the NUTS sampler.
        
        Args:
            log_prob_fn: Function that computes log probability of the target distribution
            grad_log_prob_fn: Function that computes gradient of log probability
            step_size: Initial step size for leapfrog integration
            max_treedepth: Maximum tree depth to prevent infinite recursion
        """
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.step_size = step_size
        self.max_treedepth = max_treedepth
        
    def leapfrog(self, theta: np.ndarray, r: np.ndarray, 
                 v: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one leapfrog step.
        
        Args:
            theta: Position
            r: Momentum
            v: Direction (+1 or -1)
            epsilon: Step size
            
        Returns:
            New position and momentum
        """
        if v == 1:
            # Forward direction
            r_new = r + 0.5 * epsilon * self.grad_log_prob_fn(theta)
            theta_new = theta + epsilon * r_new
            r_new = r_new + 0.5 * epsilon * self.grad_log_prob_fn(theta_new)
        else:
            # Backward direction
            r_new = r - 0.5 * epsilon * self.grad_log_prob_fn(theta)
            theta_new = theta - epsilon * r_new
            r_new = r_new - 0.5 * epsilon * self.grad_log_prob_fn(theta_new)
            
        return theta_new, r_new
    
    def compute_criterion(self, theta_plus: np.ndarray, theta_minus: np.ndarray,
                         r_plus: np.ndarray, r_minus: np.ndarray) -> bool:
        """
        Compute the no-U-turn criterion.
        
        Returns True if we should continue building the tree, False otherwise.
        """
        delta_theta = theta_plus - theta_minus
        return (np.dot(delta_theta, r_plus) >= 0) and (np.dot(delta_theta, r_minus) >= 0)
    
    def build_tree(self, theta: np.ndarray, r: np.ndarray, u: float, 
                   v: int, j: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray,
                                                           np.ndarray, int, bool]:
        """
        Build a balanced binary tree for the NUTS algorithm.
        
        Args:
            theta: Current position
            r: Current momentum
            u: Slice variable
            v: Direction (+1 or -1)
            j: Tree depth
            epsilon: Step size
            
        Returns:
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime
        """
        Delta_max = 1000  # Maximum energy change to prevent numerical issues
        
        if j == 0:
            # Base case - take one leapfrog step
            theta_prime, r_prime = self.leapfrog(theta, r, v, epsilon)
            
            # Compute log probability and check slice condition
            log_prob_prime = self.log_prob_fn(theta_prime)
            kinetic_energy = 0.5 * np.dot(r_prime, r_prime)
            joint_log_prob = log_prob_prime - kinetic_energy
            
            n_prime = int(u <= np.exp(joint_log_prob))
            s_prime = int(joint_log_prob > (np.log(u) - Delta_max))
            
            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime
        
        else:
            # Recursion - build left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime = \
                self.build_tree(theta, r, u, v, j - 1, epsilon)
            
            if s_prime == 1:
                if v == -1:
                    theta_minus, r_minus, _, _, theta_double_prime, n_double_prime, s_double_prime = \
                        self.build_tree(theta_minus, r_minus, u, v, j - 1, epsilon)
                else:
                    _, _, theta_plus, r_plus, theta_double_prime, n_double_prime, s_double_prime = \
                        self.build_tree(theta_plus, r_plus, u, v, j - 1, epsilon)
                
                # Multinomial sampling
                if n_double_prime > 0:
                    prob_accept = n_double_prime / (n_prime + n_double_prime)
                    if np.random.rand() < prob_accept:
                        theta_prime = theta_double_prime
                
                # Update stopping criterion
                s_prime = s_double_prime * int(self.compute_criterion(theta_plus, theta_minus, 
                                                                    r_plus, r_minus))
                n_prime = n_prime + n_double_prime
        
        return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime
    
    def sample(self, theta_init: np.ndarray, num_samples: int, 
               warmup: int = 1000) -> Tuple[np.ndarray, dict]:
        """
        Run the NUTS sampler.
        
        Args:
            theta_init: Initial position
            num_samples: Number of samples to draw
            warmup: Number of warmup samples for step size adaptation
            
        Returns:
            Array of samples and diagnostics dictionary
        """
        dim = len(theta_init)
        samples = np.zeros((num_samples, dim))
        
        # Initialize
        theta = theta_init.copy()
        
        # Diagnostics
        n_divergent = 0
        tree_depths = []
        step_sizes = []
        
        for m in range(num_samples):
            # Resample momentum
            r = np.random.normal(0, 1, dim)
            
            # Slice variable
            log_prob_current = self.log_prob_fn(theta)
            kinetic_energy = 0.5 * np.dot(r, r)
            joint_log_prob = log_prob_current - kinetic_energy
            
            u = np.random.uniform(0, np.exp(joint_log_prob))
            
            # Initialize tree
            theta_minus = theta.copy()
            theta_plus = theta.copy()
            r_minus = r.copy()
            r_plus = r.copy()
            j = 0
            theta_m = theta.copy()
            n = 1
            s = 1
            
            # Build tree until U-turn or maximum depth
            while s == 1:
                # Choose direction
                v_j = np.random.choice([-1, 1])
                
                if v_j == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime = \
                        self.build_tree(theta_minus, r_minus, u, v_j, j, self.step_size)
                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime = \
                        self.build_tree(theta_plus, r_plus, u, v_j, j, self.step_size)
                
                if s_prime == 1:
                    # Multinomial sampling with probability min(1, n'/n)
                    prob_accept = min(1.0, n_prime / n) if n > 0 else 0.0
                    if np.random.rand() < prob_accept:
                        theta_m = theta_prime.copy()
                
                # Update variables
                n = n + n_prime
                s = s_prime * int(self.compute_criterion(theta_plus, theta_minus, 
                                                       r_plus, r_minus))
                j = j + 1
                
                # Prevent infinite loops
                if j >= self.max_treedepth:
                    warnings.warn(f"Maximum tree depth {self.max_treedepth} reached at iteration {m}")
                    break
            
            # Update position
            theta = theta_m.copy()
            samples[m] = theta
            
            # Diagnostics
            tree_depths.append(j)
            step_sizes.append(self.step_size)
            if s_prime == 0:
                n_divergent += 1
        
        diagnostics = {
            'n_divergent': n_divergent,
            'tree_depths': np.array(tree_depths),
            'step_sizes': np.array(step_sizes),
            'divergent_rate': n_divergent / num_samples
        }
        
        return samples, diagnostics


# Example usage and test functions
def example_usage():
    """Example of how to use the NUTS sampler with a multivariate normal distribution."""
    
    # Define a simple 2D normal distribution
    mu = np.array([1.0, -0.5])
    Sigma = np.array([[2.0, 0.3], [0.3, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    def log_prob_fn(theta):
        diff = theta - mu
        return -0.5 * np.dot(diff, np.dot(Sigma_inv, diff))
    
    def grad_log_prob_fn(theta):
        diff = theta - mu
        return -np.dot(Sigma_inv, diff)
    
    # Initialize sampler
    sampler = NUTSSampler(log_prob_fn, grad_log_prob_fn, step_size=0.3)
    
    # Run sampler
    theta_init = np.array([0.0, 0.0])
    samples, diagnostics = sampler.sample(theta_init, num_samples=1000)
    
    print(f"Sample mean: {np.mean(samples, axis=0)}")
    print(f"True mean: {mu}")
    print(f"Sample covariance:\n{np.cov(samples.T)}")
    print(f"True covariance:\n{Sigma}")
    print(f"Divergent transitions: {diagnostics['n_divergent']}")
    print(f"Average tree depth: {np.mean(diagnostics['tree_depths']):.2f}")
    
    return samples, diagnostics

if __name__ == "__main__":
    samples, diagnostics = example_usage()

import time

# # Setup
dim = 5
n_samples = 5000
burn_in = 1000
total_samples = n_samples + burn_in

# Create problem
np.random.seed(42)
cond_number = 1000
eigenvals = 0.1 * np.linspace(1, cond_number, dim)
H = np.random.randn(dim, dim)
Q, _ = np.linalg.qr(H)
precision = Q @ np.diag(eigenvals) @ Q.T
precision = 0.5 * (precision + precision.T)

true_mean = np.ones(dim)
initial = np.ones(dim)

# NumPy functions
def grad_log_prob_fn(x):
    diff = x - true_mean
    return -np.dot(precision, diff)

def log_prob_fn(x):
    diff = x - true_mean
    return -0.5 * np.dot(diff, np.dot(precision, diff))


## test Gaussian
start = time.time()
sampler = NUTSSampler(log_prob_fn, grad_log_prob_fn, step_size=0.5)

samples, diagnostics = sampler.sample(
    initial, num_samples=n_samples
)

time_np = time.time() - start
flat_np = samples[burn_in:, :].reshape(-1, dim)
mean_np = np.mean(flat_np, axis=0)
error_np = np.linalg.norm(mean_np - true_mean)
print("=== NUTS Results ===")
print(f"Divergent transitions: {diagnostics['n_divergent']}")
print(f"mean error={error_np:.3f}, time={time_np:.1f}s")
print(f"Average tree depth: {np.mean(diagnostics['tree_depths']):.2f}")


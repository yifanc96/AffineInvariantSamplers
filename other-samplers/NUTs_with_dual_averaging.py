import numpy as np
from typing import Callable, Tuple, Optional
import warnings

class NUTSSampler:
    """
    No-U-Turn Sampler (NUTS) implementation based on Algorithm 3.
    
    This is an efficient Hamiltonian Monte Carlo sampler that automatically
    tunes the trajectory length to avoid the U-turn condition and uses
    dual averaging for step size adaptation.
    """
    
    def __init__(self, 
                 log_prob_fn: Callable[[np.ndarray], float],
                 grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 step_size: float = 0.1,
                 max_treedepth: int = 10,
                 target_accept: float = 0.8,
                 gamma: float = 0.05,
                 t0: float = 10.0,
                 kappa: float = 0.75):
        """
        Initialize the NUTS sampler.
        
        Args:
            log_prob_fn: Function that computes log probability of the target distribution
            grad_log_prob_fn: Function that computes gradient of log probability
            step_size: Initial step size for leapfrog integration
            max_treedepth: Maximum tree depth to prevent infinite recursion
            target_accept: Target acceptance probability for dual averaging
            gamma: Dual averaging parameter controlling adaptation rate
            t0: Dual averaging parameter for stability
            kappa: Dual averaging parameter (should be in (0.5, 1])
        """
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.step_size = step_size
        self.max_treedepth = max_treedepth
        
        # Dual averaging parameters
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        
        # Initialize dual averaging state
        self.mu = np.log(10 * step_size)  # Initial log step size
        self.log_epsilon_bar = 0.0  # Running average of log step size
        self.H_bar = 0.0  # Running average of (target_accept - accept_prob)
        
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
    
    def update_step_size(self, accept_prob: float, iteration: int, warmup_length: int):
        """
        Update step size using dual averaging algorithm.
        
        Args:
            accept_prob: Acceptance probability from current iteration
            iteration: Current iteration number (0-indexed)
            warmup_length: Total number of warmup iterations
        """
        if iteration < warmup_length:
            # Dual averaging update
            self.H_bar = ((1.0 - 1.0/(iteration + 1 + self.t0)) * self.H_bar + 
                         (self.target_accept - accept_prob) / (iteration + 1 + self.t0))
            
            log_epsilon = self.mu - np.sqrt(iteration + 1) / self.gamma * self.H_bar
            eta = (iteration + 1)**(-self.kappa)
            self.log_epsilon_bar = eta * log_epsilon + (1 - eta) * self.log_epsilon_bar
            
            self.step_size = np.exp(log_epsilon)
        else:
            # After warmup, use the averaged step size
            self.step_size = np.exp(self.log_epsilon_bar)
    
    def find_reasonable_epsilon(self, theta: np.ndarray) -> float:
        """
        Find a reasonable initial step size using the heuristic from Hoffman & Gelman.
        
        Args:
            theta: Initial position
            
        Returns:
            Reasonable step size
        """
        epsilon = 1.0
        r = np.random.normal(0, 1, len(theta))
        
        # Compute initial joint log probability
        log_prob_current = self.log_prob_fn(theta)
        joint_current = log_prob_current - 0.5 * np.dot(r, r)
        
        # Take one leapfrog step
        theta_prime, r_prime = self.leapfrog(theta, r, 1, epsilon)
        log_prob_prime = self.log_prob_fn(theta_prime)
        joint_prime = log_prob_prime - 0.5 * np.dot(r_prime, r_prime)
        
        # Determine direction to adjust epsilon
        a = 2.0 * int(joint_prime - joint_current > np.log(0.5)) - 1.0
        
        # Keep adjusting until we cross the acceptance threshold
        while a * (joint_prime - joint_current) > -a * np.log(2):
            epsilon = epsilon * (2.0**a)
            
            theta_prime, r_prime = self.leapfrog(theta, r, 1, epsilon)
            log_prob_prime = self.log_prob_fn(theta_prime)
            joint_prime = log_prob_prime - 0.5 * np.dot(r_prime, r_prime)
            
            # Prevent infinite loops
            if epsilon > 1e6 or epsilon < 1e-6:
                break
        
        return epsilon
    
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
                                                           np.ndarray, int, bool, float]:
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
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime
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
            
            # Compute acceptance probability for this step
            log_prob_orig = self.log_prob_fn(theta)
            kinetic_orig = 0.5 * np.dot(r, r)
            joint_orig = log_prob_orig - kinetic_orig
            alpha_prime = min(1.0, np.exp(joint_log_prob - joint_orig))
            
            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, alpha_prime
        
        else:
            # Recursion - build left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime = \
                self.build_tree(theta, r, u, v, j - 1, epsilon)
            
            if s_prime == 1:
                if v == -1:
                    theta_minus, r_minus, _, _, theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime = \
                        self.build_tree(theta_minus, r_minus, u, v, j - 1, epsilon)
                else:
                    _, _, theta_plus, r_plus, theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime = \
                        self.build_tree(theta_plus, r_plus, u, v, j - 1, epsilon)
                
                # Multinomial sampling
                if n_double_prime > 0:
                    prob_accept = n_double_prime / (n_prime + n_double_prime)
                    if np.random.rand() < prob_accept:
                        theta_prime = theta_double_prime
                
                # Update acceptance probability (weighted average)
                total_n = n_prime + n_double_prime
                if total_n > 0:
                    alpha_prime = (n_prime * alpha_prime + n_double_prime * alpha_double_prime) / total_n
                
                # Update stopping criterion
                s_prime = s_double_prime * int(self.compute_criterion(theta_plus, theta_minus, 
                                                                    r_plus, r_minus))
                n_prime = n_prime + n_double_prime
        
        return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime
    
    def sample(self, theta_init: np.ndarray, num_samples: int, 
               warmup: int = 1000, adapt_step_size: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Run the NUTS sampler.
        
        Args:
            theta_init: Initial position
            num_samples: Number of samples to draw
            warmup: Number of warmup samples for step size adaptation
            adapt_step_size: Whether to adapt step size using dual averaging
            
        Returns:
            Array of samples and diagnostics dictionary
        """
        dim = len(theta_init)
        total_samples = warmup + num_samples
        all_samples = np.zeros((total_samples, dim))
        
        # Initialize
        theta = theta_init.copy()
        
        # Find reasonable initial step size if adapting
        if adapt_step_size:
            self.step_size = self.find_reasonable_epsilon(theta)
            self.mu = np.log(10 * self.step_size)
        
        # Diagnostics
        n_divergent = 0
        tree_depths = []
        step_sizes = []
        accept_probs = []
        
        for m in range(total_samples):
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
            alpha = 0.0  # Track acceptance probability
            n_alpha = 0   # Track number of acceptance probability computations
            
            # Build tree until U-turn or maximum depth
            while s == 1:
                # Choose direction
                v_j = np.random.choice([-1, 1])
                
                if v_j == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime, alpha_prime = \
                        self.build_tree(theta_minus, r_minus, u, v_j, j, self.step_size)
                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime = \
                        self.build_tree(theta_plus, r_plus, u, v_j, j, self.step_size)
                
                if s_prime == 1:
                    # Multinomial sampling with probability min(1, n'/n)
                    prob_accept = min(1.0, n_prime / n) if n > 0 else 0.0
                    if np.random.rand() < prob_accept:
                        theta_m = theta_prime.copy()
                
                # Update acceptance probability tracking
                if n_prime > 0:
                    alpha = (n_alpha * alpha + n_prime * alpha_prime) / (n_alpha + n_prime)
                    n_alpha += n_prime
                
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
            all_samples[m] = theta
            
            # Update step size during warmup
            if adapt_step_size and m < warmup:
                self.update_step_size(alpha, m, warmup)
            
            # Diagnostics
            tree_depths.append(j)
            step_sizes.append(self.step_size)
            accept_probs.append(alpha)
            if s_prime == 0:
                n_divergent += 1
        
        # Return only post-warmup samples
        samples = all_samples[warmup:] if warmup > 0 else all_samples
        
        diagnostics = {
            'n_divergent': n_divergent,
            'tree_depths': np.array(tree_depths),
            'step_sizes': np.array(step_sizes),
            'accept_probs': np.array(accept_probs),
            'divergent_rate': n_divergent / total_samples,
            'mean_accept_prob': np.mean(accept_probs[warmup:] if warmup > 0 else accept_probs),
            'final_step_size': self.step_size,
            'warmup_samples': warmup
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
    
    # Initialize sampler with dual averaging
    sampler = NUTSSampler(log_prob_fn, grad_log_prob_fn, 
                         step_size=0.1, target_accept=0.8)
    
    # Run sampler with warmup
    theta_init = np.array([0.0, 0.0])
    samples, diagnostics = sampler.sample(theta_init, num_samples=1000, 
                                        warmup=500, adapt_step_size=True)
    
    print("=== NUTS Sampler with Dual Averaging Results ===")
    print(f"Sample mean: {np.mean(samples, axis=0)}")
    print(f"True mean: {mu}")
    print(f"Sample covariance:\n{np.cov(samples.T)}")
    print(f"True covariance:\n{Sigma}")
    print(f"Divergent transitions: {diagnostics['n_divergent']}")
    print(f"Divergent rate: {diagnostics['divergent_rate']:.3f}")
    print(f"Mean acceptance probability: {diagnostics['mean_accept_prob']:.3f}")
    print(f"Final step size: {diagnostics['final_step_size']:.4f}")
    print(f"Average tree depth: {np.mean(diagnostics['tree_depths']):.2f}")
    print(f"Warmup samples: {diagnostics['warmup_samples']}")
    
    return samples, diagnostics


def challenging_example():
    """Example with a more challenging distribution - Neal's funnel."""
    
    def log_prob_fn(theta):
        if len(theta) < 2:
            return -np.inf
        
        # Neal's funnel: x[0] ~ N(0, 3), x[1:] ~ N(0, exp(x[0]))
        log_prob = -0.5 * (theta[0]**2) / 9.0  # x[0] ~ N(0, 3)
        
        if len(theta) > 1:
            scale = np.exp(theta[0])
            log_prob += -0.5 * np.sum(theta[1:]**2) / scale - 0.5 * (len(theta) - 1) * theta[0]
        
        return log_prob
    
    def grad_log_prob_fn(theta):
        if len(theta) < 2:
            return np.zeros_like(theta)
        
        grad = np.zeros_like(theta)
        
        # Gradient w.r.t. x[0]
        grad[0] = -theta[0] / 9.0  # From x[0] ~ N(0, 3)
        
        if len(theta) > 1:
            scale = np.exp(theta[0])
            # Additional terms from x[1:] ~ N(0, exp(x[0]))
            grad[0] += 0.5 * np.sum(theta[1:]**2) / scale - 0.5 * (len(theta) - 1)
            
            # Gradient w.r.t. x[1:]
            grad[1:] = -theta[1:] / scale
        
        return grad
    
    # Initialize sampler
    sampler = NUTSSampler(log_prob_fn, grad_log_prob_fn, 
                         step_size=0.1, target_accept=0.8)
    
    # Run sampler
    theta_init = np.array([0.0, 0.0])
    samples, diagnostics = sampler.sample(theta_init, num_samples=2000, 
                                        warmup=1000, adapt_step_size=True)
    
    print("\n=== Neal's Funnel Example ===")
    print(f"Sample mean: {np.mean(samples, axis=0)}")
    print(f"Sample std: {np.std(samples, axis=0)}")
    print(f"Divergent rate: {diagnostics['divergent_rate']:.3f}")
    print(f"Mean acceptance probability: {diagnostics['mean_accept_prob']:.3f}")
    print(f"Final step size: {diagnostics['final_step_size']:.4f}")
    
    return samples, diagnostics

if __name__ == "__main__":
    # Run basic example
    samples, diagnostics = example_usage()
    
    # Run challenging example
    funnel_samples, funnel_diagnostics = challenging_example()
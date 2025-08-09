import numpy as np
from typing import Callable, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import time

class EnsembleNUTSSampler:
    """
    Ensemble No-U-Turn Sampler (E-NUTS) implementation.
    
    Combines NUTS with ensemble methods for improved exploration and
    automatic adaptation in challenging geometries. Uses two groups of
    dim chains each that interact through ensemble-based preconditioning.
    """
    
    def __init__(self, 
                 log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 dim: int,
                 step_size: float = 0.1,
                 max_treedepth: int = 10,
                 target_accept: float = 0.8,
                 gamma: float = 0.05,
                 t0: float = 10.0,
                 kappa: float = 0.75,
                 beta: float = 0.05):
        """
        Initialize the Ensemble NUTS sampler.
        
        Args:
            log_prob_fn: Vectorized function that computes log probability (input: (n_chains, dim) -> output: (n_chains,))
            grad_log_prob_fn: Vectorized function that computes gradient (input: (n_chains, dim) -> output: (n_chains, dim))
            dim: Dimension of the problem (also number of chains per group)
            step_size: Initial step size for leapfrog integration
            max_treedepth: Maximum tree depth to prevent infinite recursion
            target_accept: Target acceptance probability for dual averaging
            gamma: Dual averaging parameter controlling adaptation rate
            t0: Dual averaging parameter for stability
            kappa: Dual averaging parameter (should be in (0.5, 1])
            beta: Ensemble interaction strength parameter
        """
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.dim = dim
        self.step_size = step_size
        self.max_treedepth = max_treedepth
        self.beta = beta
        
        # Dual averaging parameters
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        
        # Initialize dual averaging state (single step size for all chains)
        self.mu = np.log(10 * step_size)
        self.log_epsilon_bar = 0.0
        self.H_bar = 0.0
    
    def ensemble_leapfrog(self, theta: np.ndarray, r: np.ndarray, v: int, 
                         epsilon: float, complement_ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform ensemble-preconditioned leapfrog step.
        
        Args:
            theta: Positions (dim, dim)
            r: Momenta (dim, dim) - square ensemble momentum matrix
            v: Direction (+1 or -1)
            epsilon: Step size (single value for all chains)
            complement_ensemble: The other ensemble group (dim, dim)
            
        Returns:
            New positions and momenta
        """
        # Compute centered complement ensemble for preconditioning
        complement_mean = np.mean(complement_ensemble, axis=0)
        centered_complement = (complement_ensemble - complement_mean) / np.sqrt(self.dim)
        
        if v == 1:
            # Forward direction: standard leapfrog with ensemble preconditioning
            
            # Initial half-step for momentum
            grad = self.grad_log_prob_fn(theta)
            momentum_update = np.dot(grad, centered_complement.T) * (self.beta * epsilon * 0.5)
            r_new = r + momentum_update
            
            # Full position step
            position_update = np.dot(r_new, centered_complement) * (self.beta * epsilon)
            theta_new = theta + position_update
            
            # Final half-step for momentum
            grad_new = self.grad_log_prob_fn(theta_new)
            momentum_update_final = np.dot(grad_new, centered_complement.T) * (self.beta * epsilon * 0.5)
            r_new = r_new + momentum_update_final
            
        else:
            # Backward direction: reverse leapfrog
            
            # Initial half-step for momentum  
            grad = self.grad_log_prob_fn(theta)
            momentum_update = np.dot(grad, centered_complement.T) * (self.beta * epsilon * 0.5)
            r_new = r - momentum_update
            
            # Full position step
            position_update = np.dot(r_new, centered_complement) * (self.beta * epsilon)
            theta_new = theta - position_update
            
            # Final half-step for momentum
            grad_new = self.grad_log_prob_fn(theta_new)
            momentum_update_final = np.dot(grad_new, centered_complement.T) * (self.beta * epsilon * 0.5)
            r_new = r_new - momentum_update_final
            
        return theta_new, r_new
    
    def update_step_size(self, accept_prob: float, iteration: int, warmup_length: int):
        """
        Update step size using dual averaging algorithm.
        
        Args:
            accept_prob: Mean acceptance probability across all chains
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
    
    def compute_criterion(self, theta_plus: np.ndarray, theta_minus: np.ndarray,
                         r_plus: np.ndarray, r_minus: np.ndarray, 
                         complement_ensemble: np.ndarray) -> np.ndarray:
        """
        Compute the no-U-turn criterion for each chain.
        Convert ensemble momentum back to position space for U-turn detection.
        """
        delta_theta = theta_plus - theta_minus
        
        # Convert ensemble momentum back to position space
        complement_mean = np.mean(complement_ensemble, axis=0)
        centered_complement = (complement_ensemble - complement_mean) / np.sqrt(self.dim)
        
        p_plus = np.dot(r_plus, centered_complement)
        p_minus = np.dot(r_minus, centered_complement)
        
        # U-turn criterion: both endpoints should have positive dot product with trajectory
        dot_plus = np.sum(delta_theta * p_plus, axis=1)
        dot_minus = np.sum(delta_theta * p_minus, axis=1)
        
        return (dot_plus >= 0) & (dot_minus >= 0)
    
    def build_tree_ensemble(self, theta: np.ndarray, r: np.ndarray, u: np.ndarray, 
                           v: int, j: int, epsilon: float, 
                           complement_ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
        """Build trees for ensemble of chains simultaneously."""
        Delta_max = 1000
        n_chains = theta.shape[0]
        
        if j == 0:
            # Base case - single leapfrog step
            theta_prime, r_prime = self.ensemble_leapfrog(theta, r, v, epsilon, complement_ensemble)
            
            # Compute energies and slice conditions
            log_prob_prime = self.log_prob_fn(theta_prime)
            kinetic_energy = 0.5 * np.sum(r_prime**2, axis=1)
            joint_log_prob = log_prob_prime - kinetic_energy
            
            # Safe slice sampling conditions
            u_safe = np.clip(u, 1e-300, 1.0)
            log_u = np.log(u_safe)
            
            n_prime = (log_u <= np.clip(joint_log_prob, -1000, 1000)).astype(int)
            s_prime = (joint_log_prob > (log_u - Delta_max)).astype(int)
            
            # Acceptance probabilities
            log_prob_orig = self.log_prob_fn(theta)
            kinetic_orig = 0.5 * np.sum(r**2, axis=1)
            joint_orig = log_prob_orig - kinetic_orig
            
            joint_diff = np.clip(joint_log_prob - joint_orig, -1000, 1000)
            alpha_prime = np.minimum(1.0, np.exp(joint_diff))
            
            return (theta_prime, r_prime, theta_prime, r_prime, 
                   theta_prime, n_prime, s_prime, alpha_prime)
        
        else:
            # Recursive case
            (theta_minus, r_minus, theta_plus, r_plus, 
             theta_prime, n_prime, s_prime, alpha_prime) = \
                self.build_tree_ensemble(theta, r, u, v, j - 1, epsilon, complement_ensemble)
            
            # Continue if any chain wants to continue
            if np.any(s_prime == 1):
                if v == -1:
                    (theta_minus, r_minus, _, _, 
                     theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime) = \
                        self.build_tree_ensemble(theta_minus, r_minus, u, v, j - 1, 
                                               epsilon, complement_ensemble)
                else:
                    (_, _, theta_plus, r_plus, 
                     theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime) = \
                        self.build_tree_ensemble(theta_plus, r_plus, u, v, j - 1, 
                                               epsilon, complement_ensemble)
                
                # Multinomial sampling
                total_n = n_prime + n_double_prime
                valid_chains = total_n > 0
                
                if np.any(valid_chains):
                    prob_accept = np.zeros(n_chains)
                    prob_accept[valid_chains] = n_double_prime[valid_chains] / total_n[valid_chains]
                    
                    accept_mask = (np.random.rand(n_chains) < prob_accept) & valid_chains
                    theta_prime[accept_mask] = theta_double_prime[accept_mask]
                
                # Update acceptance probabilities and stopping criterion
                alpha_prime = np.where(total_n > 0,
                                     (n_prime * alpha_prime + n_double_prime * alpha_double_prime) / total_n,
                                     alpha_prime)
                
                s_prime = (s_double_prime * 
                          self.compute_criterion(theta_plus, theta_minus, r_plus, r_minus, 
                                               complement_ensemble).astype(int))
                n_prime = total_n
        
        return (theta_minus, r_minus, theta_plus, r_plus, 
               theta_prime, n_prime, s_prime, alpha_prime)
    
    def sample(self, theta_init: np.ndarray, num_samples: int, 
               warmup: int = 1000, adapt_step_size: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Run the Ensemble NUTS sampler.
        
        Args:
            theta_init: Initial position (will be replicated for all chains)
            num_samples: Number of samples to draw
            warmup: Number of warmup samples for step size adaptation
            adapt_step_size: Whether to adapt step size using dual averaging
            
        Returns:
            Array of samples from all chains and diagnostics dictionary
        """
        total_chains = 2 * self.dim
        total_samples = warmup + num_samples
        
        # Initialize chains with small perturbations
        theta_ensemble = (np.tile(theta_init, (total_chains, 1)) + 
                         0.1 * np.random.randn(total_chains, self.dim))
        
        # Split into two equal groups
        group1_slice = slice(0, self.dim)
        group2_slice = slice(self.dim, total_chains)
        
        # Find initial step sizes if adapting
        if adapt_step_size:
            self.mu = np.log(10 * self.step_size)
        
        step_size = self.step_size  # Single step size for all chains
        
        # Storage
        all_samples = np.zeros((total_samples, total_chains, self.dim))
        
        # Diagnostics
        n_divergent = 0
        tree_depths = []
        step_sizes_history = []
        accept_probs = []
        
        for m in range(total_samples):
            # Store current positions
            all_samples[m] = theta_ensemble
            
            # Update each group
            group1_accepts, group1_depths, divergent1 = self.update_group(
                theta_ensemble[group1_slice], theta_ensemble[group2_slice], step_size)
            
            group2_accepts, group2_depths, divergent2 = self.update_group(
                theta_ensemble[group2_slice], theta_ensemble[group1_slice], step_size)
            
            # Combine diagnostics
            all_accepts = np.concatenate([group1_accepts, group2_accepts])
            all_depths = np.concatenate([group1_depths, group2_depths])
            n_divergent += divergent1 + divergent2
            
            # Update step size if adapting
            if adapt_step_size:
                mean_accept = np.mean(all_accepts)
                self.update_step_size(mean_accept, m, warmup)
                step_size = self.step_size
            
            # Store diagnostics
            tree_depths.append(all_depths)
            step_sizes_history.append(step_size)
            accept_probs.append(all_accepts)
            
            # Print progress occasionally
            if (m + 1) % 1000 == 0:
                mean_accept = np.mean(all_accepts)
                print(f"Iteration {m+1}/{total_samples}, Mean accept prob: {mean_accept:.3f}")
        
        # Return post-warmup samples
        samples = all_samples[warmup:] if warmup > 0 else all_samples
        
        diagnostics = {
            'n_divergent': n_divergent,
            'tree_depths': np.array(tree_depths),
            'step_sizes': np.array(step_sizes_history),
            'accept_probs': np.array(accept_probs),
            'divergent_rate': n_divergent / (total_samples * total_chains),
            'mean_accept_prob': np.mean(accept_probs[warmup:] if warmup > 0 else accept_probs),
            'final_step_size': step_size,
            'warmup_samples': warmup
        }
        
        return samples, diagnostics
    
    def update_group(self, group_theta: np.ndarray, complement_theta: np.ndarray, 
                    step_size: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Update one group of chains using NUTS with ensemble preconditioning.
        
        Args:
            group_theta: Positions of chains to update (dim, dim)
            complement_theta: Positions of complement ensemble (dim, dim)  
            step_size: Step size (single value for all chains)
            
        Returns:
            Acceptance probabilities, tree depths, number of divergent transitions
        """
        n_chains = group_theta.shape[0]
        
        # Generate square ensemble momentum matrix (dim x dim)
        r = np.random.randn(n_chains, n_chains)
        
        # Slice variables
        log_prob_current = self.log_prob_fn(group_theta)
        kinetic_energy = 0.5 * np.sum(r**2, axis=1)
        joint_log_prob = log_prob_current - kinetic_energy
        
        # Safe slice sampling
        joint_log_prob = np.clip(joint_log_prob, -1000, 1000)
        u = np.random.uniform(0, 1, n_chains) * np.exp(joint_log_prob)
        
        # Initialize trees for all chains
        theta_minus = group_theta.copy()
        theta_plus = group_theta.copy()
        r_minus = r.copy()
        r_plus = r.copy()
        theta_m = group_theta.copy()
        
        j = 0
        n = np.ones(n_chains, dtype=int)
        s = np.ones(n_chains, dtype=int)
        alpha = np.zeros(n_chains)
        n_alpha = np.zeros(n_chains)
        n_divergent = 0
        
        # Track final tree depth for each chain individually
        # Initialize to -1 to indicate not yet terminated
        final_tree_depths = np.full(n_chains, -1, dtype=int)
        
        # Build trees until U-turn or max depth
        while np.any(s == 1) and j < self.max_treedepth:
            # Store which chains are active before this iteration
            active_before = (s == 1).copy()
            
            # Choose direction randomly
            v_j = np.random.choice([-1, 1])
            
            # Build tree in chosen direction
            if v_j == -1:
                (theta_minus, r_minus, _, _, 
                 theta_prime, n_prime, s_prime, alpha_prime) = \
                    self.build_tree_ensemble(theta_minus, r_minus, u, -1, j, 
                                           step_size, complement_theta)
            else:
                (_, _, theta_plus, r_plus, 
                 theta_prime, n_prime, s_prime, alpha_prime) = \
                    self.build_tree_ensemble(theta_plus, r_plus, u, 1, j, 
                                           step_size, complement_theta)
            
            # Update only for chains that haven't stopped
            active_chains = s == 1
            if not np.any(active_chains):
                break
            
            # Multinomial update for active chains
            total_n = n + n_prime
            valid_chains = (total_n > 0) & active_chains
            
            if np.any(valid_chains):
                prob_accept = np.zeros(n_chains)
                prob_accept[valid_chains] = np.minimum(1.0, n_prime[valid_chains] / n[valid_chains])
                accept_mask = (np.random.rand(n_chains) < prob_accept) & valid_chains
                theta_m[accept_mask] = theta_prime[accept_mask]
            
            # Update acceptance probability tracking
            mask_prime = n_prime > 0
            if np.any(mask_prime):
                alpha[mask_prime] = ((n_alpha[mask_prime] * alpha[mask_prime] + 
                                   n_prime[mask_prime] * alpha_prime[mask_prime]) / 
                                  (n_alpha[mask_prime] + n_prime[mask_prime]))
                n_alpha[mask_prime] += n_prime[mask_prime]
            
            # Update variables
            n = total_n
            
            # Increment depth BEFORE checking termination
            j += 1
            
            # Record depth for chains that terminate in this iteration
            # (were active before, but stopped now)
            newly_stopped = active_before & (s_prime == 0)
            final_tree_depths[newly_stopped] = j
            
            # Count divergent transitions for newly stopped chains
            if np.any(newly_stopped):
                n_divergent += np.sum(newly_stopped)
            
            # Update stopping criterion
            s = s_prime
        
        # For chains still active at max depth, record max depth
        still_active = (s == 1)
        final_tree_depths[still_active] = j
        
        # Handle chains that stopped in first iteration (should have depth 1, not -1)
        never_updated = (final_tree_depths == -1)
        final_tree_depths[never_updated] = 1
        
        # Update positions
        group_theta[:] = theta_m
        
        return alpha, final_tree_depths, n_divergent


# Test with high-dimensional Gaussian - same as standard NUTS
def test_ensemble_nuts():
    """Test the Ensemble NUTS sampler on a high-dimensional Gaussian distribution."""
    print("=== Testing Ensemble NUTS Sampler ===")
    
    # Setup - same as standard NUTS test
    dim = 5
    n_samples = 4000
    burn_in = 1000
    
    # Create problem
    np.random.seed(42)
    cond_number = 10000
    eigenvals = 0.1 * np.linspace(1, cond_number, dim)
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    precision = Q @ np.diag(eigenvals) @ Q.T
    precision = 0.5 * (precision + precision.T)
    
    true_mean = np.ones(dim)
    initial = np.ones(dim)
    
    def log_prob_fn(x):
        """Vectorized log probability function."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - true_mean
        return -0.5 * np.sum(diff @ precision * diff, axis=1)
    
    def grad_log_prob_fn(x):
        """Vectorized gradient function."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - true_mean
        return -(diff @ precision)
    
    # Test Ensemble NUTS
    start = time.time()
    sampler = EnsembleNUTSSampler(
        log_prob_fn=log_prob_fn,
        grad_log_prob_fn=grad_log_prob_fn,
        dim=dim,
        max_treedepth = 5,
        step_size=1.0,
        target_accept=0.8,
        beta=1.0
    )
    
    samples, diagnostics = sampler.sample(
        initial, num_samples=n_samples, warmup=burn_in, adapt_step_size=True
    )
    
    time_ensemble = time.time() - start
    
    # Flatten samples from all chains
    flat_samples = samples.reshape(-1, dim)
    mean_ensemble = np.mean(flat_samples, axis=0)
    error_ensemble = np.linalg.norm(mean_ensemble - true_mean)
    
    print(f"=== Ensemble NUTS Results ===")
    print(f"Total chains: {2 * dim}")
    print(f"Samples shape: {samples.shape}")
    print(f"Divergent transitions: {diagnostics['n_divergent']}")
    print(f"Divergent rate: {diagnostics['divergent_rate']:.4f}")
    print(f"Mean error: {error_ensemble:.3f}, Time: {time_ensemble:.1f}s")
    print(f"Average tree depth: {np.mean(diagnostics['tree_depths']):.2f}")
    print(f"Mean acceptance probability: {diagnostics['mean_accept_prob']:.3f}")
    print(f"Final step size: {diagnostics['final_step_size']:.4f}")
    
    return samples, diagnostics

if __name__ == "__main__":
    samples, diagnostics = test_ensemble_nuts()
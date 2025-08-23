import numpy as np
from typing import Callable, Tuple
import warnings
import time

class AffineInvariantEnsembleNUTSSampler:
    """
    Affine Invariant Ensemble No-U-Turn Sampler (AIE-NUTS) implementation.
    
    Uses two groups of chains, where groups interact through 
    ensemble-based preconditioning. Each group uses the other as a 
    complement ensemble for momentum preconditioning.
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
                 beta: float = 1.0):
        """
        Initialize Affine Invariant Ensemble NUTS sampler.
        
        Args:
            log_prob_fn: Vectorized log probability function (n_chains, dim) -> (n_chains,)
            grad_log_prob_fn: Vectorized gradient function (n_chains, dim) -> (n_chains, dim)
            dim: Problem dimension
            step_size: Initial step size
            max_treedepth: Maximum tree depth
            target_accept: Target acceptance probability for dual averaging
            gamma, t0, kappa: Dual averaging parameters
            beta: Ensemble interaction strength
        """
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.dim = dim
        self.step_size = step_size
        self.max_treedepth = max_treedepth
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.beta = beta
        
        # Dual averaging state (single step size for all chains)
        self.mu = np.log(10 * step_size)
        self.log_epsilon_bar = 0.0
        self.H_bar = 0.0
    
    def update_step_size(self, accept_prob: float, iteration: int, warmup_length: int):
        """Update step size using dual averaging."""
        if iteration < warmup_length:
            self.H_bar = ((1.0 - 1.0/(iteration + 1 + self.t0)) * self.H_bar + 
                         (self.target_accept - accept_prob) / (iteration + 1 + self.t0))
            
            log_epsilon = self.mu - np.sqrt(iteration + 1) / self.gamma * self.H_bar
            eta = (iteration + 1)**(-self.kappa)
            self.log_epsilon_bar = eta * log_epsilon + (1 - eta) * self.log_epsilon_bar
            
            self.step_size = np.exp(log_epsilon)
        else:
            self.step_size = np.exp(self.log_epsilon_bar)
    
    def compute_covariance_inv(self, complement_ensemble: np.ndarray) -> np.ndarray:
        """Compute inverse empirical covariance of complement ensemble."""
        n_complement = complement_ensemble.shape[0]
        
        # Center the complement ensemble
        complement_mean = np.mean(complement_ensemble, axis=0)
        centered = (complement_ensemble - complement_mean) / np.sqrt(n_complement)
        
        # Empirical covariance: C^T @ C
        emp_cov = np.dot(centered.T, centered)
        
        # Add regularization and invert
        reg = 1e-6 * np.eye(self.dim)
        try:
            return np.linalg.inv(emp_cov + reg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(emp_cov + reg)
    
    def ensemble_leapfrog(self, theta: np.ndarray, r: np.ndarray, 
                         epsilon: float, complement_ensemble: np.ndarray, 
                         direction: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble leapfrog step.
        
        Args:
            theta: Positions (n_chains_group, dim)
            r: Momenta (n_chains_group, n_complement) - ensemble momentum!
            epsilon: Step size
            complement_ensemble: Complement group positions (n_complement, dim)
            direction: +1 for forward, -1 for backward
        """
        n_complement = complement_ensemble.shape[0]
        
        # Compute centered complement
        complement_mean = np.mean(complement_ensemble, axis=0)
        centered_complement = (complement_ensemble - complement_mean) / np.sqrt(n_complement)
        
        if direction == 1:
            # Forward: half momentum, full position, half momentum
            grad = self.grad_log_prob_fn(theta)
            # Momentum update: grad projected onto complement ensemble
            r_half = r + 0.5 * epsilon * self.beta * np.dot(grad, centered_complement.T)
            
            # Position update: momentum projected back to position space
            theta_new = theta + epsilon * self.beta * np.dot(r_half, centered_complement)
            
            grad_new = self.grad_log_prob_fn(theta_new)
            r_new = r_half + 0.5 * epsilon * self.beta * np.dot(grad_new, centered_complement.T)
        else:
            # Backward: reverse the leapfrog
            grad = self.grad_log_prob_fn(theta)
            r_half = r - 0.5 * epsilon * self.beta * np.dot(grad, centered_complement.T)
            
            theta_new = theta - epsilon * self.beta * np.dot(r_half, centered_complement)
            
            grad_new = self.grad_log_prob_fn(theta_new)
            r_new = r_half - 0.5 * epsilon * self.beta * np.dot(grad_new, centered_complement.T)
        
        return theta_new, r_new
    
    def compute_uturn_criterion(self, theta_plus: np.ndarray, theta_minus: np.ndarray,
                               r_plus: np.ndarray, r_minus: np.ndarray,
                               complement_ensemble: np.ndarray,
                               cov_inv: np.ndarray) -> np.ndarray:
        """
        Compute U-turn criterion using ensemble-weighted metric.
        
        Returns boolean array: True means continue, False means stop.
        """
        n_complement = complement_ensemble.shape[0]
        delta_theta = theta_plus - theta_minus
        
        # Convert ensemble momentum to position space
        complement_mean = np.mean(complement_ensemble, axis=0)
        centered_complement = (complement_ensemble - complement_mean) / np.sqrt(n_complement)
        
        p_plus = np.dot(r_plus, centered_complement)
        p_minus = np.dot(r_minus, centered_complement)
        
        # Weighted inner products: delta_theta^T * cov_inv * p
        weighted_delta = np.dot(delta_theta, cov_inv)
        dot_plus = np.sum(weighted_delta * p_plus, axis=1)
        dot_minus = np.sum(weighted_delta * p_minus, axis=1)
        
        return (dot_plus >= 0) & (dot_minus >= 0)
    
    def build_tree(self, theta: np.ndarray, r: np.ndarray, u: np.ndarray,
                   direction: int, depth: int, epsilon: float,
                   complement_ensemble: np.ndarray, cov_inv: np.ndarray):
        """
        Build NUTS tree for ensemble of chains.
        
        Returns: theta_minus, r_minus, theta_plus, r_plus, theta_prime, 
                n_prime, s_prime, alpha_prime
        """
        if depth == 0:
            # Base case: single leapfrog step
            theta_prime, r_prime = self.ensemble_leapfrog(
                theta, r, epsilon, complement_ensemble, direction)
            
            # Compute log probabilities and energies
            log_prob_prime = self.log_prob_fn(theta_prime)
            log_prob_orig = self.log_prob_fn(theta)
            
            kinetic_prime = 0.5 * np.sum(r_prime**2, axis=1)
            kinetic_orig = 0.5 * np.sum(r**2, axis=1)
            
            joint_prime = log_prob_prime - kinetic_prime
            joint_orig = log_prob_orig - kinetic_orig
            
            # Slice condition
            log_u = np.log(np.clip(u, 1e-300, 1.0))
            n_prime = (log_u <= joint_prime).astype(int)
            s_prime = (joint_prime > log_u - 1000).astype(int)
            
            # Acceptance probability
            alpha_prime = np.minimum(1.0, np.exp(joint_prime - joint_orig))
            
            return (theta_prime, r_prime, theta_prime, r_prime,
                   theta_prime, n_prime, s_prime, alpha_prime)
        
        else:
            # Recursive case
            # Build first subtree
            (theta_minus, r_minus, theta_plus, r_plus,
             theta_prime, n_prime, s_prime, alpha_prime) = self.build_tree(
                theta, r, u, direction, depth - 1, epsilon, complement_ensemble, cov_inv)
            
            if np.any(s_prime == 1):
                # Build second subtree
                if direction == -1:
                    (theta_minus, r_minus, _, _, theta_double_prime,
                     n_double_prime, s_double_prime, alpha_double_prime) = self.build_tree(
                        theta_minus, r_minus, u, direction, depth - 1, epsilon,
                        complement_ensemble, cov_inv)
                else:
                    (_, _, theta_plus, r_plus, theta_double_prime,
                     n_double_prime, s_double_prime, alpha_double_prime) = self.build_tree(
                        theta_plus, r_plus, u, direction, depth - 1, epsilon,
                        complement_ensemble, cov_inv)
                
                # Multinomial sampling
                total_n = n_prime + n_double_prime
                valid = total_n > 0
                
                if np.any(valid):
                    prob = np.zeros(len(theta))
                    prob[valid] = n_double_prime[valid] / total_n[valid]
                    accept_mask = (np.random.rand(len(theta)) < prob) & valid
                    theta_prime[accept_mask] = theta_double_prime[accept_mask]
                
                # Update acceptance probability
                alpha_prime = np.where(total_n > 0,
                                     (n_prime * alpha_prime + n_double_prime * alpha_double_prime) / total_n,
                                     alpha_prime)
                
                # Update stopping criterion
                continue_mask = self.compute_uturn_criterion(
                    theta_plus, theta_minus, r_plus, r_minus, complement_ensemble, cov_inv)
                s_prime = s_double_prime * continue_mask.astype(int)
                n_prime = total_n
            
            return (theta_minus, r_minus, theta_plus, r_plus,
                   theta_prime, n_prime, s_prime, alpha_prime)
    
    def nuts_step(self, theta: np.ndarray, complement_ensemble: np.ndarray,
                  epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single NUTS step for a group of chains.
        
        Returns: new_theta, acceptance_probs, tree_depths
        """
        n_chains = theta.shape[0]
        n_complement = complement_ensemble.shape[0]
        
        # Precompute covariance inverse
        cov_inv = self.compute_covariance_inv(complement_ensemble)
        
        # Sample momentum in ensemble space (n_chains, n_complement)
        r = np.random.randn(n_chains, n_complement)
        
        # Compute slice variable
        log_prob_current = self.log_prob_fn(theta)
        kinetic_current = 0.5 * np.sum(r**2, axis=1)
        joint_current = log_prob_current - kinetic_current
        u = np.random.uniform(0, 1, n_chains) * np.exp(joint_current)
        
        # Initialize tree
        theta_minus = theta.copy()
        theta_plus = theta.copy()
        r_minus = r.copy()
        r_plus = r.copy()
        theta_new = theta.copy()
        
        depth = 0
        n = np.ones(n_chains, dtype=int)
        s = np.ones(n_chains, dtype=int)
        alpha_sum = np.zeros(n_chains)
        n_alpha = np.zeros(n_chains)
        
        # Track final tree depth for each chain individually
        # Initialize to -1 to indicate not yet terminated
        final_tree_depths = np.full(n_chains, -1, dtype=int)
        
        # Build tree until U-turn or max depth
        while np.any(s == 1) and depth < self.max_treedepth:
            # Store which chains are active before this iteration
            active_before = (s == 1).copy()
            
            # Choose direction
            direction = np.random.choice([-1, 1])
            
            # Expand tree
            if direction == -1:
                (theta_minus, r_minus, _, _, theta_prime,
                 n_prime, s_prime, alpha_prime) = self.build_tree(
                    theta_minus, r_minus, u, direction, depth, epsilon,
                    complement_ensemble, cov_inv)
            else:
                (_, _, theta_plus, r_plus, theta_prime,
                 n_prime, s_prime, alpha_prime) = self.build_tree(
                    theta_plus, r_plus, u, direction, depth, epsilon,
                    complement_ensemble, cov_inv)
            
            # Update positions
            if np.any(s_prime == 1):
                prob = np.minimum(1.0, n_prime / n)
                accept_mask = (np.random.rand(n_chains) < prob) & (s_prime == 1)
                theta_new[accept_mask] = theta_prime[accept_mask]
            
            # Track acceptance probabilities
            valid_alpha = n_prime > 0
            if np.any(valid_alpha):
                alpha_sum[valid_alpha] += n_prime[valid_alpha] * alpha_prime[valid_alpha]
                n_alpha[valid_alpha] += n_prime[valid_alpha]
            
            # Update counts and stopping
            n += n_prime
            
            # Increment depth BEFORE checking termination
            depth += 1
            
            # Record depth for chains that terminate in this iteration
            # (were active before, but stopped now)
            newly_stopped = active_before & (s_prime == 0)
            final_tree_depths[newly_stopped] = depth
            
            # Update stopping criterion
            s = s_prime
        
        # For chains still active at max depth, record max depth
        still_active = (s == 1)
        final_tree_depths[still_active] = depth
        
        # Handle chains that stopped in first iteration (should have depth 1, not -1)
        never_updated = (final_tree_depths == -1)
        final_tree_depths[never_updated] = 1
        
        # Compute final acceptance probabilities
        accept_probs = np.where(n_alpha > 0, alpha_sum / n_alpha, 0.0)
        
        return theta_new, accept_probs, final_tree_depths
    
    def sample(self, theta_init: np.ndarray, num_samples: int,
               total_chains: int = None, warmup: int = 1000, 
               adapt_step_size: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Run Ensemble NUTS sampling.
        
        Args:
            theta_init: Initial position (will be replicated)
            num_samples: Number of post-warmup samples
            total_chains: Total number of chains to use (default: 2*dim, minimum: 4)
            warmup: Number of warmup samples
            adapt_step_size: Whether to adapt step size
            
        Returns:
            samples: (total_samples, total_chains, dim) array
            diagnostics: Dictionary of diagnostic information
        """
        if total_chains is None:
            total_chains = max(4, 2 * self.dim)
            
        if total_chains < 4:
            raise ValueError("total_chains must be at least 4 to form meaningful ensemble groups")
            
        total_samples = warmup + num_samples
        
        # Split chains into two groups
        group1_size = total_chains // 2
        group2_size = total_chains - group1_size
        
        print(f"Using {total_chains} total chains: Group 1 ({group1_size}), Group 2 ({group2_size})")
        
        # Initialize ensemble
        theta_ensemble = np.tile(theta_init, (total_chains, 1))
        theta_ensemble += 0.1 * np.random.randn(total_chains, self.dim)
        
        # Split into groups
        group1 = theta_ensemble[:group1_size]
        group2 = theta_ensemble[group1_size:]
        
        # Storage
        samples = np.zeros((total_samples, total_chains, self.dim))
        accept_probs_history = []
        tree_depths_history = []
        step_sizes_history = []
        
        for i in range(total_samples):
            # Store current state
            samples[i, :group1_size] = group1
            samples[i, group1_size:] = group2
            
            # Update group 1 using group 2 as complement
            group1, accept1, depths1 = self.nuts_step(group1, group2, self.step_size)
            
            # Update group 2 using group 1 as complement  
            group2, accept2, depths2 = self.nuts_step(group2, group1, self.step_size)
            
            # Combine diagnostics
            all_accepts = np.concatenate([accept1, accept2])
            all_depths = np.concatenate([depths1, depths2])
            
            # Adapt step size
            if adapt_step_size:
                mean_accept = np.mean(all_accepts)
                self.update_step_size(mean_accept, i, warmup)
            
            # Store diagnostics
            accept_probs_history.append(all_accepts)
            tree_depths_history.append(all_depths)
            step_sizes_history.append(self.step_size)
            
            # Progress
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i+1}/{total_samples}, "
                      f"Mean accept: {np.mean(all_accepts):.3f}, "
                      f"Step size: {self.step_size:.4f}")
        
        # Return post-warmup samples
        post_warmup_samples = samples[warmup:] if warmup > 0 else samples
        
        diagnostics = {
            'accept_probs': np.array(accept_probs_history),
            'tree_depths': np.array(tree_depths_history),
            'step_sizes': np.array(step_sizes_history),
            'mean_accept_prob': np.mean(accept_probs_history[warmup:] if warmup > 0 else accept_probs_history),
            'final_step_size': self.step_size,
            'warmup_samples': warmup,
            'total_chains': total_chains,
            'group_sizes': (group1_size, group2_size)
        }
        
        return post_warmup_samples, diagnostics


def test_ensemble_nuts():
    """Test Ensemble NUTS on high-dimensional Gaussian with different chain counts."""
    print("=== Testing Flexible Ensemble NUTS Sampler ===")
    
    # Problem setup
    dim = 20
    n_samples = 5000
    warmup = 1000
    
    np.random.seed(42)
    cond_number = 10000
    eigenvals = 0.1 * np.linspace(1, cond_number, dim)
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    precision = Q @ np.diag(eigenvals) @ Q.T
    precision = 0.5 * (precision + precision.T)
    
    true_mean = np.ones(dim)
    initial = np.ones(dim)
    
    # Vectorized target functions
    def log_prob_fn(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - true_mean
        return -0.5 * np.sum(diff @ precision * diff, axis=1)
    
    def grad_log_prob_fn(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - true_mean
        return -(diff @ precision)
    
    # Test with different numbers of chains
    chain_counts = [2*dim]
    
    for total_chains in chain_counts:
        print(f"\n--- Testing with {total_chains} chains ---")
        
        # Run Ensemble NUTS
        start_time = time.time()
        sampler = AffineInvariantEnsembleNUTSSampler(
            log_prob_fn=log_prob_fn,
            grad_log_prob_fn=grad_log_prob_fn,
            max_treedepth=5,
            dim=dim,
            step_size=1.0,
            target_accept=0.8,
            beta=1.0
        )
        
        samples, diagnostics = sampler.sample(
            initial, num_samples=n_samples, warmup=warmup, 
            total_chains=total_chains, adapt_step_size=True
        )

        elapsed_time = time.time() - start_time
        
        # Analysis
        flat_samples = samples.reshape(-1, dim)
        sample_mean = np.mean(flat_samples, axis=0)
        mean_error = np.linalg.norm(sample_mean - true_mean)
        
        print(f"Total chains: {diagnostics['total_chains']}")
        print(f"Group sizes: {diagnostics['group_sizes']}")
        print(f"Samples shape: {samples.shape}")
        print(f"Mean error: {mean_error:.4f}")
        print(f"Time: {elapsed_time:.1f}s")
        print(f"Mean acceptance: {diagnostics['mean_accept_prob']:.3f}")
        print(f"Final step size: {diagnostics['final_step_size']:.4f}")
        print(f"Average tree depth: {np.mean(diagnostics['tree_depths']):.2f}")
    
    return samples, diagnostics

if __name__ == "__main__":
    samples, diagnostics = test_ensemble_nuts()
import numpy as np
from typing import Callable, Tuple
import time

class HamiltonianSideMoveEnsembleNUTS:
    """
    Hamiltonian Side Move Ensemble NUTS Sampler.
    
    Uses two groups of chains where each group performs NUTS steps with 
    Hamiltonian side moves using the complement group for ensemble interaction.
    """
    
    def __init__(self, 
                 log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 dim: int,
                 step_size: float = 0.1,
                 max_treedepth: int = 10,
                 target_accept: float = 0.8):
        """
        Initialize Hamiltonian Side Move Ensemble NUTS sampler.
        
        Args:
            log_prob_fn: Vectorized log probability function (n_chains, dim) -> (n_chains,)
            grad_log_prob_fn: Vectorized gradient function (n_chains, dim) -> (n_chains, dim)
            dim: Problem dimension
            step_size: Initial step size
            max_treedepth: Maximum tree depth
            target_accept: Target acceptance probability for dual averaging
        """
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.dim = dim
        self.step_size = step_size
        self.max_treedepth = max_treedepth
        self.target_accept = target_accept
        
        # Dual averaging parameters
        self.gamma = 0.05
        self.t0 = 10.0
        self.kappa = 0.75
        
        # Dual averaging state
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
    
    def direction(self, complement_ensemble: np.ndarray, n_chains: int) -> np.ndarray:
        """
        Generate side move directions using complement ensemble.
        
        Args:
            complement_ensemble: Complement group positions (n_complement, dim)
            n_chains: Number of chains to generate directions for
            
        Returns:
            side_directions: Direction vectors (n_chains, dim)
        """
        n_complement = complement_ensemble.shape[0]
        
        # Choose two different random complement chains for each active chain
        complement_indices1 = np.random.choice(n_complement, size=n_chains, replace=True)
        complement_indices2 = np.random.choice(n_complement, size=n_chains, replace=True)
        
        # Ensure we have different complement chains when possible
        if n_complement > 1:
            mask = complement_indices1 == complement_indices2
            while np.any(mask):
                complement_indices2[mask] = np.random.choice(n_complement, size=np.sum(mask), replace=True)
                mask = complement_indices1 == complement_indices2
        
        chosen_complements1 = complement_ensemble[complement_indices1]  # (n_chains, dim)
        chosen_complements2 = complement_ensemble[complement_indices2]  # (n_chains, dim)
        
        # Side direction: difference between two complement particles, scaled by 1/sqrt(2*dim)
        side_directions = (chosen_complements1 - chosen_complements2) / np.sqrt(2 * self.dim)  # (n_chains, dim)
        
        return side_directions
    
    def leapfrog_step(self, theta: np.ndarray, r: np.ndarray, epsilon: float, 
                     side_directions: np.ndarray, direction: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leapfrog step with fixed side directions.
        
        Args:
            theta: Positions (n_chains, dim)
            r: Momenta (n_chains,) - scalar momentum for each chain
            epsilon: Step size
            side_directions: Fixed side move directions for this iteration (n_chains, dim)
            direction: +1 for forward, -1 for backward
            
        Returns:
            new_theta: Updated positions
            new_r: Updated momenta
        """
        if direction == 1:
            # Forward leapfrog: half momentum, full position, half momentum
            grad = self.grad_log_prob_fn(theta)
            # Momentum update: project gradient onto side direction
            grad_proj = np.sum(grad * side_directions, axis=1)  # (n_chains,)
            r_half = r + 0.5 * epsilon * grad_proj
            
            # Position update using scalar momentum and side directions
            theta_new = theta + epsilon * r_half.reshape(-1, 1) * side_directions
            
            # Final momentum update
            grad_new = self.grad_log_prob_fn(theta_new)
            grad_proj_new = np.sum(grad_new * side_directions, axis=1)
            r_new = r_half + 0.5 * epsilon * grad_proj_new
        else:
            # Backward leapfrog
            grad = self.grad_log_prob_fn(theta)
            grad_proj = np.sum(grad * side_directions, axis=1)
            r_half = r - 0.5 * epsilon * grad_proj
            
            theta_new = theta - epsilon * r_half.reshape(-1, 1) * side_directions
            
            grad_new = self.grad_log_prob_fn(theta_new)
            grad_proj_new = np.sum(grad_new * side_directions, axis=1)
            r_new = r_half - 0.5 * epsilon * grad_proj_new
        
        return theta_new, r_new
    
    def compute_uturn_criterion(self, theta_plus: np.ndarray, theta_minus: np.ndarray,
                               r_plus: np.ndarray, r_minus: np.ndarray,
                               side_directions: np.ndarray) -> np.ndarray:
        """
        Compute U-turn criterion for Hamiltonian side move.
        
        Args:
            theta_plus, theta_minus: Boundary positions
            r_plus, r_minus: Boundary momenta (scalars)
            side_directions: Side move directions (n_chains, dim)
            
        Returns:
            continue_mask: Boolean array, True means continue building tree
        """
        # U-turn criterion: both terms should be non-negative
        # Term 1: (theta_plus - theta_minus) · side_directions * r_plus >= 0
        # Term 2: (theta_plus - theta_minus) · side_directions * r_minus >= 0
        delta_theta = theta_plus - theta_minus  # (n_chains, dim)
        
        # Project position difference onto side directions
        delta_theta_proj = np.sum(delta_theta * side_directions, axis=1)  # (n_chains,)
        
        # Two terms for U-turn criterion
        term_plus = delta_theta_proj * r_plus   # (n_chains,)
        term_minus = delta_theta_proj * r_minus # (n_chains,)
        
        # U-turn condition: both terms should be non-negative
        return (term_plus >= 0) & (term_minus >= 0)
    
    def build_tree(self, theta: np.ndarray, r: np.ndarray, u: np.ndarray,
                   direction: int, depth: int, epsilon: float,
                   side_directions: np.ndarray):
        """
        Recursively build NUTS tree with fixed side directions.
        
        Returns:
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, 
            n_prime, s_prime, alpha_prime
        """
        if depth == 0:
            # Base case: single leapfrog step with fixed side directions
            theta_prime, r_prime = self.leapfrog_step(theta, r, epsilon, side_directions, direction)
            
            # Compute energies
            log_prob_prime = self.log_prob_fn(theta_prime)
            log_prob_orig = self.log_prob_fn(theta)
            
            kinetic_prime = 0.5 * r_prime**2  # Scalar momentum
            kinetic_orig = 0.5 * r**2
            
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
            # Recursive case: build subtrees
            (theta_minus, r_minus, theta_plus, r_plus,
             theta_prime, n_prime, s_prime, alpha_prime) = self.build_tree(
                theta, r, u, direction, depth - 1, epsilon, side_directions)
            
            if np.any(s_prime == 1):
                # Build second subtree
                if direction == -1:
                    (theta_minus, r_minus, _, _, theta_double_prime,
                     n_double_prime, s_double_prime, alpha_double_prime) = self.build_tree(
                        theta_minus, r_minus, u, direction, depth - 1, epsilon, side_directions)
                else:
                    (_, _, theta_plus, r_plus, theta_double_prime,
                     n_double_prime, s_double_prime, alpha_double_prime) = self.build_tree(
                        theta_plus, r_plus, u, direction, depth - 1, epsilon, side_directions)
                
                # Multinomial sampling for proposal
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
                
                # Check U-turn criterion using fixed side directions
                continue_mask = self.compute_uturn_criterion(theta_plus, theta_minus, r_plus, r_minus, side_directions)
                s_prime = s_double_prime * continue_mask.astype(int)
                n_prime = total_n
            
            return (theta_minus, r_minus, theta_plus, r_plus,
                   theta_prime, n_prime, s_prime, alpha_prime)
    
    def nuts_step(self, theta: np.ndarray, complement_ensemble: np.ndarray, 
                  epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single NUTS step for a group of chains.
        
        Args:
            theta: Current positions (n_chains, dim)
            complement_ensemble: Complement group positions (n_complement, dim)
            epsilon: Step size
            
        Returns:
            new_theta: Updated positions
            accept_probs: Acceptance probabilities for each chain
            tree_depths: Tree depths reached for each chain
        """
        n_chains = theta.shape[0]
        
        # Generate side directions ONCE at the beginning of the iteration
        side_directions = self.direction(complement_ensemble, n_chains)
        
        # Sample scalar momentum for each chain
        r = np.random.randn(n_chains)
        
        # Compute slice variable
        log_prob_current = self.log_prob_fn(theta)
        kinetic_current = 0.5 * r**2  # Scalar kinetic energy
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
        final_tree_depths = np.zeros(n_chains, dtype=int)
        
        # Build tree until U-turn or max depth (side_directions fixed throughout)
        while np.any(s == 1) and depth < self.max_treedepth:
            # Choose direction randomly
            direction = np.random.choice([-1, 1])
            
            # Expand tree in chosen direction using fixed side directions
            if direction == -1:
                (theta_minus, r_minus, _, _, theta_prime,
                 n_prime, s_prime, alpha_prime) = self.build_tree(
                    theta_minus, r_minus, u, direction, depth, epsilon, side_directions)
            else:
                (_, _, theta_plus, r_plus, theta_prime,
                 n_prime, s_prime, alpha_prime) = self.build_tree(
                    theta_plus, r_plus, u, direction, depth, epsilon, side_directions)
            
            # Update positions with multinomial sampling
            if np.any(s_prime == 1):
                prob = np.minimum(1.0, n_prime / n)
                accept_mask = (np.random.rand(n_chains) < prob) & (s_prime == 1)
                theta_new[accept_mask] = theta_prime[accept_mask]
            
            # Track acceptance probabilities
            valid_alpha = n_prime > 0
            if np.any(valid_alpha):
                alpha_sum[valid_alpha] += n_prime[valid_alpha] * alpha_prime[valid_alpha]
                n_alpha[valid_alpha] += n_prime[valid_alpha]
            
            # Update counters and stopping criterion
            n += n_prime
            
            # Update final tree depth for chains that are still active
            depth += 1
            final_tree_depths[s == 1] = depth
            
            # Update stopping criterion
            s = s_prime
        
        # For chains that never stopped (reached max depth), set to max depth
        final_tree_depths[final_tree_depths == 0] = depth
        
        # Compute final acceptance probabilities
        accept_probs = np.where(n_alpha > 0, alpha_sum / n_alpha, 0.0)
        
        return theta_new, accept_probs, final_tree_depths
    
    def sample(self, theta_init: np.ndarray, num_samples: int,
               total_chains: int = None, warmup: int = 1000, 
               adapt_step_size: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Run Hamiltonian Side Move Ensemble NUTS sampling.
        
        Args:
            theta_init: Initial position (will be replicated across chains)
            num_samples: Number of post-warmup samples
            total_chains: Total number of chains (default: max(4, 2*dim))
            warmup: Number of warmup samples
            adapt_step_size: Whether to adapt step size during warmup
            
        Returns:
            samples: Array of shape (num_samples, total_chains, dim)
            diagnostics: Dictionary with diagnostic information
        """
        if total_chains is None:
            total_chains = max(4, 2 * self.dim)
            
        if total_chains < 4:
            raise ValueError("total_chains must be at least 4")
        
        # Split chains into two groups
        group1_size = total_chains // 2
        group2_size = total_chains - group1_size
        
        print(f"Running with {total_chains} chains: Group 1 ({group1_size}), Group 2 ({group2_size})")
        
        # Initialize chains
        theta_ensemble = np.tile(theta_init.reshape(1, -1), (total_chains, 1))
        theta_ensemble += 0.1 * np.random.randn(total_chains, self.dim)
        
        group1 = theta_ensemble[:group1_size]
        group2 = theta_ensemble[group1_size:]
        
        # Storage
        total_iterations = warmup + num_samples
        samples = np.zeros((total_iterations, total_chains, self.dim))
        accept_probs_history = []
        tree_depths_history = []
        step_sizes_history = []
        
        # Sampling loop
        for i in range(total_iterations):
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
            
            # Adapt step size during warmup
            if adapt_step_size and i < warmup:
                mean_accept = np.mean(all_accepts)
                self.update_step_size(mean_accept, i, warmup)
            
            # Store diagnostics
            accept_probs_history.append(all_accepts)
            tree_depths_history.append(all_depths)
            step_sizes_history.append(self.step_size)
            
            # Progress reporting
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i+1}/{total_iterations}, "
                      f"Accept rate: {np.mean(all_accepts):.3f}, "
                      f"Step size: {self.step_size:.4f}, "
                      f"Avg tree depth: {np.mean(all_depths):.1f}")
        
        # Return post-warmup samples
        post_warmup_samples = samples[warmup:] if warmup > 0 else samples
        
        diagnostics = {
            'accept_probs': np.array(accept_probs_history),
            'tree_depths': np.array(tree_depths_history),
            'step_sizes': np.array(step_sizes_history),
            'mean_accept_prob': np.mean(accept_probs_history[warmup:] if warmup > 0 else accept_probs_history),
            'final_step_size': self.step_size,
            'warmup_iterations': warmup,
            'total_chains': total_chains,
            'group_sizes': (group1_size, group2_size)
        }
        
        return post_warmup_samples, diagnostics


def test_ensemble_nuts():
    """Test Ensemble NUTS on high-dimensional Gaussian with different chain counts."""
    print("=== Testing Hamiltonian Side Move Ensemble NUTS Sampler ===")
    
    # Problem setup
    dim = 5
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
    
    # Run Ensemble NUTS
    start_time = time.time()
    sampler = HamiltonianSideMoveEnsembleNUTS(
        log_prob_fn=log_prob_fn,
        grad_log_prob_fn=grad_log_prob_fn,
        dim=dim,
        step_size=1.0,
        max_treedepth=5,
        target_accept=0.8
    )
    
    samples, diagnostics = sampler.sample(
        initial, num_samples=n_samples, total_chains=2*dim, warmup=warmup, adapt_step_size=True
    )

    elapsed_time = time.time() - start_time
    
    # Analysis
    flat_samples = samples.reshape(-1, dim)
    sample_mean = np.mean(flat_samples, axis=0)
    mean_error = np.linalg.norm(sample_mean - true_mean)
    
    print(f"\n=== Results ===")
    print(f"Total chains: {diagnostics['total_chains']}")
    print(f"Samples shape: {samples.shape}")
    print(f"Mean error: {mean_error:.4f}")
    print(f"Time: {elapsed_time:.1f}s")
    print(f"Mean acceptance: {diagnostics['mean_accept_prob']:.3f}")
    print(f"Final step size: {diagnostics['final_step_size']:.4f}")
    print(f"Average tree depth: {np.mean(diagnostics['tree_depths']):.2f}")
    
    return samples, diagnostics


if __name__ == "__main__":
    samples, diagnostics = test_ensemble_nuts()
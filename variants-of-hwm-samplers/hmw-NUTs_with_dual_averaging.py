import numpy as np
from typing import Callable, Tuple, Optional
import warnings

class EnsembleNUTSSampler:
    """
    Ensemble No-U-Turn Sampler (E-NUTS) implementation.
    
    Combines NUTS with ensemble methods for improved exploration and
    automatic adaptation in challenging geometries. Uses multiple chains
    that interact through ensemble-based preconditioning.
    """
    
    def __init__(self, 
                 log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
                 n_chains_per_group: int = 5,
                 step_size: float = 0.1,
                 max_treedepth: int = 3,
                 target_accept: float = 0.8,
                 gamma: float = 0.05,
                 t0: float = 10.0,
                 kappa: float = 0.75,
                 beta: float = 0.05):
        """
        Initialize the Ensemble NUTS sampler.
        
        Args:
            log_prob_fn: Function that computes log probability (vectorized for multiple samples)
            grad_log_prob_fn: Function that computes gradient (vectorized for multiple samples)
            n_chains_per_group: Number of chains per ensemble group
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
        self.n_chains_per_group = n_chains_per_group
        self.step_size = step_size
        self.max_treedepth = max_treedepth
        self.beta = beta
        
        # Dual averaging parameters
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        
        # Initialize dual averaging state for each chain
        self.mu = np.log(10 * step_size)
        self.log_epsilon_bar = np.zeros(2 * n_chains_per_group)
        self.H_bar = np.zeros(2 * n_chains_per_group)
    
    def update_step_sizes(self, accept_probs: np.ndarray, iteration: int, warmup_length: int):
        """
        Update step sizes using dual averaging algorithm for each chain.
        
        Args:
            accept_probs: Acceptance probabilities for each chain
            iteration: Current iteration number (0-indexed)
            warmup_length: Total number of warmup iterations
        """
        if iteration < warmup_length:
            for i in range(len(accept_probs)):
                # Dual averaging update for each chain
                self.H_bar[i] = ((1.0 - 1.0/(iteration + 1 + self.t0)) * self.H_bar[i] + 
                               (self.target_accept - accept_probs[i]) / (iteration + 1 + self.t0))
                
                log_epsilon = self.mu - np.sqrt(iteration + 1) / self.gamma * self.H_bar[i]
                eta = (iteration + 1)**(-self.kappa)
                self.log_epsilon_bar[i] = eta * log_epsilon + (1 - eta) * self.log_epsilon_bar[i]
        
        # Use averaged step sizes
        step_sizes = np.exp(self.log_epsilon_bar if iteration >= warmup_length else 
                           self.mu - np.sqrt(max(1, iteration + 1)) / self.gamma * self.H_bar)
        return np.clip(step_sizes, 1e-6, 10.0)  # Reasonable bounds
    
    def ensemble_leapfrog(self, theta: np.ndarray, r: np.ndarray, v: int, 
                         epsilon: np.ndarray, complement_ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform ensemble-preconditioned leapfrog step.
        
        Args:
            theta: Positions (n_chains, dim)
            r: Momenta (n_chains, n_chains) - ensemble structure
            v: Direction (+1 or -1)
            epsilon: Step sizes for each chain
            complement_ensemble: The other ensemble group for preconditioning
            
        Returns:
            New positions and momenta
        """
        n_chains, dim = theta.shape
        
        # Compute centered complement ensemble for preconditioning
        centered_complement = ((complement_ensemble - np.mean(complement_ensemble, axis=0)) / 
                              np.sqrt(len(complement_ensemble)))
        
        # Reshape epsilon for broadcasting
        eps_reshaped = epsilon.reshape(-1, 1)
        beta_eps = self.beta * eps_reshaped
        beta_eps_half = beta_eps / 2
        
        if v == 1:
            # Forward direction
            # Initial half-step for momentum
            grad = self.grad_log_prob_fn(theta)
            grad = np.nan_to_num(grad, nan=0.0)
            
            # Ensemble momentum update: r -= β*ε/2 * grad @ centered_complement.T
            momentum_update = beta_eps_half.flatten()[:, np.newaxis] * np.dot(grad, centered_complement.T)
            r_new = r - momentum_update
            
            # Full position step: θ += β*ε * r @ centered_complement
            position_update = beta_eps.flatten()[:, np.newaxis] * np.dot(r_new, centered_complement)
            theta_new = theta + position_update
            
            # Final half-step for momentum
            grad_new = self.grad_log_prob_fn(theta_new)
            grad_new = np.nan_to_num(grad_new, nan=0.0)
            momentum_update_final = beta_eps_half.flatten()[:, np.newaxis] * np.dot(grad_new, centered_complement.T)
            r_new = r_new - momentum_update_final
        else:
            # Backward direction
            # Initial half-step for momentum
            grad = self.grad_log_prob_fn(theta)
            grad = np.nan_to_num(grad, nan=0.0)
            
            momentum_update = beta_eps_half.flatten()[:, np.newaxis] * np.dot(grad, centered_complement.T)
            r_new = r + momentum_update
            
            # Full position step
            position_update = beta_eps.flatten()[:, np.newaxis] * np.dot(r_new, centered_complement)
            theta_new = theta - position_update
            
            # Final half-step for momentum
            grad_new = self.grad_log_prob_fn(theta_new)
            grad_new = np.nan_to_num(grad_new, nan=0.0)
            momentum_update_final = beta_eps_half.flatten()[:, np.newaxis] * np.dot(grad_new, centered_complement.T)
            r_new = r_new + momentum_update_final
            
        return theta_new, r_new
    
    def compute_criterion(self, theta_plus: np.ndarray, theta_minus: np.ndarray,
                         r_plus: np.ndarray, r_minus: np.ndarray, 
                         complement_ensemble: np.ndarray) -> np.ndarray:
        """
        Compute the no-U-turn criterion for each chain.
        
        For ensemble NUTS, we convert ensemble momentum back to position space
        for proper U-turn detection.
        
        Returns boolean array indicating which chains should continue.
        """
        delta_theta = theta_plus - theta_minus
        n_chains = delta_theta.shape[0]
        
        # Convert ensemble momentum back to position space momentum
        # The ensemble momentum is in the space of the complement ensemble
        centered_complement = ((complement_ensemble - np.mean(complement_ensemble, axis=0)) / 
                              np.sqrt(len(complement_ensemble)))
        
        # Convert momentum: p_position = r_ensemble @ centered_complement
        p_plus = np.dot(r_plus, centered_complement)
        p_minus = np.dot(r_minus, centered_complement)
        
        # Compute dot products for U-turn criterion
        dot_plus = np.sum(delta_theta * p_plus, axis=1)
        dot_minus = np.sum(delta_theta * p_minus, axis=1)
        
        return (dot_plus >= 0) & (dot_minus >= 0)
    
    def build_tree_ensemble(self, theta: np.ndarray, r: np.ndarray, u: np.ndarray, 
                           v: int, j: int, epsilon: np.ndarray, 
                           complement_ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
        """
        Build trees for ensemble of chains simultaneously.
        """
        Delta_max = 1000
        n_chains = theta.shape[0]
        
        if j == 0:
            # Base case - take one leapfrog step for all chains
            theta_prime, r_prime = self.ensemble_leapfrog(theta, r, v, epsilon, complement_ensemble)
            
            # Compute log probabilities and check slice conditions
            log_prob_prime = self.log_prob_fn(theta_prime)
            
            # Kinetic energy for ensemble momentum: sum over each chain's momentum vector
            kinetic_energy = 0.5 * np.sum(r_prime**2, axis=1)
            joint_log_prob = log_prob_prime - kinetic_energy
            
            n_prime = (u <= np.exp(np.clip(joint_log_prob, -1000, 1000))).astype(int)
            s_prime = (joint_log_prob > (np.log(np.clip(u, 1e-300, 1.0)) - Delta_max)).astype(int)
            
            # Compute acceptance probabilities
            log_prob_orig = self.log_prob_fn(theta)
            kinetic_orig = 0.5 * np.sum(r**2, axis=1)
            joint_orig = log_prob_orig - kinetic_orig
            
            # Clip to prevent overflow in exp
            joint_diff = np.clip(joint_log_prob - joint_orig, -1000, 1000)
            alpha_prime = np.minimum(1.0, np.exp(joint_diff))
            
            return (theta_prime, r_prime, theta_prime, r_prime, 
                   theta_prime, n_prime, s_prime, alpha_prime)
        
        else:
            # Recursion - build subtrees
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
                
                # Multinomial sampling for each chain
                total_n = n_prime + n_double_prime
                valid_chains = total_n > 0
                
                if np.any(valid_chains):
                    prob_accept = np.zeros(n_chains)
                    prob_accept[valid_chains] = n_double_prime[valid_chains] / total_n[valid_chains]
                    
                    # Update theta_prime where accepted
                    accept_mask = (np.random.rand(n_chains) < prob_accept) & valid_chains
                    theta_prime[accept_mask] = theta_double_prime[accept_mask]
                
                # Update acceptance probabilities
                alpha_prime = np.where(total_n > 0,
                                     (n_prime * alpha_prime + n_double_prime * alpha_double_prime) / total_n,
                                     alpha_prime)
                
                # Update stopping criterion
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
            warmup: Number of warmup samples
            adapt_step_size: Whether to adapt step sizes
            
        Returns:
            Array of samples from all chains and diagnostics
        """
        dim = len(theta_init)
        total_chains = 2 * self.n_chains_per_group
        total_samples = warmup + num_samples
        
        # Initialize chains with small perturbations
        theta_ensemble = (np.tile(theta_init, (total_chains, 1)) + 
                         0.1 * np.random.randn(total_chains, dim))
        
        # Split into two groups
        group1 = slice(0, self.n_chains_per_group)
        group2 = slice(self.n_chains_per_group, total_chains)
        
        # Storage
        all_samples = np.zeros((total_chains, total_samples, dim))
        
        # Diagnostics
        n_divergent = np.zeros(total_chains)
        tree_depths = []
        step_sizes_history = []
        accept_probs_history = []
        
        for m in range(total_samples):
            all_samples[:, m] = theta_ensemble
            
            # Get current step sizes
            if adapt_step_size:
                current_step_sizes = self.update_step_sizes(
                    np.ones(total_chains) * 0.8,  # Placeholder, will be updated
                    m, warmup)
            else:
                current_step_sizes = np.full(total_chains, self.step_size)
            
            # Update Group 1 using Group 2 as complement
            group1_accepts, group1_depths = self.update_group(
                theta_ensemble[group1], theta_ensemble[group2], 
                current_step_sizes[group1], group1)
            
            # Update Group 2 using Group 1 as complement  
            group2_accepts, group2_depths = self.update_group(
                theta_ensemble[group2], theta_ensemble[group1], 
                current_step_sizes[group2], group2)
            
            # Combine diagnostics
            all_accepts = np.concatenate([group1_accepts, group2_accepts])
            all_depths = np.concatenate([group1_depths, group2_depths])
            
            # Update step sizes if adapting
            if adapt_step_size and m < warmup:
                self.update_step_sizes(all_accepts, m, warmup)
            
            # Store diagnostics
            tree_depths.append(all_depths)
            step_sizes_history.append(current_step_sizes.copy())
            accept_probs_history.append(all_accepts)
        
        # Return post-warmup samples
        samples = all_samples[:, warmup:] if warmup > 0 else all_samples
        
        diagnostics = {
            'n_divergent': n_divergent,
            'tree_depths': np.array(tree_depths),
            'step_sizes': np.array(step_sizes_history),
            'accept_probs': np.array(accept_probs_history),
            'mean_accept_prob': np.mean(accept_probs_history[warmup:] if warmup > 0 else accept_probs_history),
            'warmup_samples': warmup,
            'n_chains': total_chains
        }
        
        return samples, diagnostics
    
    def update_group(self, group_theta: np.ndarray, complement_theta: np.ndarray, 
                    step_sizes: np.ndarray, group_slice: slice) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update one group of chains using NUTS with ensemble preconditioning.
        """
        n_chains, dim = group_theta.shape
        
        # Generate ensemble-structured momentum
        r = np.random.randn(n_chains, n_chains)
        
        # Slice variables
        log_prob_current = self.log_prob_fn(group_theta)
        # Kinetic energy for ensemble momentum structure
        kinetic_energy = 0.5 * np.sum(r**2, axis=1)
        joint_log_prob = log_prob_current - kinetic_energy
        
        # Prevent numerical issues with slice sampling
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
        
        # Build trees until U-turn or max depth
        while np.any(s == 1) and j < self.max_treedepth:
            # Choose directions for each chain
            v_j = np.random.choice([-1, 1], size=n_chains)
            
            # Process chains that chose -1 direction
            mask_neg = (v_j == -1) & (s == 1)
            neg_indices = np.where(mask_neg)[0]
            
            if len(neg_indices) > 0:
                (theta_minus_updated, r_minus_updated, _, _, 
                 theta_prime_neg, n_prime_neg, s_prime_neg, alpha_prime_neg) = \
                    self.build_tree_ensemble(theta_minus[neg_indices], r_minus[neg_indices], 
                                           u[neg_indices], -1, j, step_sizes[neg_indices], 
                                           complement_theta)
                
                # Update the arrays
                theta_minus[neg_indices] = theta_minus_updated
                r_minus[neg_indices] = r_minus_updated
                
                # Update for negative direction chains
                total_n_neg = n[neg_indices] + n_prime_neg
                valid_neg = total_n_neg > 0
                
                if np.any(valid_neg):
                    valid_neg_indices = neg_indices[valid_neg]
                    prob_accept_neg = n_prime_neg[valid_neg] / total_n_neg[valid_neg]
                    accept_mask_neg = np.random.rand(len(valid_neg_indices)) < prob_accept_neg
                    
                    # Update theta_m for accepted chains
                    accepted_neg_indices = valid_neg_indices[accept_mask_neg]
                    theta_m[accepted_neg_indices] = theta_prime_neg[valid_neg][accept_mask_neg]
                
                # Update alpha for all negative direction chains
                alpha[neg_indices] = np.where(
                    total_n_neg > 0,
                    (n[neg_indices] * alpha[neg_indices] + n_prime_neg * alpha_prime_neg) / total_n_neg,
                    alpha[neg_indices]
                )
                
                n[neg_indices] = total_n_neg
                s[neg_indices] = s_prime_neg
            
            # Process chains that chose +1 direction
            mask_pos = (v_j == 1) & (s == 1)
            pos_indices = np.where(mask_pos)[0]
            
            if len(pos_indices) > 0:
                (_, _, theta_plus_updated, r_plus_updated, 
                 theta_prime_pos, n_prime_pos, s_prime_pos, alpha_prime_pos) = \
                    self.build_tree_ensemble(theta_plus[pos_indices], r_plus[pos_indices], 
                                           u[pos_indices], 1, j, step_sizes[pos_indices], 
                                           complement_theta)
                
                # Update the arrays
                theta_plus[pos_indices] = theta_plus_updated
                r_plus[pos_indices] = r_plus_updated
                
                # Update for positive direction chains
                total_n_pos = n[pos_indices] + n_prime_pos
                valid_pos = total_n_pos > 0
                
                if np.any(valid_pos):
                    valid_pos_indices = pos_indices[valid_pos]
                    prob_accept_pos = n_prime_pos[valid_pos] / total_n_pos[valid_pos]
                    accept_mask_pos = np.random.rand(len(valid_pos_indices)) < prob_accept_pos
                    
                    # Update theta_m for accepted chains
                    accepted_pos_indices = valid_pos_indices[accept_mask_pos]
                    theta_m[accepted_pos_indices] = theta_prime_pos[valid_pos][accept_mask_pos]
                
                # Update alpha for all positive direction chains
                alpha[pos_indices] = np.where(
                    total_n_pos > 0,
                    (n[pos_indices] * alpha[pos_indices] + n_prime_pos * alpha_prime_pos) / total_n_pos,
                    alpha[pos_indices]
                )
                
                n[pos_indices] = total_n_pos
                s[pos_indices] = s_prime_pos
            
            j += 1
        
        # Update the group
        group_theta[:] = theta_m
        
        return alpha, np.full(n_chains, j)


# Example usage
def example_ensemble_usage():
    """Example using Ensemble NUTS with a challenging distribution."""
    
    # Define a 2D mixture of Gaussians (challenging for single-chain methods)
    def log_prob_fn(theta_batch):
        if theta_batch.ndim == 1:
            theta_batch = theta_batch[np.newaxis, :]
        
        # Two well-separated Gaussians
        mu1 = np.array([-2.0, 0.0])
        mu2 = np.array([2.0, 0.0])
        sigma = 0.5
        
        # Log probabilities for each component
        log_prob1 = -0.5 * np.sum((theta_batch - mu1)**2, axis=1) / sigma**2
        log_prob2 = -0.5 * np.sum((theta_batch - mu2)**2, axis=1) / sigma**2
        
        # Log-sum-exp for mixture
        max_log_prob = np.maximum(log_prob1, log_prob2)
        log_prob = max_log_prob + np.log(np.exp(log_prob1 - max_log_prob) + 
                                        np.exp(log_prob2 - max_log_prob)) + np.log(0.5)
        
        return log_prob
    
    def grad_log_prob_fn(theta_batch):
        if theta_batch.ndim == 1:
            theta_batch = theta_batch[np.newaxis, :]
            
        n_batch = theta_batch.shape[0]
        grads = np.zeros_like(theta_batch)
        
        mu1 = np.array([-2.0, 0.0])
        mu2 = np.array([2.0, 0.0])
        sigma = 0.5
        
        for i in range(n_batch):
            theta = theta_batch[i]
            
            # Compute component probabilities
            log_prob1 = -0.5 * np.sum((theta - mu1)**2) / sigma**2
            log_prob2 = -0.5 * np.sum((theta - mu2)**2) / sigma**2
            
            # Softmax weights
            max_log_prob = max(log_prob1, log_prob2)
            w1 = np.exp(log_prob1 - max_log_prob)
            w2 = np.exp(log_prob2 - max_log_prob)
            total_w = w1 + w2
            
            if total_w > 0:
                w1 /= total_w
                w2 /= total_w
                
                # Weighted gradient
                grad1 = -(theta - mu1) / sigma**2
                grad2 = -(theta - mu2) / sigma**2
                grads[i] = w1 * grad1 + w2 * grad2
        
        return grads
    
    # Initialize ensemble sampler
    sampler = EnsembleNUTSSampler(
        log_prob_fn, grad_log_prob_fn,
        n_chains_per_group=4,
        step_size=0.5,
        target_accept=0.8,
        beta=0.5
    )
    
    # Run sampler
    theta_init = np.array([0.0, 0.0])
    samples, diagnostics = sampler.sample(
        theta_init, num_samples=4000, 
        warmup=1000, adapt_step_size=True
    )
    
    print("=== Ensemble NUTS Results ===")
    print(f"Number of chains: {diagnostics['n_chains']}")
    print(f"Mean acceptance probability: {diagnostics['mean_accept_prob']:.3f}")
    print(f"Sample means across chains:")
    for i in range(samples.shape[0]):
        print(f"  Chain {i}: {np.mean(samples[i], axis=0)}")
    
    # Combine all chains for overall statistics
    all_samples = samples.reshape(-1, samples.shape[-1])
    print(f"Combined sample mean: {np.mean(all_samples, axis=0)}")
    print(f"Combined sample std: {np.std(all_samples, axis=0)}")
    
    return samples, diagnostics

if __name__ == "__main__":
    samples, diagnostics = example_ensemble_usage()
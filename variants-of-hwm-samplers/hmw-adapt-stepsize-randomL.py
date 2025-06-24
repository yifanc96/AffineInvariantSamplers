import numpy as np

def hamiltonian_walk_move_adaptive(gradient_func, potential_func, initial, n_samples, n_chains_per_group=5, 
                                 epsilon_init=0.01, n_leapfrog=10, beta=0.05, 
                                 target_accept=0.8, gamma=0.05, t0=10, kappa=0.75,
                                 adapt_steps=None, randomize_steps=True):
    """
    Vectorized Hamiltonian Walk Move sampler with dual averaging for adaptive step size.
    
    Parameters:
    -----------
    gradient_func : callable
        Function that computes gradients of the log probability
    potential_func : callable  
        Function that computes the negative log probability (potential energy)
    initial : array_like
        Initial state
    n_samples : int
        Number of samples to generate
    n_chains_per_group : int
        Number of chains per group (default: 5)
    epsilon_init : float
        Initial step size (default: 0.01)
    n_leapfrog : int
        Number of leapfrog steps (default: 10)
    beta : float
        Preconditioning parameter (default: 0.05)
    target_accept : float
        Target acceptance rate for adaptation (default: 0.8)
    gamma : float
        Dual averaging parameter (default: 0.05)
    t0 : float
        Dual averaging parameter (default: 10)
    kappa : float
        Dual averaging parameter, should be in (0.5, 1] (default: 0.75)
    adapt_steps : int or None
        Number of steps to adapt for. If None, adapts for all steps (default: None)
    randomize_steps : bool
        Whether to randomize the number of leapfrog steps for better ergodicity (default: True)
    
    Returns:
    --------
    samples : ndarray
        Generated samples of shape (total_chains, n_samples, *original_shape)
    acceptance_rates : ndarray
        Final acceptance rates for each chain
    epsilon_history : ndarray
        History of step sizes during adaptation
    """
    
    # Initialize
    orig_dim = initial.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    # Set adaptation period
    if adapt_steps is None:
        adapt_steps = n_samples
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1 = slice(0, n_chains_per_group)
    group2 = slice(n_chains_per_group, total_chains)
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts = np.zeros(total_chains)
    
    # Dual averaging variables
    epsilon = epsilon_init
    log_epsilon = np.log(epsilon_init)
    log_epsilon_bar = 0.0
    H_bar = 0.0
    epsilon_history = np.zeros(n_samples)
    
    # Precompute some constants for efficiency
    log_target_accept = np.log(target_accept)
    
    # Main sampling loop
    for i in range(n_samples):
        # Store current state from all chains
        samples[:, i] = states
        
        # Store current epsilon
        epsilon_history[i] = epsilon
        
        # Precompute step size terms
        beta_eps = beta * epsilon
        beta_eps_half = beta_eps / 2
        
        # Compute centered ensembles for preconditioning
        centered2 = (states[group2] - np.mean(states[group2], axis=0)) / np.sqrt(n_chains_per_group)
        
        # First group update
        # Generate momentum - fully vectorized with correct dimensions
        p1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        # Store current state and energy
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        current_K1 = np.clip(0.5 * np.sum(p1**2, axis=1), 0, 1000)
        
        # Leapfrog integration with preconditioning
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum - vectorized
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        # Matrix multiplication for projection - fully vectorized
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Randomize number of leapfrog steps for second group too
        if randomize_steps:
            n_steps = np.random.randint(max(1, n_leapfrog//2), 2*n_leapfrog + 1)
        else:
            n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - vectorized matrix multiplication
            q1 += beta_eps * np.dot(p1_current, centered2)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - vectorized
                grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad1 = np.nan_to_num(grad1, nan=0.0)
                
                p1_current -= beta_eps * np.dot(grad1, centered2.T)
        
        # Final half-step for momentum - vectorized
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = np.clip(0.5 * np.sum(p1_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs1 = np.ones_like(dH1)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH1[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        
        # Update states - vectorized
        states[group1][accepts1] = q1[accepts1]
        accepts[group1] += accepts1

        # Second group update - vectorized the same way
        centered1 = (states[group1] - np.mean(states[group1], axis=0)) / np.sqrt(n_chains_per_group)
        
        p2 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        current_q2 = states[group2].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        current_K2 = np.clip(0.5 * np.sum(p2**2, axis=1), 0, 1000)
        
        q2 = current_q2.copy()
        p2_current = p2.copy()
        
        # Initial half-step for momentum - vectorized
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Full leapfrog steps
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - vectorized
            q2 += beta_eps * np.dot(p2_current, centered1)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - vectorized
                grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad2 = np.nan_to_num(grad2, nan=0.0)
                
                p2_current -= beta_eps * np.dot(grad2, centered1.T)
        
        # Final half-step for momentum - vectorized
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Compute proposed energy
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = np.clip(0.5 * np.sum(p2_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs2 = np.ones_like(dH2)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH2 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH2[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs2[exp_needed] = np.exp(-safe_dH)
        
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        
        # Update states
        states[group2][accepts2] = q2[accepts2]
        
        # Track acceptance for second chains
        accepts[group2] += accepts2
        
        # DUAL AVERAGING STEP SIZE ADAPTATION
        if i < adapt_steps:
            # Compute average acceptance probability for this iteration
            # We use the min of acceptance probabilities to be conservative
            all_accept_probs = np.concatenate([accept_probs1, accept_probs2])
            avg_accept_prob = np.mean(all_accept_probs)
            
            # Dual averaging update
            m = i + 1  # iteration number (1-indexed)
            
            # Update running average of log acceptance probability
            eta = 1.0 / (m + t0)
            H_bar = (1 - eta) * H_bar + eta * (target_accept - avg_accept_prob)
            
            # Update log step size
            log_epsilon = log_epsilon - eta * H_bar / gamma
            
            # Update running average of log step size
            m_kappa = m**(-kappa)
            log_epsilon_bar = m_kappa * log_epsilon + (1 - m_kappa) * log_epsilon_bar
            
            # Update step size
            epsilon = np.exp(log_epsilon)
            
            # Prevent step size from becoming too small or too large
            epsilon = np.clip(epsilon, 1e-8, 10.0)
            
        else:
            # After adaptation period, use the average step size
            epsilon = np.exp(log_epsilon_bar)
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / n_samples
    
    return samples, acceptance_rates, epsilon_history


def find_reasonable_epsilon_hwm(gradient_func, potential_func, initial_state, 
                               n_chains_per_group=5, n_leapfrog=10, beta=0.05, 
                               max_attempts=50):
    """
    Find a reasonable initial step size for Hamiltonian Walk Move sampler.
    
    This function mimics the HWM dynamics to find an appropriate step size,
    unlike standard HMC initialization.
    
    Parameters:
    -----------
    gradient_func : callable
        Function that computes gradients
    potential_func : callable
        Function that computes potential energy
    initial_state : array_like
        Initial state to test
    n_chains_per_group : int
        Number of chains per group (same as main sampler)
    n_leapfrog : int
        Number of leapfrog steps (same as main sampler)
    beta : float
        Preconditioning parameter (same as main sampler)
    max_attempts : int
        Maximum number of attempts to find reasonable epsilon
        
    Returns:
    --------
    epsilon : float
        Reasonable initial step size
    """
    epsilon = 0.1  # Start with a reasonable guess for HWM
    
    orig_dim = initial_state.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    attempts = 0
    target_accept_range = (0.3, 0.9)  # Broader range for initialization
    
    while attempts < max_attempts:
        # Create ensemble similar to main sampler
        states = np.tile(initial_state.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
        
        # Split into groups
        group1 = slice(0, n_chains_per_group)
        group2 = slice(n_chains_per_group, total_chains)
        
        # Test one HWM update step for group 1
        centered2 = (states[group2] - np.mean(states[group2], axis=0)) / np.sqrt(n_chains_per_group)
        
        # Generate momentum
        p1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        # Current state and energy
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        current_K1 = np.clip(0.5 * np.sum(p1**2, axis=1), 0, 1000)
        
        # Precompute step size terms
        beta_eps = beta * epsilon
        beta_eps_half = beta_eps / 2
        
        # HWM leapfrog integration
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        grad1 = np.nan_to_num(grad1, nan=0.0)
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Full leapfrog steps with HWM dynamics
        for step in range(n_leapfrog):
            # Position update with ensemble preconditioning
            q1 += beta_eps * np.dot(p1_current, centered2)
            
            if step < n_leapfrog - 1:
                # Momentum update
                grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                grad1 = np.nan_to_num(grad1, nan=0.0)
                p1_current -= beta_eps * np.dot(grad1, centered2.T)
        
        # Final half-step for momentum
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        grad1 = np.nan_to_num(grad1, nan=0.0)
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = np.clip(0.5 * np.sum(p1_current**2, axis=1), 0, 1000)
        
        # Compute acceptance probabilities
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Calculate acceptance probabilities
        accept_probs1 = np.ones_like(dH1)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            safe_dH = np.clip(dH1[exp_needed], None, 100)
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        # Average acceptance probability for this test
        avg_accept_prob = np.mean(accept_probs1)
        
        # Check if acceptance rate is in reasonable range
        if target_accept_range[0] <= avg_accept_prob <= target_accept_range[1]:
            break
            
        # Adjust epsilon
        if avg_accept_prob < target_accept_range[0]:
            epsilon *= 0.7  # Too low acceptance, decrease step size
        else:
            epsilon *= 1.3  # Too high acceptance, increase step size
            
        attempts += 1
        
        # Prevent infinite loops with extreme values
        epsilon = np.clip(epsilon, 1e-6, 1.0)
    
    if attempts >= max_attempts:
        print(f"Warning: Could not find optimal epsilon after {max_attempts} attempts. Using epsilon={epsilon:.6f}")
    
    return np.clip(epsilon, 1e-8, 1.0)


# Example usage with automatic epsilon initialization:
def hamiltonian_walk_move_auto(gradient_func, potential_func, initial, n_samples, 
                             n_chains_per_group=5, n_leapfrog=10, beta=0.05,
                             target_accept=0.8, **kwargs):
    """
    Hamiltonian Walk Move with automatic step size initialization and adaptation.
    """
    # Find reasonable initial step size using HWM dynamics
    epsilon_init = find_reasonable_epsilon_hwm(
        gradient_func, potential_func, initial, 
        n_chains_per_group=n_chains_per_group, 
        n_leapfrog=n_leapfrog, 
        beta=beta
    )
    print(f"Using initial step size: {epsilon_init:.6f}")
    
    # Run adaptive sampling
    return hamiltonian_walk_move_adaptive(
        gradient_func, potential_func, initial, n_samples,
        n_chains_per_group=n_chains_per_group, epsilon_init=epsilon_init,
        n_leapfrog=n_leapfrog, beta=beta, target_accept=target_accept, **kwargs
    )

import time

# Setup
dim = 50
n_samples = 10000
burn_in = 2000
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
initial = np.zeros(dim)

# Parameters
params = {"n_chains_per_group": max(dim,20), "n_leapfrog": 5, "beta": 1.0}

print(f"{dim}D Gaussian test: {params}")

# NumPy functions
def gradient_np(x):
    if x.ndim == 1: x = x.reshape(1, -1)
    return np.einsum('jk,ij->ik', precision, x - true_mean)

def potential_np(x):
    if x.ndim == 1: x = x.reshape(1, -1)
    centered = x - true_mean
    return 0.5 * np.einsum('ij,jk,ik->i', centered, precision, centered)



# Run NumPy
try:
    start = time.time()
    samples_np, acc_np, epsilon_history = hamiltonian_walk_move_auto(
gradient_np, potential_np, initial, n_samples=total_samples,randomize_steps=True, **params
)
    
    time_np = time.time() - start

    flat_np = samples_np[:, burn_in:, :].reshape(-1, dim)
    mean_np = np.mean(flat_np, axis=0)
    error_np = np.linalg.norm(mean_np - true_mean)
    print(epsilon_history)
    print(f"NumPy: accept={np.mean(acc_np):.3f}, mean error={error_np:.3f}, time={time_np:.1f}s")
except Exception as e:
    print(f"NumPy failed: {e}")

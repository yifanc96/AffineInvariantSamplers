import numpy as np
import matplotlib.pyplot as plt
import time
import os
import wandb


def autocorrelation_fft(x, max_lag=None):
    """
    Efficiently compute autocorrelation function using FFT.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    max_lag : int, optional
        Maximum lag to compute (default: len(x)//3)
        
    Returns:
    --------
    acf : array
        Autocorrelation function values
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 3, 20000)  # Cap at 20000 to prevent slow computation
    
    # Remove mean and normalize
    x_norm = x - np.mean(x)
    var = np.var(x_norm)
    x_norm = x_norm / np.sqrt(var)
    
    # Compute autocorrelation using FFT
    # Pad the signal with zeros to avoid circular correlation
    fft = np.fft.fft(x_norm, n=2*n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n]
    acf = acf.real / n  # Normalize
    
    return acf[:max_lag]

def integrated_autocorr_time(x, M=5, c=10):
    """
    Estimate the integrated autocorrelation time using a self-consistent window.
    Based on the algorithm described by Goodman and Weare.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    M : int, default=5
        Window size multiplier (typically 5-10)
    c : int, default=10
        Maximum lag cutoff for window determination
        
    Returns:
    --------
    tau : float
        Integrated autocorrelation time
    acf : array
        Autocorrelation function values
    ess : float
        Effective sample size
    """
    n = len(x)
    orig_x = x.copy()
    
    # Initial pairwise reduction if needed
    k = 0
    max_iterations = 10  # Prevent infinite loop
    
    while k < max_iterations:
        # Calculate autocorrelation function
        acf = autocorrelation_fft(x)
        
        # Calculate integrated autocorrelation time with self-consistent window
        tau = 1.0  # Initialize with the first term
        
        # Find the window size where window <= M * tau
        for window in range(1, len(acf)):
            # Update tau with this window
            tau_window = 1.0 + 2.0 * sum(acf[1:window+1])
            
            # Check window consistency: window <= M*tau
            if window <= M * tau_window:
                tau = tau_window
            else:
                break
        
        # If we have a robust estimate, we're done
        if n >= c * tau:
            # Scale tau back to the original time scale: tau_0 = 2^k * tau_k
            tau = tau * (2**k)
            break
            
        # If we don't have a robust estimate, perform pairwise reduction
        k += 1
        n_half = len(x) // 2
        x_new = np.zeros(n_half)
        for i in range(n_half):
            if 2*i + 1 < len(x):
                x_new[i] = 0.5 * (x[2*i] + x[2*i+1])
            else:
                x_new[i] = x[2*i]
        x = x_new
        n = len(x)
    
    # If we exited without a robust estimate, compute one final estimate
    if k >= max_iterations or n < c * tau:
        acf = autocorrelation_fft(orig_x)
        tau_reduced = 1.0 + 2.0 * sum(acf[1:min(len(acf), int(M)+1)])
        # Scale tau back to the original time scale
        tau = tau_reduced * (2**k)
    
    # Calculate effective sample size using original series length
    ess = len(orig_x) / tau
    
    return tau, acf, ess


def hamiltonian_walk_move_dual_avg(gradient_func, potential_func, initial, n_samples, 
                                  n_chains_per_group=5, epsilon_init=0.01, n_leapfrog=10, 
                                  beta=0.05, n_thin=1, target_accept=0.65, n_warmup=1000,
                                  gamma=0.05, t0=10, kappa=0.75):
    """
    Hamiltonian Walk Move sampler with dual averaging for automatic step size adaptation.
    
    Parameters:
    -----------
    gradient_func : callable
        Function that computes gradients of the log probability
    potential_func : callable  
        Function that computes the negative log probability (potential energy)
    initial : array_like
        Initial state
    n_samples : int
        Number of samples to collect (after warmup)
    n_chains_per_group : int
        Number of chains per group (default: 5)
    epsilon_init : float
        Initial step size (default: 0.01)
    n_leapfrog : int
        Number of leapfrog steps (default: 10)
    beta : float
        Preconditioning parameter (default: 0.05)
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
    target_accept : float
        Target acceptance rate for dual averaging (default: 0.65)
    n_warmup : int
        Number of warmup iterations for step size adaptation (default: 1000)
    gamma : float
        Dual averaging parameter controlling adaptation rate (default: 0.05)
    t0 : float
        Dual averaging parameter for numerical stability (default: 10)
    kappa : float
        Dual averaging parameter controlling decay (default: 0.75, should be in (0.5, 1])
    
    Returns:
    --------
    samples : ndarray
        Generated samples from all chains (after warmup)
    acceptance_rates : ndarray
        Final acceptance rates for each chain
    step_size_history : ndarray
        History of step sizes during adaptation
    """
    
    # Initialize
    orig_dim = initial.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1 = slice(0, n_chains_per_group)
    group2 = slice(n_chains_per_group, total_chains)
    
    # Dual averaging initialization
    log_epsilon = np.log(epsilon_init)
    log_epsilon_bar = 0.0
    H_bar = 0.0
    step_size_history = []
    
    # Calculate total iterations needed based on thinning factor
    total_sampling_iterations = n_samples * n_thin
    total_iterations = n_warmup + total_sampling_iterations
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts_warmup = np.zeros(total_chains)  # Track accepts during warmup
    accepts_sampling = np.zeros(total_chains)  # Track accepts during sampling
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Main sampling loop
    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_epsilon = epsilon_init if not is_warmup else np.exp(log_epsilon)
        
        # Store current state from all chains (only during sampling phase)
        if not is_warmup and (i - n_warmup) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # Precompute step size terms
        beta_eps = beta * current_epsilon
        beta_eps_half = beta_eps / 2
        
        # Compute centered ensembles for preconditioning
        centered2 = (states[group2] - np.mean(states[group2], axis=0)) / np.sqrt(n_chains_per_group)
        
        # First group update
        p1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        # Store current state and energy
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        current_K1 = np.clip(0.5 * np.sum(p1**2, axis=1), 0, 1000)
        
        # Leapfrog integration with preconditioning
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        grad1 = np.nan_to_num(grad1, nan=0.0)
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Full leapfrog steps
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
        
        # Metropolis acceptance with numerical stability
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        accept_probs1 = np.ones_like(dH1)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            safe_dH = np.clip(dH1[exp_needed], None, 100)
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        states[group1][accepts1] = q1[accepts1]
        
        # Track acceptances
        if is_warmup:
            accepts_warmup[group1] += accepts1
        else:
            accepts_sampling[group1] += accepts1
        
        # Second group update
        centered1 = (states[group1] - np.mean(states[group1], axis=0)) / np.sqrt(n_chains_per_group)
        
        p2 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        current_q2 = states[group2].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        current_K2 = np.clip(0.5 * np.sum(p2**2, axis=1), 0, 1000)
        
        q2 = current_q2.copy()
        p2_current = p2.copy()
        
        # Initial half-step for momentum
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        grad2 = np.nan_to_num(grad2, nan=0.0)
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Full leapfrog steps
        for step in range(n_leapfrog):
            q2 += beta_eps * np.dot(p2_current, centered1)
            
            if step < n_leapfrog - 1:
                grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                grad2 = np.nan_to_num(grad2, nan=0.0)
                p2_current -= beta_eps * np.dot(grad2, centered1.T)
        
        # Final half-step for momentum
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        grad2 = np.nan_to_num(grad2, nan=0.0)
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Compute proposed energy
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = np.clip(0.5 * np.sum(p2_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        accept_probs2 = np.ones_like(dH2)
        exp_needed = dH2 > 0
        if np.any(exp_needed):
            safe_dH = np.clip(dH2[exp_needed], None, 100)
            accept_probs2[exp_needed] = np.exp(-safe_dH)
        
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        states[group2][accepts2] = q2[accepts2]
        
        # Track acceptances
        if is_warmup:
            accepts_warmup[group2] += accepts2
        else:
            accepts_sampling[group2] += accepts2
        
        # Dual averaging step size adaptation during warmup
        if is_warmup:
            # Average acceptance probability across all chains in this iteration
            current_accept_rate = (np.sum(accepts1) + np.sum(accepts2)) / total_chains
            
            # Dual averaging update
            m = i + 1  # iteration number (1-indexed)
            eta = 1.0 / (m + t0)
            
            # Update log step size
            H_bar = (1 - eta) * H_bar + eta * (target_accept - current_accept_rate)
            
            # Compute log step size with shrinkage
            log_epsilon = np.log(epsilon_init) - np.sqrt(m) / gamma * H_bar
            
            # Update log_epsilon_bar for final step size
            eta_bar = m**(-kappa)
            log_epsilon_bar = (1 - eta_bar) * log_epsilon_bar + eta_bar * log_epsilon
            
            # Store step size history
            step_size_history.append(np.exp(log_epsilon))
        
        # After warmup, fix step size to the adapted value
        if i == n_warmup - 1:
            epsilon_init = np.exp(log_epsilon_bar)
            print(f"Warmup complete. Final adapted step size: {epsilon_init:.6f}")
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for sampling phase only
    acceptance_rates = accepts_sampling / total_sampling_iterations
    
    return samples, acceptance_rates, np.array(step_size_history)


def benchmark_samplers_allen_cahn(N=100, n_samples=10000, burn_in=1000, n_thin=1, save_dir=None):
    """
    Benchmark HWM sampler on the invariant measure of the Allen-Cahn SPDE.
    
    The Allen-Cahn SPDE has the invariant measure with density proportional to:
    exp(-∫[1/(2h) * (du/dx)² + V(u)] dx)
    
    where V(u) = (1 - u²)² is the double-well potential and h is the discretization step.
    """
    # Define discretization parameters
    h = 1.0 / N
    dim = N + 1  # Including boundary points
    
    # Define potential function V'(u) = -4u(1 - u^2)
    def V_prime(u):
        return -4 * u * (1 - u**2)
    
    # Define the gradient of the negative log density - vectorized
    def gradient(u):
        """Numerically stable vectorized gradient of the negative log density"""
        if u.ndim == 1:
            u = u.reshape(1, -1)
            
        grad = np.zeros_like(u)
        
        # Handle interior points (j=1 to j=N-1) with vectorization
        u_prev = u[:, :-2]  # u[j-1] for j=1...N-1
        u_curr = u[:, 1:-1]  # u[j] for j=1...N-1
        u_next = u[:, 2:]    # u[j+1] for j=1...N-1
        
        # Coupling term contribution: (2*u[j] - u[j-1] - u[j+1])/h
        coupling_term = (2 * u_curr - u_prev - u_next) / h
        
        # Potential term contribution
        avg_prev = (u_curr + u_prev) / 2
        avg_next = (u_curr + u_next) / 2
        
        v_prime_prev = -4 * avg_prev * (1 - avg_prev**2)
        v_prime_next = -4 * avg_next * (1 - avg_next**2)
        
        potential_term = h * (v_prime_prev + v_prime_next) / 4
        
        # Combine contributions for interior points
        grad[:, 1:-1] = coupling_term + potential_term
        
        # Handle boundary points
        u_first = u[:, 0]
        u_second = u[:, 1]
        grad[:, 0] = (u_first - u_second) / h + h * V_prime(u_first) / 4
        
        u_last = u[:, -1]
        u_second_last = u[:, -2]
        grad[:, -1] = (u_last - u_second_last) / h + h * V_prime(u_last) / 4
        
        return grad
    
    # Define the potential energy function - vectorized
    def potential(u):
        """Numerically stable vectorized negative log density (potential energy)"""
        if u.ndim == 1:
            u = u.reshape(1, -1)
            
        u_right = u[:, 1:]
        u_left = u[:, :-1]
        diffs = u_right - u_left
        
        coupling_term = np.sum(diffs**2, axis=1) / (2*h)
        
        u_avg = (u_right + u_left) / 2
        v_values = (1 - u_avg**2)**2
        potential_term = np.sum(h * v_values / 2, axis=1)
        
        total_potential = np.clip(coupling_term + potential_term, -1e10, 1e10)
            
        return total_potential
    
    # Function to compute path integral consistently
    def compute_path_integral(path):
        """Efficiently calculate the path integral using vectorized operations"""
        if path.ndim > 1:
            return np.array([compute_path_integral(p) for p in path])
            
        # For a single path
        left_points = path[:-1]
        right_points = path[1:]
        segment_areas = h * (left_points + right_points) / 2
        return np.sum(segment_areas)
    
    # Initial state - start near one of the stable states
    initial = np.ones(dim) * 0.8 + 0.1 * np.random.randn(dim)
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark
    samplers = {
        "Hamiltonian Walk Move": lambda: hamiltonian_walk_move_dual_avg(
            gradient_func=gradient, potential_func=potential, initial=initial, 
            n_samples=n_samples, n_warmup=burn_in, 
            n_chains_per_group=max(10, dim), epsilon_init=1/(dim**(1/4)), 
            n_leapfrog=3, beta=1.0,
            target_accept=0.65, n_thin=n_thin),
    }
    
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        
        try:
            samples, acceptance_rates, step_size_history = sampler_func()
            elapsed = time.time() - start_time
            
            # samples shape: (n_chains, n_samples, dim)
            post_burn_in_samples = samples
            
            # Flatten samples from all chains
            flat_samples = post_burn_in_samples.reshape(-1, dim)
            
            # Compute sample statistics
            sample_mean = np.mean(flat_samples, axis=0)
            
            # Calculate path integrals for all samples
            path_integrals = compute_path_integral(flat_samples)
            mean_path_integral = np.mean(path_integrals)
            path_integral_std = np.std(path_integrals)
            
            # Compute potential energies
            potential_energies = potential(flat_samples)
            mean_potential = np.mean(potential_energies)
            potential_var = np.var(potential_energies)
            
            # Check well mixing
            positive_well = np.mean(path_integrals > 0.5)
            negative_well = np.mean(path_integrals < -0.5)
            well_mixing = min(positive_well, negative_well)
            
            # Compute autocorrelation for path integral - following reference pattern
            # Average over chains first, then compute path integrals
            path_integrals_chain1 = compute_path_integral(np.mean(post_burn_in_samples, axis=0))
            acf = autocorrelation_fft(path_integrals_chain1)
            
            # Compute integrated autocorrelation time
            try:
                tau, _, ess = integrated_autocorr_time(path_integrals_chain1)
            except:
                tau, ess = np.nan, np.nan
                print("  Warning: Could not compute integrated autocorrelation time")
            
            # Measure fraction of time spent in positive well
            positive_fraction = np.mean(flat_samples > 0)
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            # Create dummy data in case of error
            flat_samples = np.zeros((10, dim))
            acceptance_rates = np.zeros(2)
            sample_mean = np.zeros(dim)
            mean_path_integral = np.nan
            path_integral_std = np.nan
            mean_potential = np.nan
            potential_var = np.nan
            well_mixing = np.nan
            positive_fraction = np.nan
            acf = np.zeros(100)
            tau, ess = np.nan, np.nan
            elapsed = time.time() - start_time
        
        # Store results
        results[name] = {
            "samples": flat_samples,
            "acceptance_rates": acceptance_rates,
            "sample_mean": sample_mean,
            "path_integral_mean": mean_path_integral,
            "path_integral_std": path_integral_std,
            "mean_potential": mean_potential,
            "potential_var": potential_var,
            "well_mixing": well_mixing,
            "positive_fraction": positive_fraction,
            "autocorrelation": acf,
            "tau": tau,
            "ess": ess,
            "time": elapsed
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Path integral mean: {mean_path_integral:.4f}")
        print(f"  Path integral std: {path_integral_std:.4f}")
        print(f"  Well mixing rate: {well_mixing:.4f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}" if np.isfinite(tau) else "  Integrated autocorrelation time: NaN")
        print(f"  Time: {elapsed:.2f} seconds")

        if save_dir:
            np.save(os.path.join(save_dir, f"samples_{name}_allen_cahn.npy"), samples)
            np.save(os.path.join(save_dir, f"acf_{name}_allen_cahn.npy"), acf)
            
    return results

# Main benchmark script
n_samples = 5000
burn_in = 10**3
array_dim = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_thin = 3

home = "/scratch/yc3400/AffineInvariant/"
timestamp = time.strftime("%Y%m%d-%H%M%S")
folder = f"benchmark_results_HWM_Allen-Cahn_5000samples_thin3_{timestamp}"

wandb_project = "AffineInvariant"
wandb_entity = 'yifanc96'
wandb_run = wandb.init(
    project=wandb_project,
    entity=wandb_entity,
    resume=None,
    id=None,
    name=folder
)
wandb.run.log_code(".")

print(f'n_sample{n_samples}, burn_in{burn_in}, n_thin{n_thin}')
    
for dim in array_dim:
    print(f"dim={dim}")
    # Create a timestamped directory for this run
    save_dir = os.path.join(home + folder, f"{dim}")
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Run benchmarks and save results
    results = benchmark_samplers_allen_cahn(
        N=dim, 
        n_samples=n_samples, 
        burn_in=burn_in, 
        n_thin=n_thin,
        save_dir=save_dir
    )
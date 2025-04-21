import numpy as np
import matplotlib.pyplot as plt
import time

from samplers import side_move, stretch_move, hmc, hamiltonian_walk_move, hamiltonian_side_move
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time


def path_integral(path, h=None):
    """Efficiently calculate the path integral using vectorized operations
    
    Parameters:
    -----------
    path : ndarray
        Path or array of paths to integrate
    h : float, optional
        Step size for integration. If None, calculated as 1/N where N = len(path)-1
    """
    if path.ndim > 1:
        # If given multiple paths, apply to each path and return array of results
        return np.array([path_integral(p, h) for p in path])
        
    # For a single path
    if h is None:
        h = 1.0 / (len(path) - 1)
        
    left_points = path[:-1]  # All points except the last
    right_points = path[1:]  # All points except the first
    
    # Trapezoidal rule: h * (f(a) + f(b))/2 for each segment
    segment_areas = h * (left_points + right_points) / 2
    
    # Sum all segment areas
    return np.sum(segment_areas)

def benchmark_samplers_spde(N=100, n_samples=10000, burn_in=1000):
    """
    Benchmark different MCMC samplers on the invariant measure of the Allen-Cahn SPDE.
    
    Parameters:
    -----------
    N : int
        Number of discretization points (N+1 total points including boundaries)
    n_samples : int
        Number of samples to generate after burn-in
    burn_in : int
        Number of initial samples to discard as burn-in
    """
    # Define discretization parameters
    h = 1.0 / N
    dim = N + 1  # Including boundary points
    
    # Define potential function V(u) = (1 - u^2)^2
    def V(u):
        return (1 - u**2)**2
    
    # Define potential gradient V'(u) = -4u(1 - u^2)
    def V_prime(u):
        return -4 * u * (1 - u**2)
    
    # Define the discretized log density function based on equation (22) - vectorized
    def log_density(u):
        """Numerically stable vectorized log density of the discretized SPDE distribution"""
        if u.ndim == 1:
            u = u.reshape(1, -1)
            
        batch_size = u.shape[0]
        
        # Compute differences for all paths at once
        # For each path, calculate u[j+1] - u[j] for all j
        u_right = u[:, 1:]  # All elements except the first
        u_left = u[:, :-1]  # All elements except the last
        diffs = u_right - u_left
        
        # Calculate the first term (coupling neighboring values)
        # Sum (diff^2)/(2h) for each path
        coupling_term = np.sum(diffs**2, axis=1) / (2*h)
        
        # Calculate the second term (double well potential)
        # Average of adjacent points for trapezoidal rule
        u_avg = (u_right + u_left) / 2
        # Calculate V(u_avg) for all average points
        v_values = (1 - u_avg**2)**2
        # Sum h*V(u_avg)/2 for each path
        potential_term = np.sum(h * v_values / 2, axis=1)
        
        # Combine terms with careful handling of extreme values
        total_potential = coupling_term + potential_term
        log_dens = -np.clip(total_potential, -1e10, 1e10)
            
        return log_dens
    
    # Define the gradient of the negative log density - vectorized
    def gradient(u):
        """Numerically stable vectorized gradient of the negative log density"""
        if u.ndim == 1:
            u = u.reshape(1, -1)
            
        batch_size = u.shape[0]
        grad = np.zeros_like(u)
        
        # Handle interior points (j=1 to j=N-1) with vectorization
        # For each path, we'll compute the gradient at all interior points at once
        
        # Use slicing to get neighboring values for all interior points
        u_prev = u[:, :-2]  # u[j-1] for j=1...N-1
        u_curr = u[:, 1:-1]  # u[j] for j=1...N-1
        u_next = u[:, 2:]    # u[j+1] for j=1...N-1
        
        # Coupling term contribution: (2*u[j] - u[j-1] - u[j+1])/h
        coupling_term = (2 * u_curr - u_prev - u_next) / h
        
        # Potential term contribution
        # Calculate averages for trapezoidal rule
        avg_prev = (u_curr + u_prev) / 2  # (u[j] + u[j-1])/2
        avg_next = (u_curr + u_next) / 2  # (u[j] + u[j+1])/2
        
        # Calculate V'(avg) for all averages
        v_prime_prev = -4 * avg_prev * (1 - avg_prev**2)  # V'((u[j] + u[j-1])/2)
        v_prime_next = -4 * avg_next * (1 - avg_next**2)  # V'((u[j] + u[j+1])/2)
        
        # Contribution from potential terms
        potential_term = h * (v_prime_prev + v_prime_next) / 4
        
        # Combine contributions for interior points
        grad[:, 1:-1] = coupling_term + potential_term
        
        # Handle boundary points (j=0 and j=N) separately - still using vectorization
        
        # First point (j=0)
        u_first = u[:, 0]
        u_second = u[:, 1]
        grad[:, 0] = (u_first - u_second) / h + h * V_prime(u_first) / 4
        
        # Last point (j=N)
        u_last = u[:, -1]
        u_second_last = u[:, -2]
        grad[:, -1] = (u_last - u_second_last) / h + h * V_prime(u_last) / 4
        
        return grad
    
    # Define the potential energy function - vectorized
    def potential(u):
        """Numerically stable vectorized negative log density (potential energy)"""
        if u.ndim == 1:
            u = u.reshape(1, -1)
            
        # Compute differences for all paths at once
        u_right = u[:, 1:]  # All elements except the first
        u_left = u[:, :-1]  # All elements except the last
        diffs = u_right - u_left
        
        # Calculate the first term (coupling neighboring values)
        coupling_term = np.sum(diffs**2, axis=1) / (2*h)
        
        # Calculate the second term (double well potential)
        u_avg = (u_right + u_left) / 2
        v_values = (1 - u_avg**2)**2
        potential_term = np.sum(h * v_values / 2, axis=1)
        
        # Combine terms with careful handling of extreme values
        total_potential = np.clip(coupling_term + potential_term, -1e10, 1e10)
            
        return total_potential
    
    # Initial state with careful initialization
    # Start with a random path that's slightly biased toward one of the wells
    np.random.seed(42)  # For reproducibility
    initial = np.random.randn(dim) * 0.1
    if np.random.rand() < 0.5:
        initial = initial + 1  # Start near the +1 well
    else:
        initial = initial - 1  # Start near the -1 well
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - parameters tuned for the SPDE
    samplers = {
        "Side Move": lambda: side_move(log_density, initial, total_samples, n_walkers=dim*2, gamma=1.6869),
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=dim*2, a=1.0+2.1515/np.sqrt(dim)),
        "HMC n=10": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.1, L=10, n_chains=1),
        "HMC n=2": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.5, L=2, n_chains=1),
        "Hamiltonian Walk Move n=10": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0),
        "Hamiltonian Walk Move n=2": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0),
        "Hamitonian Side Move n=10": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples,
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0),
        "Hamitonian Side Move n=2": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples, 
                                                                              n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0),
    }
    
    # Function to evaluate from equation (23) - vectorized trapezoidal rule approximation of ∫u(x)dx
    def compute_path_integral(path):
        """Efficiently calculate the path integral using vectorized operations"""
        if path.ndim > 1:
            # If given multiple paths, apply to each path and return array of results
            return np.array([compute_path_integral(p) for p in path])
            
        # For a single path
        left_points = path[:-1]  # All points except the last
        right_points = path[1:]  # All points except the first
        
        # Trapezoidal rule: h * (f(a) + f(b))/2 for each segment
        segment_areas = h * (left_points + right_points) / 2
        
        # Sum all segment areas
        return np.sum(segment_areas)
    
    # Benchmark each sampler with careful error handling
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        
        try:
            samples, acceptance_rates = sampler_func()
            
            # Apply burn-in: discard the first burn_in samples
            post_burn_in_samples = samples[:, burn_in:, :]
            
            # Flatten samples from all chains
            n_chains = post_burn_in_samples.shape[0]
            flat_samples = post_burn_in_samples.reshape(-1, dim)
            
            # Calculate path integrals for all samples at once
            path_integrals = compute_path_integral(flat_samples)
            
            # Calculate bimodality statistics
            sample_mean = np.mean(flat_samples, axis=0)
            path_integral_mean = np.mean(path_integrals)
            path_integral_std = np.std(path_integrals)
            
            # Check if samples go between the two wells
            positive_well = np.mean(path_integrals > 0.5)
            negative_well = np.mean(path_integrals < -0.5)
            well_mixing = min(positive_well, negative_well)
            
            # Compute autocorrelation for path integral
            # average over chains
            path_integrals_chain1 = compute_path_integral(np.mean(post_burn_in_samples,axis=0))
            acf = autocorrelation_fft(path_integrals_chain1)
            
            # Compute integrated autocorrelation time for path integral
            try:
                tau, _, ess = integrated_autocorr_time(path_integrals_chain1)
            except:
                tau, ess = np.nan, np.nan
                print("  Warning: Could not compute integrated autocorrelation time")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            # Create dummy data in case of error
            flat_samples = np.zeros((10, dim))
            acceptance_rates = np.zeros(n_chains)
            sample_mean = np.zeros(dim)
            path_integral_mean = np.nan
            path_integral_std = np.nan
            well_mixing = np.nan
            acf = np.zeros(100)
            tau, ess = np.nan, np.nan
        
        elapsed = time.time() - start_time
        
        # Store results
        results[name] = {
            "samples": flat_samples,
            "acceptance_rates": acceptance_rates,
            "sample_mean": sample_mean,
            "path_integral_mean": path_integral_mean,
            "path_integral_std": path_integral_std,
            "well_mixing": well_mixing,
            "autocorrelation": acf,
            "tau": tau,
            "ess": ess,
            "time": elapsed
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Path integral mean: {path_integral_mean:.4f}")
        print(f"  Path integral std: {path_integral_std:.4f}")
        print(f"  Well mixing rate: {well_mixing:.4f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}" if np.isfinite(tau) else "  Integrated autocorrelation time: NaN")
        # print(f"  Effective sample size: {ess:.2f}" if np.isfinite(ess) else "  Effective sample size: NaN")
        # print(f"  ESS/sec: {ess/elapsed:.2f}" if np.isfinite(ess) else "  ESS/sec: NaN")
        print(f"  Time: {elapsed:.2f} seconds")
    
    return results, N, h

def plot_spde_results(results, N=100, h=0.01):
    """Plot comparison of sampler results for SPDE problem"""
    samplers = list(results.keys())
    
    # 1. Plot autocorrelation functions
    plt.figure(figsize=(12, 6))
    for i, name in enumerate(samplers):
        acf = results[name]["autocorrelation"]
        max_lag = min(300, len(acf))
        plt.plot(np.arange(max_lag), acf[:max_lag], label=name)
    
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Functions (Path Integral)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spde_autocorrelation.png")
    
    # 2. Plot example paths from each sampler
    plt.figure(figsize=(15, 12))
    n_rows = len(samplers)
    
    # Create x-coordinates for plotting
    x_coords = np.linspace(0, 1, N+1)
    
    for i, name in enumerate(samplers):
        plt.subplot(n_rows, 1, i+1)
        
        samples = results[name]["samples"]
        
        # Plot a few random sample paths
        n_paths = min(10, len(samples))
        path_indices = np.random.choice(len(samples), n_paths, replace=False)
        
        for idx in path_indices:
            plt.plot(x_coords, samples[idx], alpha=0.5, linewidth=1)
            
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Upper Well')
        plt.axhline(y=-1, color='b', linestyle='--', alpha=0.5, label='Lower Well')
        
        plt.ylim(-2, 2)
        plt.title(f"{name} - Example Paths")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig("spde_example_paths.png")
    
    # 3. Plot histogram of path integrals
    plt.figure(figsize=(12, 6))
    
    for i, name in enumerate(samplers):
        samples = results[name]["samples"]
        
    # Calculate path integrals for samples
    for name in samplers:
        samples = results[name]["samples"]
        
        # Calculate path integrals using trapezoidal rule
        path_integrals = np.zeros(len(samples))
        for i, path in enumerate(samples):
            # Compute integral for this path using trapezoidal rule
            left_points = path[:-1]  # All points except the last
            right_points = path[1:]  # All points except the first
            
            # Trapezoidal rule: h * (f(a) + f(b))/2 for each segment
            segment_areas = h * (left_points + right_points) / 2
            path_integrals[i] = np.sum(segment_areas)
            
        # Plot histogram
        plt.hist(path_integrals, bins=50, alpha=0.6, label=name, density=True)
    
    plt.xlabel("Path Integral (∫u(x)dx)")
    plt.ylabel("Density")
    plt.title("Distribution of Path Integrals")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spde_path_integral_distribution.png")
    
        
    print("Plot generation complete.")
    return

# Run the benchmark for the SPDE problem
# Note: You need to have the sampler functions (side_move, stretch_move, etc.) defined elsewhere
results, N, h = benchmark_samplers_spde(N=50, n_samples=100000, burn_in=20000)

# Plot the results
# plot_spde_results(results, N, h)

print("SPDE benchmark complete. Check the output directory for plots.")
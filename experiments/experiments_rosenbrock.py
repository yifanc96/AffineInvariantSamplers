import numpy as np
import matplotlib.pyplot as plt
import time

from samplers import side_move, stretch_move, hmc, hamiltonian_walk_move, hamiltonian_side_move
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time


def benchmark_samplers_rosenbrock(dim=2, n_samples=10000, burn_in=1000, a=1.0, b=100.0):
    """
    Benchmark different MCMC samplers on the Rosenbrock distribution.
    
    Parameters:
    -----------
    dim : int
        Dimension of the Rosenbrock distribution (must be even)
    n_samples : int
        Number of samples to generate after burn-in
    burn_in : int
        Number of initial samples to discard as burn-in
    a : float
        Rosenbrock parameter (typically 1.0)
    b : float
        Rosenbrock parameter (typically 100.0)
    """
    # Ensure dimension is even for Rosenbrock
    if dim % 2 != 0:
        dim += 1
        print(f"Adjusted dimension to {dim} to ensure it's even for Rosenbrock")
    
    # Numerically stable implementation of Rosenbrock log-density and gradient
    def log_density(x):
        """Numerically stable log density of the Rosenbrock distribution"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Separate odd and even indices
        x_even = x[:, ::2]  # x₁, x₃, x₅, ...
        x_odd = x[:, 1::2]  # x₂, x₄, x₆, ...
        
        # Compute difference term with numerical stability
        # Use intermediate clipping to prevent extreme values
        x_even_clipped = np.clip(x_even, -1e8, 1e8)
        x_even_squared = x_even_clipped**2
        diff_term = x_odd - x_even_squared
        
        # Calculate the two terms with stable operations
        # Use clipping to prevent potential overflows
        term1 = b * np.sum(np.clip(diff_term**2, 0, 1e12), axis=1)
        term2 = np.sum(np.clip((x_even - a)**2, 0, 1e12), axis=1)
        
        # Combine terms with careful clipping
        result = -np.clip(term1 + term2, -1e12, 1e12)
        
        return result
    
    def gradient(x):
        """Correct gradient of the negative log density for Rosenbrock distribution"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Initialize gradient array
        grad = np.zeros_like(x)
        
        # Separate odd and even indices
        x_even = x[:, ::2]  # x₁, x₃, x₅, ...
        x_odd = x[:, 1::2]  # x₂, x₄, x₆, ...
        
        # Compute intermediate terms with numerical safeguards
        x_even_clipped = np.clip(x_even, -1e8, 1e8)
        diff_term = x_odd - x_even_clipped**2
        diff_term_clipped = np.clip(diff_term, -1e8, 1e8)
        
        # For point [-0.5, 0.5], we need to fix the signs in the even index gradients
        # Compute gradients for even indices (x₁, x₃, ...)
        term1 = 4 * b * x_even_clipped * diff_term_clipped
        term2 = -2 * (x_even - a)
        grad_even = np.clip((term1 + term2), -1e10, 1e10)
        
        # Compute gradients for odd indices (x₂, x₄, ...)
        grad_odd = -2 * b * diff_term_clipped
        grad_odd = np.clip(grad_odd, -1e10, 1e10)
        
        # Place gradients back in the right positions
        grad[:, ::2] = grad_even
        grad[:, 1::2] = grad_odd
        
        return -grad
    
    def potential(x):
        """Numerically stable negative log density (potential energy)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Use the log density with careful bounds
        log_dens = log_density(x)
        return -np.clip(log_dens, -1e10, 1e10)
    
    # Initial state - make more robust by using a reasonable starting point
    initial = np.zeros(dim)
    initial[::2] = 1.0  # Start away from mode but not too extreme
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - adjust parameters for Rosenbrock
    samplers = {
        "Side Move": lambda: side_move(log_density, initial, total_samples, n_walkers=dim*10, gamma=1.687),
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=dim*10, a=1.0+2.151/np.sqrt(dim)),
        "HMC n=10": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.1, L=10, n_chains=1),
        "HMC n=2": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.5, L=2, n_chains=1),
        "Hamiltonian Walk Move n=10": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim*5, epsilon=0.1, n_leapfrog=10, beta=1.0),
        "Hamiltonian Walk Move n=2": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim*5, epsilon=0.5, n_leapfrog=2, beta=1.0),
        "Hamitonian Side Move n=10": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples,
                                                        n_chains_per_group=dim*5, epsilon=0.1, n_leapfrog=10, beta=1.0),
        "Hamitonian Side Move n=2": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples, 
                                                                              n_chains_per_group=dim*5, epsilon=0.5, n_leapfrog=2, beta=1.0),
    }
    
    # Benchmark each sampler with careful error handling
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        
        try:
            samples, acceptance_rates = sampler_func()
            
            # Apply burn-in: discard the first burn_in samples
            post_burn_in_samples = samples[:, burn_in:, :]
            
            # Flatten samples from all chains
            flat_samples = post_burn_in_samples.reshape(-1, dim)

            mean_x1 = np.mean(flat_samples[:, 0])
            mean_x2 = np.mean(flat_samples[:, 1])


            # Compute autocorrelation for first dimension
    
            if np.all(np.isfinite(samples[0, :, 0])):
                acf = autocorrelation_fft(np.mean(post_burn_in_samples[:, :, 0],axis=0))
            else:
                acf = np.zeros(100)  # Fallback if there are non-finite values
                print("  Warning: Non-finite values in samples, cannot compute autocorrelation")
            
            # Compute integrated autocorrelation time for first dimension
            try:
                tau, _, ess = integrated_autocorr_time(np.mean(post_burn_in_samples[:, :, 0],axis=0))
            except:
                tau, ess = np.nan, np.nan
                print("  Warning: Could not compute integrated autocorrelation time")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            # Create dummy data in case of error
            flat_samples = np.zeros((10, dim))
            acceptance_rates = np.zeros(dim)
            mean_x1 = np.nan
            mean_x2 = np.nan
            acf = np.zeros(100)
            tau, ess = np.nan, np.nan
        
        elapsed = time.time() - start_time
        
        # Store results
        results[name] = {
            "samples": flat_samples,
            "acceptance_rates": acceptance_rates,
            "mean_x1": mean_x1,
            "mean_x2": mean_x2,
            "autocorrelation": acf,
            "tau": tau,
            "ess": ess,
            "time": elapsed
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Mean x1, x2: {mean_x1:.6f}, {mean_x2:.6f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}" if np.isfinite(tau) else "  Integrated autocorrelation time: NaN")
        # print(f"  Effective sample size: {ess:.2f}" if np.isfinite(ess) else "  Effective sample size: NaN")
        # print(f"  ESS/sec: {ess/elapsed:.2f}" if np.isfinite(ess) else "  ESS/sec: NaN")
        print(f"  Time: {elapsed:.2f} seconds")
    
    return results, log_density

def plot_rosenbrock_results(results, log_density_func, dim=2, a=1.0, b=100.0):
    """Plot comparison of sampler results for Rosenbrock distribution"""
    samplers = list(results.keys())
    
    # 1. Plot autocorrelation functions
    plt.figure(figsize=(12, 6))
    for i, name in enumerate(samplers):
        acf = results[name]["autocorrelation"]
        max_lag = min(300, len(acf))
        plt.plot(np.arange(max_lag), acf[:max_lag], label=name)
    
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Functions (First Dimension)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rosenbrock_autocorrelation.png")
    
    # 2. Plot 2D scatter for first pair of dimensions
    plt.figure(figsize=(15, 12))
    
    # Create a grid of x-y values for contour plot
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 5, 200)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Rosenbrock density over the grid with numerical safety
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.zeros(dim)
            point[0] = X[j, i]
            point[1] = Y[j, i]
            # Fill remaining dimensions with optimal values if dim > 2
            for k in range(2, dim, 2):
                point[k] = a
                point[k+1] = a**2
            
            # Compute log density and apply exponential with clipping
            log_dens = log_density_func(point)[0]
            # Clip to prevent overflow/underflow
            log_dens_clipped = np.clip(log_dens, -30, 10)
            Z[j, i] = np.exp(log_dens_clipped)
    
    # Plot contours of the true Rosenbrock density
    plt.contour(X, Y, Z, levels=20, colors='k', alpha=0.5, linestyles='--')
    
    # Plot samples from each sampler
    colors = plt.cm.tab10(np.linspace(0, 1, len(samplers)))
    
    for i, name in enumerate(samplers):
        samples = results[name]["samples"]
        # Subsample for clarity if needed
        if len(samples) > 1000:
            idx = np.random.choice(len(samples), 1000, replace=False)
            plot_samples = samples[idx]
        else:
            plot_samples = samples
        
        # Clip extreme values for better visualization
        plot_samples_clipped = np.clip(plot_samples, -5, 5)
        
        plt.scatter(plot_samples_clipped[:, 0], plot_samples_clipped[:, 1], 
                   color=colors[i], alpha=0.5, label=name)
    
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Rosenbrock Distribution: First 2 Dimensions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rosenbrock_samples.png")
    

# Run the benchmark for Rosenbrock distribution
results, log_density_func = benchmark_samplers_rosenbrock(dim=2, n_samples=1000000, burn_in=200000, a=1.0, b=100.0)

# Plot the results
plot_rosenbrock_results(results, log_density_func, dim=2, a=1.0, b=100.0)

print("Rosenbrock benchmark complete. Check the output directory for plots.")
import numpy as np
import matplotlib.pyplot as plt

x = 100          # Prix initial de l'actif
r = 0.03         # Taux sans risque
sigma = 0.2      # Volatilité
T = 1            # Temps jusqu'à l'échéance (années)
K1 = 100         # Prix d'exercice 1
K2 = 110         # Prix d'exercice 2
N = 51000        # Nombre de simulations Monte Carlo
M = 50           # Nombre de pas de temps
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Valeurs d'epsilon pour le calcul de Delta
np.random.seed(42)


def geo_brownian_ipp(M, N, T, x, r, sigma):
    dt = T / M
    S_path = np.zeros((M+1, N))
    S_path[0] = x
    B_increments = np.zeros((M, N))  
    for t in range(1, M+1):
        Z = np.random.standard_normal(N)
        S_path[t] = S_path[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        B_increments[t-1] = Z * np.sqrt(dt)
    B_total = np.sum(B_increments, axis=0)  
    return S_path, B_total, B_increments

def simulate_asian_call(T, M, r, sigma, x, N, K1):
    S_path, _, _ = geo_brownian_ipp(M, N, T, x, r, sigma)
    S_int = (T/M) * S_path.sum(axis=0)
    payoff = np.maximum(S_int - K1, 0)
    call_price = np.exp(-r*T) * payoff.mean()
    return call_price

def delta_asian_call(T, M, r, sigma, x, N, K):
    S_path, B_total, B_increments = geo_brownian_ipp(M, N, T, x, r, sigma)
    time_points = np.linspace(0, T, M+1)  
    
    S_int = (T/M) * S_path.sum(axis=0)
    integral_sX = np.sum(S_path * time_points.reshape(-1, 1), axis=0) * (T/M)
    integral_s2X = np.sum(S_path * (time_points**2).reshape(-1, 1), axis=0) * (T/M)
    
    # weight Π1 et Π2
    Pi1 = (S_int / integral_sX) * (B_total / sigma + integral_s2X / (x * integral_sX))
    Pi2 = (2 / sigma**2) * ((S_path[-1] - x) / S_int - r) + 1
    
    # Payoff Call
    payoff = np.maximum(S_int - K, 0)
    
    # Delta
    delta_Pi1 = (np.exp(-r * T) / x * np.mean(payoff * Pi1))
    delta_Pi2 = (np.exp(-r * T) / x * np.mean(payoff * Pi2))
    
    # variance et intervalle de confiance
    samples_Pi1 = np.exp(-r * T) * payoff * Pi1 / x
    samples_Pi2 = np.exp(-r * T) * payoff * Pi2 / x
    var_Pi1 = np.var(samples_Pi1) / N
    var_Pi2 = np.var(samples_Pi2) / N
    ci_Pi1 = (delta_Pi1 - 1.96 * np.sqrt(var_Pi1), delta_Pi1 + 1.96 * np.sqrt(var_Pi1))
    ci_Pi2 = (delta_Pi2 - 1.96 * np.sqrt(var_Pi2), delta_Pi2 + 1.96 * np.sqrt(var_Pi2))
    
    return delta_Pi1, delta_Pi2, var_Pi1, var_Pi2, ci_Pi1, ci_Pi2

if __name__ == "__main__":
    
    call_price = simulate_asian_call(T, M, r, sigma, x, N, K1)
    delta_Pi1, delta_Pi2, var_Pi1, var_Pi2, ci_Pi1, ci_Pi2 = delta_asian_call(T, M, r, sigma, x, N, K1)
    
    print("Asian Call Option Results:")
    print(f"Price = {call_price:.4f}")
    print(f"Delta (Π1 Method) = {delta_Pi1:.6f}, Variance = {var_Pi1:.6f}")
    print(f"95% CI (Π1): [{ci_Pi1[0]:.6f}, {ci_Pi1[1]:.6f}]")
    print(f"Delta (Π2 Method) = {delta_Pi2:.6f}, Variance = {var_Pi2:.6f}")
    print(f"95% CI (Π2): [{ci_Pi2[0]:.6f}, {ci_Pi2[1]:.6f}]")
    
    


def simulate_asian_barriere(T, M, r, sigma, x, N, K1, K2):
    S_path,_,_ = geo_brownian_ipp(M, N ,T,x,r,sigma) 
    S_int = (T/M) * S_path.sum(axis=0)
    condition = (S_int > K1) & (S_int < K2)
    barriere_price = np.exp(-r*T)* condition.mean()
    return barriere_price


def delta_asian_barrier(T, M, r, sigma, x, N, K1,K2):
    S_path, B_total, B_increments = geo_brownian_ipp(M, N, T, x, r, sigma)
    time_points = np.linspace(0, T, M+1)  
    
    S_int = (T/M) * S_path.sum(axis=0)
    integral_sX = np.sum(S_path * time_points.reshape(-1, 1), axis=0) * (T/M)
    integral_s2X = np.sum(S_path * (time_points**2).reshape(-1, 1), axis=0) * (T/M)
    
    # weight Π1 et Π2
    Pi1 = (S_int / integral_sX) * (B_total / sigma + integral_s2X / (x * integral_sX))
    Pi2 = (2 / sigma**2) * ((S_path[-1] - x) / S_int - r) + 1
    
    payoff_b = ((S_int > K1) & (S_int < K2)).astype(float) 
    
    # Delta
    delta_Pi1 = (np.exp(-r * T) / x) * np.mean(payoff_b * Pi1)
    delta_Pi2 = (np.exp(-r * T) / x) * np.mean(payoff_b * Pi2)
    
    # variance et intervalle de confiance
    samples_Pi1 = np.exp(-r * T) * payoff_b * Pi1 / x
    samples_Pi2 = np.exp(-r * T) * payoff_b * Pi2 / x
    var_Pi1 = np.var(samples_Pi1) / N
    var_Pi2 = np.var(samples_Pi2) / N
    ci_Pi1 = (delta_Pi1 - 1.96 * np.sqrt(var_Pi1), delta_Pi1 + 1.96 * np.sqrt(var_Pi1))
    ci_Pi2 = (delta_Pi2 - 1.96 * np.sqrt(var_Pi2), delta_Pi2 + 1.96 * np.sqrt(var_Pi2))
    
    return delta_Pi1, delta_Pi2, var_Pi1, var_Pi2, ci_Pi1, ci_Pi2
    
    

if __name__ == "__main__":
    
    barrier_price = simulate_asian_barriere(T, M, r, sigma, x, N, K1, K2)
    delta_Pi1_barrier, delta_Pi2_barrier, var_Pi1_barrier, var_Pi2_barrier, ci_Pi1_barrier, ci_Pi2_barrier = delta_asian_barrier(T, M, r, sigma, x, N, K1, K2)
    
    print("\nAsian Barrier Option Results:")
    print(f"Price = {barrier_price:.4f}")
    print(f"Delta (Π1 Method) = {delta_Pi1_barrier:.6f}, Variance = {var_Pi1_barrier:.6f}")
    print(f"95% CI (Π1): [{ci_Pi1_barrier[0]:.6f}, {ci_Pi1_barrier[1]:.6f}]")
    print(f"Delta (Π2 Method) = {delta_Pi2_barrier:.6f}, Variance = {var_Pi2_barrier:.6f}")
    print(f"95% CI (Π2): [{ci_Pi2_barrier[0]:.6f}, {ci_Pi2_barrier[1]:.6f}]")

   
# question d    
# Define a range of N values to test
N_values = np.arange(1000, 51000, 5000)

# Initialize arrays to store variances
var_Pi1_call = np.zeros(len(N_values))
var_Pi2_call = np.zeros(len(N_values))
var_Pi1_barrier = np.zeros(len(N_values))
var_Pi2_barrier = np.zeros(len(N_values))

# Calculate variances for each N
for i, N in enumerate(N_values):
    _, _, var1, var2, _, _ = delta_asian_call(T, M, r, sigma, x, N, K1)
    var_Pi1_call[i] = var1
    var_Pi2_call[i] = var2
    
    _, _, var1_b, var2_b, _, _ = delta_asian_barrier(T, M, r, sigma, x, N, K1, K2)
    var_Pi1_barrier[i] = var1_b
    var_Pi2_barrier[i] = var2_b

# Plotting
plt.figure(figsize=(12, 6))

# Asian Call Option
plt.subplot(1, 2, 1)
plt.plot(N_values, var_Pi1_call, label='Π1 Method', marker='o')
plt.plot(N_values, var_Pi2_call, label='Π2 Method', marker='s')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Variance')
plt.title('Asian Call Option Delta Variance')
plt.legend()
plt.grid(True)

# Asian Barrier Option
plt.subplot(1, 2, 2)
plt.plot(N_values, var_Pi1_barrier, label='Π1 Method', marker='o')
plt.plot(N_values, var_Pi2_barrier, label='Π2 Method', marker='s')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Variance')
plt.title('Asian Barrier Option Delta Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()  































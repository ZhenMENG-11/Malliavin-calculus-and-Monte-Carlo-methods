import numpy as np
import matplotlib.pyplot as plt
# Paramètres initiaux
x = 100          # Prix initial de l'actif
r = 0.03         # Taux sans risque
sigma = 0.2      # Volatilité
T = 1            # Temps jusqu'à l'échéance (années)
K1 = 100         # Prix d'exercice 1
K2 = 110         # Prix d'exercice 2
N = 51000        # Nombre de simulations Monte Carlo
M = 50           # Nombre de pas de temps
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Valeurs d'epsilon pour le calcul de Delta
np.random.seed(42)  # Graine aléatoire pour la reproductibilité


#Génère des chemins de mouvement brownien géométrique

def geo_brownian_common(M, N, T, x, r, sigma, Z):
   
    dt = T / M
    S_path = np.zeros((M + 1, N))
    S_path[0] = x
    for t in range(1, M + 1):
        S_path[t] = S_path[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t - 1])
    return S_path


def prix_call_asiatique(S_path, T, M, K1):
    S_int = (T / M) * S_path.sum(axis=0)
    return np.maximum(S_int - K1, 0)  


def prix_barrière(S_path, T, M, K1, K2):
    S_int = (T / M) * S_path.sum(axis=0)
    return ((S_int > K1) & (S_int < K2)).astype(float)  



def delta(M, N, x, sigma, T, r, epsilon, option_type):  
    Z = np.random.standard_normal((M, N))
    
    # trajectoire
    S_plus = geo_brownian_common(M, N, T, x + epsilon, r, sigma, Z)
    S_moins = geo_brownian_common(M, N, T, x - epsilon, r, sigma, Z)
    
    
    if option_type == 'call':
        payoff_plus = prix_call_asiatique(S_plus, T, M, K1)
        payoff_moins = prix_call_asiatique(S_moins, T, M, K1)
    elif option_type == 'barrier':
        payoff_plus = prix_barrière(S_plus, T, M, K1, K2)
        payoff_moins = prix_barrière(S_moins, T, M, K1, K2)
    
    # payoff
    actualisation = np.exp(-r * T)
    V_plus = actualisation * payoff_plus.mean()
    V_moins = actualisation * payoff_moins.mean()
    
    #Delta
    delta_value = (V_plus - V_moins) / (2 * epsilon)
    
    # variance et intervalle de confiance
    actualisation_squared = np.exp(-2 * r * T)
    covariance = np.cov(payoff_plus, payoff_moins)[0, 1]
    var_plus = payoff_plus.var(ddof=1)
    var_moins = payoff_moins.var(ddof=1)
    
   
    if option_type == 'call':
        denominator = 4 * epsilon**2 * N
    elif option_type == 'barrier':
        denominator = 4 * epsilon * N  
    
    var_delta = (actualisation_squared * (var_plus + var_moins - 2 * covariance)) / denominator
    erreur_type = np.sqrt(var_delta) if var_delta >= 0 else 0.0
    ci_bas = delta_value - 1.96 * erreur_type
    ci_haut = delta_value + 1.96 * erreur_type
        
    return delta_value, var_delta, (ci_bas, ci_haut)

print("Analyse du Delta pour l'option d'achat asiatique:")
for eps in epsilon_values:
    delta_val, var, (ci_bas, ci_haut) = delta(M, N, x, sigma, T, r, eps, 'call')  
    print(f"ε={eps:.1f}: Delta = {delta_val:.4f}, Variance = {var:.6f}, IC 95% = [{ci_bas:.4f}, {ci_haut:.4f}]")

print("\nAnalyse du Delta pour l'option barrière:")
for eps in epsilon_values:
    delta_val, var, (ci_bas, ci_haut) = delta(M, N, x, sigma, T, r, eps, 'barrier')  
    print(f"ε={eps:.1f}: Delta = {delta_val:.4f}, Variance = {var:.6f}, IC 95% = [{ci_bas:.4f}, {ci_haut:.4f}]")





# question d
    
# Define a range of N values to test
N_values = np.arange(1000, 51000, 5000)  # From 1,000 to 51,000 in steps of 5,000
epsilon = 0.5  # Choose an epsilon value for the finite difference

# Initialize arrays to store variances
var_call = np.zeros(len(N_values))
var_barrier = np.zeros(len(N_values))

# Calculate variances for each N
for i, N in enumerate(N_values):
    # Calculate variance for Asian call
    _, var_call[i], _ = delta(M, N, x, sigma, T, r, epsilon, 'call')
    
    # Calculate variance for Asian barrier
    _, var_barrier[i], _ = delta(M, N, x, sigma, T, r, epsilon, 'barrier')

# Plotting
plt.figure(figsize=(12, 6))

# Asian Call Option Variance
plt.subplot(1, 2, 1)
plt.plot(N_values, var_call, 'b-o', label='Asian Call')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Variance')
plt.title('Variance of Asian Call Delta Estimator')
plt.grid(True)
plt.legend()

# Asian Barrier Option Variance
plt.subplot(1, 2, 2)
plt.plot(N_values, var_barrier, 'r-s', label='Asian Barrier')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Variance')
plt.title('Variance of Asian Barrier Delta Estimator')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Additional plot showing both variances on the same axes for comparison
plt.figure(figsize=(8, 6))
plt.plot(N_values, var_call, 'b-o', label='Asian Call')
plt.plot(N_values, var_barrier, 'r-s', label='Asian Barrier')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Variance')
plt.title('Comparison of Delta Estimator Variances')
plt.grid(True)
plt.legend()
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
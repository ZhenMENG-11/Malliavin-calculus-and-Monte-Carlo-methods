import numpy as np
import matplotlib.pyplot as plt

# Paramètres initiaux
x = 100          # Prix initial de l'actif
r = 0.03         # Taux sans risque
sigma = 0.2      # Volatilité
T = 1            # Temps jusqu'à l'échéance (années)
K1 = 100         # Prix d'exercice du call
K2 = 110         # Prix d'exercice 2
N = 51000        # Nombre de simulations

# Définir la graine aléatoire pour la reproductibilité
np.random.seed(42)


def geo_brownian(M ,N ,T,x,r,sigma):  
    
    dt = T / M
    S_path = np.zeros((M+1,N))   
    S_path[0] = x 
    
    for t in range(1,M+1):
        Z = np.random.standard_normal(N) 
        S_path[t] = S_path[t - 1] * np.exp((r-0.5*sigma**2)*dt +sigma*np.sqrt(dt)*Z) 
    return S_path




def simulate_asian_Call(T, M, r, sigma, x, N, K1):
    S_path = geo_brownian(M, N ,T,x,r,sigma) 
    S_int = (T/M) * S_path.sum(axis=0)  
    payoff = np.maximum(S_int - K1, 0)
    Call_price = np.exp(-r*T) * payoff.mean()
    return Call_price

# Nombre de pas de temps
M1 = 50
M2 = 150
M3 = 250

print(simulate_asian_Call(T, M1, r, sigma, x, N, K1))
print(simulate_asian_Call(T, M2, r, sigma, x, N, K1))
print(simulate_asian_Call(T, M3, r, sigma, x, N, K1))
      



def simulate_asian_barriere(T, M, r, sigma, x, N, K1, K2):
    S_path = geo_brownian(M, N ,T,x,r,sigma) 
    S_int = (T/M) * S_path.sum(axis=0)
    condition = (S_int > K1) & (S_int < K2)
    barriere_price = np.exp(-r*T)* condition.mean()
    return barriere_price


M1 = 50
M2 = 150
M3 = 250

print(simulate_asian_barriere(T, M1, r, sigma, x, N, K1, K2))
print(simulate_asian_barriere(T, M2, r, sigma, x, N, K1, K2))
print(simulate_asian_barriere(T, M3, r, sigma, x, N, K1, K2))




#Calculer la moyenne, la variance et l'intervalle de confiance à 95 %
def compute_statistics(payoffs, discount_factor):
    
    mean = np.mean(payoffs) * discount_factor
    variance = np.var(payoffs, ddof=1) * (discount_factor**2)
    std_err = np.std(payoffs, ddof=1) * discount_factor / np.sqrt(len(payoffs))
    ci_low = mean - 1.96 * std_err
    ci_high = mean + 1.96 * std_err
    return mean, variance, (ci_low, ci_high)






def analyze_convergence(M_fixed, N_steps, payoff_type='call'):
    
    discount = np.exp(-r*T)
    means, variances, ci_lows, ci_highs = [], [], [], []
    
    for n in N_steps:
        S_path = geo_brownian(M_fixed, n, T, x, r, sigma)
        
        if payoff_type == 'call':
            S_int = (T/M_fixed) * S_path.sum(axis=0)  
            payoffs = np.maximum(S_int - K1, 0)
        else:
            S_int = (T/M_fixed) * S_path.sum(axis=0)  
            payoffs = ((S_int > K1) & (S_int < K2)).astype(float)
        
        mean, var, ci = compute_statistics(payoffs, discount)
        means.append(mean)
        variances.append(var)
        ci_lows.append(ci[0])
        ci_highs.append(ci[1])
    
    return means, variances, ci_lows, ci_highs


N_steps = np.arange(1000, 51001, 2000)  

#Exemples M = 50
call_means, call_vars, call_ci_low, call_ci_high = analyze_convergence(M_fixed=50, N_steps=N_steps, payoff_type='call')
cond_means, cond_vars, cond_ci_low, cond_ci_high = analyze_convergence(M_fixed=50, N_steps=N_steps, payoff_type='barrier')



# schema de la convergence
plt.figure(figsize=(14, 6))

# Call asiatique option
plt.subplot(1, 2, 1)
plt.plot(N_steps, call_means, label='Price', color='blue')
plt.fill_between(N_steps, call_ci_low, call_ci_high, alpha=0.2, color='blue', label='95% CI')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Price')
plt.title('Asian Call Option (M=50)')
plt.grid(True)
plt.legend()

# barrière asiatique option
plt.subplot(1, 2, 2)
plt.plot(N_steps, cond_means, label='Price', color='green')
plt.fill_between(N_steps, cond_ci_low, cond_ci_high, alpha=0.2, color='green', label='95% CI')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Price')
plt.title('Conditional Probability (M=50)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()







































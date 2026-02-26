import math
import matplotlib.pyplot as plt

lambda_km1 = 1
beta_k = 0

lambda_ls = []
beta_ls = []
beta_pock_ls = []
k_ls = []

for k in range(1, 100):
    lambda_k = (1 + math.sqrt(1 + lambda_km1**2 * 4)) / 2
    beta_k = (lambda_km1 - 1) / lambda_k
    lambda_km1 = lambda_k

    beta_k_pock = (k - 1) / (k + 2)

    lambda_ls.append(lambda_k)
    beta_ls.append(beta_k)
    beta_pock_ls.append(beta_k_pock)
    k_ls.append(k)


#plt.plot(k_ls, lambda_ls)
plt.plot(k_ls, beta_ls, label = "$\\beta_k(\\lambda_k)$")
plt.plot(k_ls, beta_pock_ls, label = "$\\beta_k = \\frac{k - 1}{k + 2}$")
plt.legend()
plt.show()
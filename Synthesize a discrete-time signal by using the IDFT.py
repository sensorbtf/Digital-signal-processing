import numpy as np
import matplotlib.pyplot as plt

def create_dft_matrix(N):
    W = np.zeros((N, N), dtype=complex)
    for n in range(N):
        for k in range(N):
            W[n, k] = np.exp(-2j * np.pi * n * k / N)
    return W

def idft_matrix_method(x_mu, N):
    W = create_dft_matrix(N)
    W_inv = np.conj(W) / N 
    x_time = np.dot(W_inv, x_mu)
    return x_time


x_mu = np.array([6,8,2,4,3,4,5,0,0,0], dtype=complex)
N = len(x_mu)

x_time = idft_matrix_method(x_mu, N)

plt.stem(np.real(x_time))
plt.title(f"IDFT Synthesized Signal (N = {N})")
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)
plt.show()

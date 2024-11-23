import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Generate a noisy sinusoidal signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
desired_signal = np.sin(2 * np.pi * 5 * t)  # Clean sinusoidal signal (5 Hz)
noise = 0.5 * np.random.randn(len(t))  # Additive Gaussian noise
noisy_signal = desired_signal + noise

# FIR Filter Implementation
def fir_filter(x, b):
    M = len(b)  # Number of coefficients
    y = np.zeros(len(x))  # Filtered output
    for n in range(M, len(x)):
        y[n] = np.dot(b, x[n-M+1:n+1][::-1])
    return y

# FIR Filter Coefficients
fir_b = [0.2, 0.2, 0.2, 0.2, 0.2]  # Example coefficients
fir_filtered_signal = fir_filter(noisy_signal, fir_b)

# IIR Filter Implementation
def iir_filter(x, b, a):
    M = len(b)  # Numerator coefficients
    N = len(a)  # Denominator coefficients
    y = np.zeros(len(x))  # Filtered output
    for n in range(len(x)):
        # Feedforward part (numerator)
        x_segment = x[max(0, n-M+1):n+1]  # Input slice
        y[n] = np.dot(b[:len(x_segment)], x_segment[::-1])  # Convolution

        # Feedback part (denominator)
        if n > 0:
            y_segment = y[max(0, n-N+1):n]  # Output slice
            y[n] -= np.dot(a[1:min(N, len(y_segment)+1)], y_segment[::-1])  # Feedback convolution
    return y

# IIR Filter Coefficients
iir_b = [0.1, 0.2, 0.3]  # Numerator coefficients
iir_a = [1, -0.5, 0.2]   # Denominator coefficients
iir_filtered_signal = iir_filter(noisy_signal, iir_b, iir_a)

# LMS Filter Implementation
def lms_filter(x, d, mu, num_taps):
    n = len(x)
    w = np.zeros(num_taps)  # Filter weights
    y = np.zeros(n)         # Filtered output
    e = np.zeros(n)         # Error signal

    for i in range(num_taps, n):
        x_segment = x[i-num_taps:i][::-1]
        y[i] = np.dot(w, x_segment)  # Filter output
        e[i] = d[i] - y[i]           # Error signal
        w += mu * e[i] * x_segment   # Update weights using LMS rule
    
    return y, e, w

# LMS Parameters
lms_mu = 0.01  # Step size
lms_num_taps = 5  # Number of filter coefficients
lms_filtered_signal, lms_error, lms_weights = lms_filter(noisy_signal, desired_signal, lms_mu, lms_num_taps)

# Plot Results
plt.figure(figsize=(15, 12))

# Original and Noisy Signal
plt.subplot(4, 1, 1)
plt.plot(t, desired_signal, label="Original Signal", linewidth=2)
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.7)
plt.legend()
plt.title("Original and Noisy Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

# FIR Filter Output
plt.subplot(4, 1, 2)
plt.plot(t, fir_filtered_signal, label="FIR Filtered Signal", linewidth=2)
plt.legend()
plt.title("FIR Filter Output")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

# IIR Filter Output
plt.subplot(4, 1, 3)
plt.plot(t, iir_filtered_signal, label="IIR Filtered Signal", linewidth=2)
plt.legend()
plt.title("IIR Filter Output")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

# LMS Filter Output
plt.subplot(4, 1, 4)
plt.plot(t, lms_filtered_signal, label="LMS Filtered Signal", linewidth=2)
plt.plot(t, desired_signal, label="Original Signal", alpha=0.7, linestyle='dashed')
plt.legend()
plt.title("LMS Adaptive Filter Output")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Variant 7: Apply DCT to the signal [10, 20, 30, 40, 50, 60] 
and reconstruct it with a threshold of 15.

AND:
    Reconstruct a sine wave with f = 10Hz, sampled at fs = 30Hz.
Solve the tasks for: 

- sampling and reconstruction
- coding and decoding

@author: mateu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.fftpack import dct, idct

# Step 1: Define the original sine wave
f_signal = 10  # Frequency of the sine wave (Hz)
t = np.linspace(0, 1, 1000, endpoint=False)  # Continuous time vector (1 second)
original_signal = np.sin(2 * np.pi * f_signal * t)  # Original sine wave

# Step 2: Sample the sine wave
f_sample = 30  # Sampling frequency (Hz)
t_sample = np.arange(0, 1, 1 / f_sample)  # Sampled time points
sampled_signal = np.sin(2 * np.pi * f_signal * t_sample)  # Sampled sine wave

# Step 3: Reconstruct the signal using resample
num_samples = len(t)  # Match the original number of points
reconstructed_signal = resample(sampled_signal, num_samples)

# Step 4: Plot original, sampled, and reconstructed signals
plt.figure(figsize=(10, 6))
plt.plot(t, original_signal, label="Original Signal", linewidth=1)
plt.stem(t_sample, sampled_signal, linefmt='r-', markerfmt='ro', basefmt=" ", label="Sampled Signal")
plt.plot(t, reconstructed_signal, linestyle='--', label="Reconstructed Signal", linewidth=1)
plt.title("Signal Sampling and Reconstruction")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

#This example demonstrates signal reconstruction after compression using a simple DCT.

# Apply Discrete Cosine Transform (DCT)
def apply_dct(signal):
    return dct(signal, norm='ortho')

# Reconstruct signal using inverse DCT
def reconstruct_signal(dct_signal, threshold):
    dct_signal[np.abs(dct_signal) < threshold] = 0
    return idct(dct_signal, norm='ortho')

# Example
original_signal = np.array([10, 20, 30, 40, 50, 60])
dct_signal = apply_dct(original_signal)
reconstructed_signal = reconstruct_signal(dct_signal, threshold=15)

print("Original Signal:", original_signal)
print("Compressed Signal:", dct_signal)
print("Reconstructed Signal:", np.round(reconstructed_signal, 2))
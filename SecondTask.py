import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann, flattop
from numpy.fft import fft, fftshift

# Parameters
f1 = 400     # Frequency 1 in Hz
f2 = 400.25  # Frequency 2 in Hz
f3 = 399.75  # Frequency 3 in Hz
fs = 600     # Sampling frequency in Hz
N = 3000     # Number of samples
A = 3        # Amplitude

# Time vector
k = np.arange(N)

# Generate sine signals
x1 = A * np.sin(2 * np.pi * f1 * k / fs)
x2 = A * np.sin(2 * np.pi * f2 * k / fs)
x3 = A * np.sin(2 * np.pi * f3 * k / fs)

# Generate window functions
wrect = np.ones(N)                   # Rectangular window
whann = hann(N, sym=False)           # Hann window
wflattop = flattop(N, sym=False)     # Flat-top window

# Apply windows to signals
X1wrect = fft(x1 * wrect)
X2wrect = fft(x2 * wrect)
X3wrect = fft(x3 * wrect)

X1whann = fft(x1 * whann)
X2whann = fft(x2 * whann)
X3whann = fft(x3 * whann)

X1wflattop = fft(x1 * wflattop)
X2wflattop = fft(x2 * wflattop)
X3wflattop = fft(x3 * wflattop)

# Define DFT normalization function
def fft2db(X):
    N = X.size
    Xtmp = 2 / N * X  # Normalize for sine amplitudes
    Xtmp[0] /= 2      # bin for f=0 Hz exists only once
    if N % 2 == 0:
        Xtmp[N//2] /= 2  # fs/2 bin exists only once for even N
    return 20 * np.log10(np.abs(Xtmp))

# Frequency vector for DFT
df = fs / N
f = np.arange(N) * df

# Plot normalized DFT spectra (175 Hz to 225 Hz)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(f, fft2db(X1wrect), label='f1 Rectangular')
plt.plot(f, fft2db(X2wrect), label='f2 Rectangular')
plt.plot(f, fft2db(X3wrect), label='f3 Rectangular')
plt.xlim(175, 225)
plt.ylim(-60, 0)
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(f, fft2db(X1whann), label='f1 Hann')
plt.plot(f, fft2db(X2whann), label='f2 Hann')
plt.plot(f, fft2db(X3whann), label='f3 Hann')
plt.xlim(175, 225)
plt.ylim(-60, 0)
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(f, fft2db(X1wflattop), label='f1 Flat-Top')
plt.plot(f, fft2db(X2wflattop), label='f2 Flat-Top')
plt.plot(f, fft2db(X3wflattop), label='f3 Flat-Top')
plt.xlim(175, 225)
plt.ylim(-60, 0)
plt.grid(True)
plt.legend()
plt.show()

# DTFT-like spectra using zero-padding
def winDTFTdB(w):
    N = w.size
    Nz = 100 * N  # Zero-padding length
    W = np.zeros(Nz)
    W[0:N] = w
    W = np.abs(fftshift(fft(W)))  # FFT and shift
    W /= np.max(W)  # Normalize to mainlobe maximum
    W = 20 * np.log10(W)  # Convert to dB
    Omega = 2 * np.pi / Nz * np.arange(Nz) - np.pi  # Digital frequencies
    return Omega, W

# Plot window DTFT spectra normalized to mainlobe maximum
Omega, Wrect = winDTFTdB(wrect)
Omega, Whann = winDTFTdB(whann)
Omega, Wflattop = winDTFTdB(wflattop)

plt.figure(figsize=(12, 8))
plt.plot(Omega, Wrect, label='Rectangular')
plt.plot(Omega, Whann, label='Hann')
plt.plot(Omega, Wflattop, label='Flat-Top')
plt.xlim(-np.pi, np.pi)
plt.ylim(-120, 10)
plt.grid(True)
plt.legend()
plt.show()


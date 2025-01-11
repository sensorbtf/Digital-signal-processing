import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann, flattop

# Parametry dla wariantu 7
f1, f2, f3 = 400, 400.25, 399.75
fs = 600  # Częstotliwość próbkowania
N = 3000  # Liczba próbek
amplituda = 3  # Amplituda

# Wektor czasu
k = np.arange(N)
t = k / fs

# Generowanie sygnałów sinusoidalnych
x1 = amplituda * np.sin(2 * np.pi * f1 * t)
x2 = amplituda * np.sin(2 * np.pi * f2 * t)
x3 = amplituda * np.sin(2 * np.pi * f3 * t)

# Diagnostyka sygnałów
print("Pierwsze 10 próbek sygnału x1:", x1[:10])
print("Pierwsze 10 próbek sygnału x2:", x2[:10])
print("Pierwsze 10 próbek sygnału x3:", x3[:10])

# Wykres sygnałów sinusoidalnych
plt.figure(figsize=(10, 5))
plt.plot(t[:500], x1[:500], label=f'Sygnał 1: f1={f1} Hz')
plt.plot(t[:500], x2[:500], label=f'Sygnał 2: f2={f2} Hz')
plt.plot(t[:500], x3[:500], label=f'Sygnał 3: f3={f3} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.title('Wygenerowane sygnały sinusoidalne dla wariantu 7')
plt.grid()
plt.show()

# Definiowanie okien
w_rect = np.ones(N)  # Okno prostokątne
w_hann = hann(N, sym=False)  # Okno Hann'a
w_flattop = flattop(N, sym=False)  # Okno płaskie

# Obliczanie globalnej maksymalnej magnitudy do normalizacji
max_rect = np.max(np.abs(np.fft.fft(x1 * w_rect)))
max_hann = np.max(np.abs(np.fft.fft(x1 * w_hann)))
max_flattop = np.max(np.abs(np.fft.fft(x1 * w_flattop)))
global_max = max(max_rect, max_hann, max_flattop)
print("Globalna maksymalna magnituda (dla DFT):", global_max)

# Funkcja obliczająca DFT i normalizująca wynik
def fft_normalized_global(x, window, global_max):
    X = np.fft.fft(x * window)
    X = np.fft.fftshift(X)  # Przesunięcie częstotliwości zerowej na środek
    magnitude = np.abs(X)
    magnitude /= global_max  # Normalizacja względem globalnego maksimum
    magnitude[magnitude < 1e-10] = 1e-10  # Uniknięcie log10(0)
    return 20 * np.log10(magnitude)  # Konwersja na dB

# Obliczanie widma DFT dla sygnałów
freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
X1_rect_dB = fft_normalized_global(x1, w_rect, global_max)
X1_hann_dB = fft_normalized_global(x1, w_hann, global_max)
X1_flattop_dB = fft_normalized_global(x1, w_flattop, global_max)

# Wykres widm DFT
plt.figure(figsize=(12, 8))
plt.plot(freq, X1_rect_dB, label='Okno prostokątne')
plt.plot(freq, X1_hann_dB, label='Okno Hann\'a')
plt.plot(freq, X1_flattop_dB, label='Okno płaskie')
plt.xlim(-fs/2, fs/2)  # Zakres częstotliwości
plt.ylim(-120, 0)  # Zakres w dB
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Magnituda [dB]')
plt.title('Widmo DFT dla x1 z różnymi oknami (wariant 7)')
plt.legend()
plt.grid()
plt.show()

# Funkcja obliczająca widmo DTFT dla okien
def dtft_spectrum(window):
    Nz = 100 * N  # Liczba punktów z zeropaddingiem
    W = np.zeros(Nz)
    W[:N] = window
    W = np.fft.fftshift(np.fft.fft(W))
    W = np.abs(W) / np.max(np.abs(W))  # Normalizacja względem maksimum
    W[W < 1e-10] = 1e-10  # Uniknięcie log10(0)
    W_dB = 20 * np.log10(W)  # Konwersja na dB
    Omega = np.linspace(-np.pi, np.pi, Nz)  # Zakres częstotliwości cyfrowych
    return Omega, W_dB

# Obliczanie widma DTFT dla okien
Omega, W_rect_dB = dtft_spectrum(w_rect)
_, W_hann_dB = dtft_spectrum(w_hann)
_, W_flattop_dB = dtft_spectrum(w_flattop)

# Wykres widm DTFT
plt.figure(figsize=(12, 8))
plt.plot(Omega, W_rect_dB, label='Okno prostokątne')
plt.plot(Omega, W_hann_dB, label='Okno Hann\'a')
plt.plot(Omega, W_flattop_dB, label='Okno płaskie')
plt.xlim(-np.pi, np.pi)  # Zakres częstotliwości
plt.ylim(-120, 0)  # Zakres w dB
plt.xlabel(r'$\Omega$ [rad/sample]')
plt.ylabel('Magnituda [dB]')
plt.title('Widma DTFT znormalizowane do wartości głównej dla wariantu 7')
plt.legend()
plt.grid()
plt.show()

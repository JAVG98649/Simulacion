"""
Simulación y Análisis de Señales con la Transformada de Fourier
Objetivo: Analizar señales elementales y propiedades de la Transformada de Fourier
en el dominio del tiempo y la frecuencia utilizando NumPy y Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN GENERAL DE MUESTREO
# =============================================================================
fs = 1000                     # Frecuencia de muestreo (1000 Hz)
T = 1.0 / fs                  # Periodo de muestreo (0.001 s)
L = 1000                      # Número total de muestras (1 segundo de duración)
t = np.linspace(-0.5, 0.5, L, endpoint=False)  # Vector de tiempo centrado en 0 (-0.5 s a 0.5 s)

# Vector de frecuencias centrado en 0 Hz para graficar correctamente el espectro
freqs = np.fft.fftshift(np.fft.fftfreq(L, d=T))


def calcular_espectro(senal):
    """
    Calcula la Transformada de Fourier centrada, su magnitud normalizada y su fase.
    Aplica un umbral para limpiar el ruido numérico en la fase cuando la magnitud es casi 0.
    """
    # Uso de np.fft.fft() según los requerimientos de la actividad
    espectro_complejo = np.fft.fftshift(np.fft.fft(senal))
    magnitud = np.abs(espectro_complejo) / L
    
    # Limpieza de ruido numérico para la fase (umbral del 1% de la magnitud máxima)
    espectro_limpio = np.where(magnitud < 0.01 * np.max(magnitud), 0, espectro_complejo)
    fase = np.angle(espectro_limpio)
    
    return espectro_complejo, magnitud, fase


# =============================================================================
# PARTE A: SEÑALES ELEMENTALES (TIEMPO, MAGNITUD Y FASE)
# =============================================================================

# 1. Definición de las señales en el dominio del tiempo
# a) Pulso rectangular (ancho T_p = 0.2 s centrado en t = 0)
pulso_rect = np.where(np.abs(t) <= 0.1, 1.0, 0.0)

# b) Función escalón unitario u(t) (cambia de 0 a 1 en t = 0)
escalon = np.where(t >= 0.0, 1.0, 0.0)

# c) Función senoidal de f0 = 15 Hz y amplitud = 1
f0 = 15
senoidal = np.sin(2 * np.pi * f0 * t)

# 2. Cálculo de la Transformada de Fourier
_, mag_pulso, fase_pulso = calcular_espectro(pulso_rect)
_, mag_escalon, fase_escalon = calcular_espectro(escalon)
_, mag_seno, fase_seno = calcular_espectro(senoidal)

# 3. Visualización de los resultados de señales elementales
fig1, axs1 = plt.subplots(3, 3, figsize=(15, 10))
fig1.suptitle("Análisis de Señales Elementales: Tiempo vs. Frecuencia (Magnitud y Fase)", fontsize=14, fontweight='bold')

senales_info = [
    ("Pulso Rectangular", pulso_rect, mag_pulso, fase_pulso, (-100, 100)),
    ("Función Escalón u(t)", escalon, mag_escalon, fase_escalon, (-50, 50)),
    ("Función Senoidal (15 Hz)", senoidal, mag_seno, fase_seno, (-50, 50))
]

for fila, (nombre, sig_t, mag_f, fase_f, lim_f) in enumerate(senales_info):
    # Columna 1: Dominio del tiempo
    axs1[fila, 0].plot(t, sig_t, color='tab:blue', lw=1.8)
    axs1[fila, 0].set_title(f"{nombre} - Dominio del Tiempo")
    axs1[fila, 0].set_xlabel("Tiempo (s)")
    axs1[fila, 0].set_ylabel("Amplitud")
    axs1[fila, 0].grid(True, alpha=0.4)

    # Columna 2: Magnitud del Espectro
    axs1[fila, 1].plot(freqs, mag_f, color='tab:red', lw=1.8)
    axs1[fila, 1].set_title(f"Espectro de Magnitud |X(f)|")
    axs1[fila, 1].set_xlabel("Frecuencia (Hz)")
    axs1[fila, 1].set_ylabel("Magnitud Normalizada")
    axs1[fila, 1].set_xlim(lim_f)
    axs1[fila, 1].grid(True, alpha=0.4)

    # Columna 3: Fase del Espectro
    axs1[fila, 2].plot(freqs, fase_f, color='tab:green', lw=1.5)
    axs1[fila, 2].set_title(f"Espectro de Fase ∠X(f)")
    axs1[fila, 2].set_xlabel("Frecuencia (Hz)")
    axs1[fila, 2].set_ylabel("Fase (radianes)")
    axs1[fila, 2].set_xlim(lim_f)
    axs1[fila, 2].grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.96])


# =============================================================================
# PARTE B: ANÁLISIS DE PROPIEDADES Y TRANSFORMACIONES
# =============================================================================

# --- PROPIEDAD 1: LINEALIDAD ---
# a*x1(t) + b*x2(t) <---> a*X1(f) + b*X2(f)
a, b = 2.0, 1.5
x1 = np.cos(2 * np.pi * 10 * t)   # Coseno de 10 Hz
x2 = np.cos(2 * np.pi * 35 * t)   # Coseno de 35 Hz
senal_combinada = a * x1 + b * x2

# FFT directa de la combinación vs. Suma lineal de las FFT individuales
fft_directa, mag_directa, _ = calcular_espectro(senal_combinada)
fft_x1, _, _ = calcular_espectro(x1)
fft_x2, _, _ = calcular_espectro(x2)
mag_suma_individual = np.abs(a * fft_x1 + b * fft_x2) / L

# Error máximo entre ambos métodos para verificar linealidad numérica
error_linealidad = np.max(np.abs(mag_directa - mag_suma_individual))
print(f"[Verificación de Linealidad] Error máximo entre FFT(a*x1 + b*x2) y a*FFT(x1) + b*FFT(x2): {error_linealidad:.2e}")


# --- PROPIEDAD 2: DESPLAZAMIENTO EN EL TIEMPO ---
# x(t - t0) <---> X(f) * exp(-j * 2*pi * f * t0)
# Un retardo temporal NO altera la magnitud, pero añade una pendiente lineal a la fase.
t0 = 0.15  # Desplazamiento de 0.15 segundos
pulso_desplazado = np.where(np.abs(t - t0) <= 0.1, 1.0, 0.0)
_, mag_desplazado, fase_desplazado = calcular_espectro(pulso_desplazado)


# --- PROPIEDAD 3: ESCALAMIENTO EN TIEMPO Y FRECUENCIA ---
# x(a*t) <---> (1/|a|) * X(f/a)
# Comprimir el pulso en el tiempo (ancho menor) expande el espectro en frecuencia.
pulso_angosto = np.where(np.abs(t) <= 0.04, 1.0, 0.0)  # Ancho 0.08 s (comprimido en el tiempo)
_, mag_angosto, _ = calcular_espectro(pulso_angosto)


# Visualización de las 3 propiedades
fig2, axs2 = plt.subplots(3, 2, figsize=(14, 10))
fig2.suptitle("Verificación de Propiedades de la Transformada de Fourier", fontsize=14, fontweight='bold')

# Fila 1: Linealidad
axs2[0, 0].plot(t, senal_combinada, color='tab:purple')
axs2[0, 0].set_title("Linealidad (Tiempo): 2·cos(2π·10t) + 1.5·cos(2π·35t)")
axs2[0, 0].set_xlabel("Tiempo (s)")
axs2[0, 0].set_ylabel("Amplitud")
axs2[0, 0].set_xlim(-0.2, 0.2)
axs2[0, 0].grid(True, alpha=0.4)

axs2[0, 1].plot(freqs, mag_directa, 'r-', lw=2, label="FFT de la suma: |FFT(a·x1 + b·x2)|")
axs2[0, 1].plot(freqs, mag_suma_individual, 'k--', lw=2, label="Suma de FFTs: |a·FFT(x1) + b·FFT(x2)|")
axs2[0, 1].set_title("Linealidad (Frecuencia): Superposición Exacta de Espectros")
axs2[0, 1].set_xlabel("Frecuencia (Hz)")
axs2[0, 1].set_ylabel("Magnitud")
axs2[0, 1].set_xlim(-60, 60)
axs2[0, 1].legend()
axs2[0, 1].grid(True, alpha=0.4)

# Fila 2: Desplazamiento en el Tiempo (Magnitud idéntica, Fase modificada)
axs2[1, 0].plot(freqs, mag_pulso, 'b-', lw=2, label="Pulso Original (t0 = 0 s)")
axs2[1, 0].plot(freqs, mag_desplazado, 'r--', lw=2, label="Pulso Desplazado (t0 = 0.15 s)")
axs2[1, 0].set_title("Desplazamiento Temporal: La Magnitud |X(f)| NO cambia")
axs2[1, 0].set_xlabel("Frecuencia (Hz)")
axs2[1, 0].set_ylabel("Magnitud")
axs2[1, 0].set_xlim(-50, 50)
axs2[1, 0].legend()
axs2[1, 0].grid(True, alpha=0.4)

axs2[1, 1].plot(freqs, fase_pulso, 'b-', lw=1.5, label="Fase Original (t0 = 0 s)")
axs2[1, 1].plot(freqs, fase_desplazado, 'r--', lw=1.5, label="Fase Desplazada (t0 = 0.15 s)")
axs2[1, 1].set_title("Desplazamiento Temporal: La Fase ∠X(f) adquiere pendiente lineal")
axs2[1, 1].set_xlabel("Frecuencia (Hz)")
axs2[1, 1].set_ylabel("Fase (rad)")
axs2[1, 1].set_xlim(-25, 25)
axs2[1, 1].legend()
axs2[1, 1].grid(True, alpha=0.4)

# Fila 3: Escalamiento en Tiempo y Frecuencia
axs2[2, 0].plot(t, pulso_rect, 'b-', lw=2, label="Pulso Ancho (τ = 0.20 s)")
axs2[2, 0].plot(t, pulso_angosto, 'darkorange', lw=2, label="Pulso Comprimido (τ = 0.08 s)")
axs2[2, 0].set_title("Escalamiento (Tiempo): Compresión Temporal")
axs2[2, 0].set_xlabel("Tiempo (s)")
axs2[2, 0].set_ylabel("Amplitud")
axs2[2, 0].set_xlim(-0.25, 0.25)
axs2[2, 0].legend()
axs2[2, 0].grid(True, alpha=0.4)

axs2[2, 1].plot(freqs, mag_pulso, 'b-', lw=2, label="Espectro Pulso Ancho (Lóbulo estrecho)")
axs2[2, 1].plot(freqs, mag_angosto, 'darkorange', lw=2, label="Espectro Pulso Comprimido (Lóbulo expandido)")
axs2[2, 1].set_title("Escalamiento (Frecuencia): Compresión en Tiempo = Expansión en Frecuencia")
axs2[2, 1].set_xlabel("Frecuencia (Hz)")
axs2[2, 1].set_ylabel("Magnitud")
axs2[2, 1].set_xlim(-100, 100)
axs2[2, 1].legend()
axs2[2, 1].grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
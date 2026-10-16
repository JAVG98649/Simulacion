# Simulacion
Proyecto para la actividad numero 2 de la clase de señales y sistemas

# Simulación y Análisis de Señales con la Transformada de Fourier

Este repositorio contiene la implementación en **Python** para el análisis de señales en el dominio del tiempo y de la frecuencia empleando la Transformada Rápida de Fourier (`FFT`).

## Propósito del Proyecto
Visualizar y comparar cómo se representan las señales elementales en el dominio del tiempo frente a su espectro de magnitud y fase en el dominio de la frecuencia, además de comprobar experimentalmente las propiedades fundamentales de la Transformada de Fourier.

## Contenido del Código (`simulacion_fourier.py`)
1. **Señales Elementales:**
   - **Pulso Rectangular:** Transformación hacia la función $\text{sinc}(f)$.
   - **Función Escalón Unitario $u(t)$:** Análisis de su componente DC ($0\text{ Hz}$) y decaimiento espectral.
   - **Función Senoidal ($15\text{ Hz}$):** Localización de impulsos en $\pm f_0$ tanto en magnitud como en fase.
2. **Verificación de Propiedades:**
   - **Linealidad:** Superposición espectral de una combinación lineal de señales.
   - **Desplazamiento en el Tiempo:** Invarianza del espectro de magnitud y modificación lineal del espectro de fase.
   - **Escalamiento en Frecuencia:** Demostración de cómo la compresión temporal expande el ancho de banda en frecuencia.

## Requisitos e Instalación
Asegúrate de contar con Python 3.x e instala las dependencias necesarias ejecutando:

```bash
pip install numpy scipy matplotlib

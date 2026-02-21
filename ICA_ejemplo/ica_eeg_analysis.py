# ICA (Independent Component Analysis) aplicado a senales EEG
# Dataset: EEG During Mental Arithmetic Tasks (PhysioNet)
# CC3074 - Mineria de Datos, UVG

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import pyedflib
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


# -- Carga y preprocesamiento --

def cargar_eeg(filepath):
    """Lee un archivo EDF y devuelve las senales, nombres de canales y freq de muestreo."""
    f = pyedflib.EdfReader(filepath)
    n_channels = f.signals_in_file
    channel_names = f.getSignalLabels()
    sample_rate = f.getSampleFrequency(0)

    signals = np.zeros((n_channels, f.getNSamples()[0]))
    for i in range(n_channels):
        signals[i, :] = f.readSignal(i)

    f.close()
    return signals, channel_names, sample_rate


def preprocesar(signals):
    """Estandariza las senales (media=0, std=1) y transpone a (muestras x canales)."""
    signals_T = signals.T
    scaler = StandardScaler()
    signals_scaled = scaler.fit_transform(signals_T)
    return signals_scaled, scaler


def aplicar_ica(signals_scaled, n_components=None, random_state=42):
    """Aplica FastICA y devuelve fuentes, matriz de mezcla y el modelo."""
    ica = FastICA(
        n_components=n_components,
        algorithm='parallel',
        whiten='unit-variance',
        fun='logcosh',            # funcion de contraste
        max_iter=1000,
        tol=1e-4,
        random_state=random_state
    )
    sources = ica.fit_transform(signals_scaled)
    mixing_matrix = ica.mixing_
    return sources, mixing_matrix, ica


# -- Graficas --

def plot_senales_originales(signals, channel_names, sample_rate, n_channels=6,
                            n_seconds=5, title="Senales EEG Originales"):
    n_samples = int(n_seconds * sample_rate)
    time = np.arange(n_samples) / sample_rate

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    colors = plt.cm.Set2(np.linspace(0, 1, n_channels))

    for i in range(n_channels):
        axes[i].plot(time, signals[i, :n_samples], color=colors[i], linewidth=0.7)
        axes[i].set_ylabel(channel_names[i], fontsize=9, rotation=0, labelpad=50)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    axes[-1].set_xlabel('Tiempo (s)')
    plt.tight_layout()
    plt.savefig('01_senales_originales.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_componentes_independientes(sources, sample_rate, n_components=6,
                                    n_seconds=5, title="Componentes Independientes (ICA)"):
    n_samples = int(n_seconds * sample_rate)
    time = np.arange(n_samples) / sample_rate

    fig, axes = plt.subplots(n_components, 1, figsize=(14, 2 * n_components), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    colors = plt.cm.Dark2(np.linspace(0, 1, n_components))

    for i in range(n_components):
        axes[i].plot(time, sources[:n_samples, i], color=colors[i], linewidth=0.7)
        axes[i].set_ylabel(f'IC {i+1}', fontsize=9, rotation=0, labelpad=30)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    axes[-1].set_xlabel('Tiempo (s)')
    plt.tight_layout()
    plt.savefig('02_componentes_independientes.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_matriz_mezcla(mixing_matrix, channel_names, n_components=None):
    """Heatmap de la matriz de mezcla A (como contribuye cada IC a cada canal)."""
    matrix = mixing_matrix[:, :n_components] if n_components else mixing_matrix

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')

    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(channel_names, fontsize=8)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f'IC {i+1}' for i in range(matrix.shape[1])], fontsize=8, rotation=45)
    ax.set_title('Matriz de Mezcla (A) - Contribucion de cada IC a cada canal EEG',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Componentes Independientes')
    ax.set_ylabel('Canales EEG')

    plt.colorbar(im, ax=ax, label='Peso de la contribucion')
    plt.tight_layout()
    plt.savefig('03_matriz_mezcla.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_kurtosis_componentes(sources):
    """Curtosis de cada IC - mide que tan no-gaussiana es cada componente."""
    kurt_values = kurtosis(sources, fisher=True)
    n_components = len(kurt_values)
    indices_sorted = np.argsort(np.abs(kurt_values))[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e74c3c' if k > 0 else '#3498db' for k in kurt_values[indices_sorted]]
    ax.bar(range(n_components), kurt_values[indices_sorted], color=colors,
           edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f'IC {i+1}' for i in indices_sorted], fontsize=8, rotation=45)
    ax.set_ylabel('Curtosis (Fisher)')
    ax.set_title('Curtosis de las Componentes Independientes\n(Mayor |curtosis| = Mayor no-gaussianidad)',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=3, color='gray', linewidth=0.8, linestyle='--',
               label='Curtosis leptocurtica (ref.)')
    ax.axhline(y=-1.2, color='gray', linewidth=0.8, linestyle=':',
               label='Curtosis platicurtica (ref.)')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('04_kurtosis_componentes.png', dpi=150, bbox_inches='tight')
    plt.show()
    return kurt_values


def plot_reconstruccion(signals_scaled, ica, channel_names, sample_rate,
                        n_channels=4, n_seconds=3):
    """Compara senales originales vs reconstruidas para validar que ICA es invertible."""
    sources = ica.transform(signals_scaled)
    reconstructed = ica.inverse_transform(sources)

    n_samples = int(n_seconds * sample_rate)
    time = np.arange(n_samples) / sample_rate

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2.5 * n_channels), sharex=True)
    fig.suptitle('Senales Originales vs. Reconstruidas desde ICA',
                 fontsize=15, fontweight='bold', y=1.02)

    for i in range(n_channels):
        axes[i].plot(time, signals_scaled[:n_samples, i],
                     color='#2c3e50', linewidth=0.8, label='Original', alpha=0.7)
        axes[i].plot(time, reconstructed[:n_samples, i],
                     color='#e74c3c', linewidth=0.8, label='Reconstruida',
                     linestyle='--', alpha=0.7)
        axes[i].set_ylabel(channel_names[i], fontsize=9, rotation=0, labelpad=50)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        if i == 0:
            axes[i].legend(fontsize=8, loc='upper right')

    error = np.mean((signals_scaled - reconstructed) ** 2)
    axes[-1].set_xlabel(f'Tiempo (s) | MSE de reconstruccion: {error:.2e}')

    plt.tight_layout()
    plt.savefig('05_reconstruccion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"MSE de reconstruccion: {error:.2e}")


def plot_espectro_componentes(sources, sample_rate, n_components=6):
    """FFT de las ICs para ver en que bandas de frecuencia se concentra cada una."""
    fig, axes = plt.subplots(n_components, 1, figsize=(14, 2 * n_components), sharex=True)
    fig.suptitle('Espectro de Frecuencias de las Componentes Independientes',
                 fontsize=15, fontweight='bold', y=1.02)

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_components))

    # bandas tipicas de EEG
    bandas = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
    }

    for i in range(n_components):
        n = len(sources[:, i])
        fft_vals = np.fft.rfft(sources[:, i])
        fft_freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
        power = np.abs(fft_vals) ** 2

        # solo hasta 40 Hz que es lo relevante para EEG
        mask = fft_freqs <= 40
        axes[i].plot(fft_freqs[mask], power[mask], color=colors[i], linewidth=0.8)
        axes[i].fill_between(fft_freqs[mask], power[mask], alpha=0.3, color=colors[i])
        axes[i].set_ylabel(f'IC {i+1}', fontsize=9, rotation=0, labelpad=30)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        # marcar bandas solo en el primer subplot para no saturar
        if i == 0:
            for nombre, (f_low, f_high) in bandas.items():
                axes[i].axvspan(f_low, f_high, alpha=0.1, color='gray')
                axes[i].text((f_low + f_high)/2, axes[i].get_ylim()[1]*0.9,
                            nombre.split('(')[0].strip(),
                            ha='center', fontsize=7, color='gray')

    axes[-1].set_xlabel('Frecuencia (Hz)')
    plt.tight_layout()
    plt.savefig('06_espectro_componentes.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparacion_baseline_task(sources_baseline, sources_task, sample_rate,
                                    n_components=4, n_seconds=3):
    """Side by side de componentes en reposo vs durante la tarea aritmetica."""
    n_samples = int(n_seconds * sample_rate)
    time = np.arange(n_samples) / sample_rate

    fig, axes = plt.subplots(n_components, 2, figsize=(16, 2.5 * n_components),
                              sharex=True)
    fig.suptitle('Comparacion: Baseline (Reposo) vs. Tarea Mental\nComponentes Independientes',
                 fontsize=15, fontweight='bold', y=1.02)

    for i in range(n_components):
        axes[i, 0].plot(time, sources_baseline[:n_samples, i],
                        color='#2ecc71', linewidth=0.7)
        axes[i, 0].set_ylabel(f'IC {i+1}', fontsize=9, rotation=0, labelpad=30)
        axes[i, 0].spines['top'].set_visible(False)
        axes[i, 0].spines['right'].set_visible(False)
        if i == 0:
            axes[i, 0].set_title('Baseline (Reposo)', fontsize=12, fontweight='bold')

        axes[i, 1].plot(time, sources_task[:n_samples, i],
                        color='#e74c3c', linewidth=0.7)
        axes[i, 1].spines['top'].set_visible(False)
        axes[i, 1].spines['right'].set_visible(False)
        if i == 0:
            axes[i, 1].set_title('Tarea Aritmetica Mental', fontsize=12, fontweight='bold')

    axes[-1, 0].set_xlabel('Tiempo (s)')
    axes[-1, 1].set_xlabel('Tiempo (s)')

    plt.tight_layout()
    plt.savefig('07_comparacion_baseline_task.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_varianza_componentes(sources, title="Varianza Explicada por Componente"):
    varianzas = np.var(sources, axis=0)
    varianzas_rel = varianzas / varianzas.sum() * 100
    n_comp = len(varianzas)
    idx_sorted = np.argsort(varianzas_rel)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # barras
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_comp))
    ax1.bar(range(n_comp), varianzas_rel[idx_sorted], color=colors, edgecolor='white')
    ax1.set_xticks(range(n_comp))
    ax1.set_xticklabels([f'IC {i+1}' for i in idx_sorted], fontsize=7, rotation=45)
    ax1.set_ylabel('Varianza relativa (%)')
    ax1.set_title('Varianza por Componente', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # acumulada
    var_acum = np.cumsum(varianzas_rel[idx_sorted])
    ax2.plot(range(n_comp), var_acum, 'o-', color='#8e44ad', linewidth=2, markersize=5)
    ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=95, color='gray', linestyle=':', alpha=0.7, label='95%')
    ax2.set_xticks(range(n_comp))
    ax2.set_xticklabels([f'IC {i+1}' for i in idx_sorted], fontsize=7, rotation=45)
    ax2.set_ylabel('Varianza acumulada (%)')
    ax2.set_title('Varianza Acumulada', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('08_varianza_componentes.png', dpi=150, bbox_inches='tight')
    plt.show()


# -- Main --

if __name__ == "__main__":

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'eeg-during-mental-arithmetic-tasks-1.0.0')

    SUBJECT = "Subject00"

    print("=" * 60)
    print("  ICA - Analisis de senales EEG")
    print("  Dataset: EEG During Mental Arithmetic Tasks")
    print("=" * 60)

    # cargar info de sujetos
    subject_info = pd.read_csv(os.path.join(DATA_DIR, 'subject-info.csv'))
    print(subject_info.to_string(index=False))

    suj = subject_info[subject_info['Subject'] == SUBJECT].iloc[0]
    grupo = 'Bueno (G)' if suj['Count quality'] == 1 else 'Malo (B)'
    print(f"\nSujeto: {SUBJECT} | Edad: {suj['Age']} | Genero: {suj['Gender']} | Grupo: {grupo}")

    # cargar EEG baseline (reposo) y tarea (aritmetica mental)
    filepath_baseline = os.path.join(DATA_DIR, f"{SUBJECT}_1.edf")
    signals_baseline, channel_names, sample_rate = cargar_eeg(filepath_baseline)
    print(f"Baseline: {signals_baseline.shape[0]} canales x {signals_baseline.shape[1]} muestras, {sample_rate} Hz")
    print(f"Duracion: {signals_baseline.shape[1]/sample_rate:.1f}s | Canales: {', '.join(channel_names)}")

    filepath_task = os.path.join(DATA_DIR, f"{SUBJECT}_2.edf")
    signals_task, _, _ = cargar_eeg(filepath_task)
    print(f"Tarea: {signals_task.shape[0]} canales x {signals_task.shape[1]} muestras, duracion {signals_task.shape[1]/sample_rate:.1f}s")

    # graficar senales originales
    plot_senales_originales(signals_baseline, channel_names, sample_rate,
                           n_channels=6, n_seconds=5,
                           title=f"Senales EEG Originales - {SUBJECT} (Baseline)")

    # preprocesamiento: estandarizacion
    signals_baseline_scaled, scaler_baseline = preprocesar(signals_baseline)
    signals_task_scaled, scaler_task = preprocesar(signals_task)
    print(f"\nDespues de estandarizar -> baseline: {signals_baseline_scaled.shape}, tarea: {signals_task_scaled.shape}")
    print(f"Media (primeros 3 canales): {signals_baseline_scaled.mean(axis=0)[:3].round(6)}")
    print(f"Std (primeros 3 canales):   {signals_baseline_scaled.std(axis=0)[:3].round(6)}")

    # aplicar ICA
    n_components = signals_baseline.shape[0]   # un componente por canal
    print(f"\nAplicando FastICA con {n_components} componentes...")

    sources_baseline, mixing_baseline, ica_baseline = aplicar_ica(
        signals_baseline_scaled, n_components=n_components
    )
    print(f"ICA baseline -> fuentes: {sources_baseline.shape}, mezcla: {mixing_baseline.shape}")

    sources_task, mixing_task, ica_task = aplicar_ica(
        signals_task_scaled, n_components=n_components
    )
    print(f"ICA tarea -> listo")

    # visualizaciones
    plot_componentes_independientes(sources_baseline, sample_rate,
                                    n_components=6, n_seconds=5,
                                    title=f"Componentes Independientes - {SUBJECT} (Baseline)")

    plot_matriz_mezcla(mixing_baseline, channel_names)

    # analisis de curtosis (no-gaussianidad)
    kurt_values = plot_kurtosis_componentes(sources_baseline)

    # tabla de curtosis
    kurt_df = pd.DataFrame({
        'Componente': [f'IC {i+1}' for i in range(len(kurt_values))],
        'Curtosis': kurt_values,
        '|Curtosis|': np.abs(kurt_values),
        'Tipo': ['Super-gaussiana' if k > 0 else 'Sub-gaussiana' for k in kurt_values]
    }).sort_values('|Curtosis|', ascending=False)
    print("\nCurtosis por componente:")
    print(kurt_df.to_string(index=False))

    # reconstruccion (para verificar que ICA no pierde info)
    plot_reconstruccion(signals_baseline_scaled, ica_baseline, channel_names,
                        sample_rate, n_channels=4, n_seconds=3)

    # espectro de frecuencias
    plot_espectro_componentes(sources_baseline, sample_rate, n_components=6)

    # comparacion baseline vs tarea usando el mismo ICA
    # se usa el modelo del baseline para transformar la tarea y poder comparar
    sources_task_aligned = ica_baseline.transform(signals_task_scaled)
    plot_comparacion_baseline_task(sources_baseline, sources_task_aligned,
                                   sample_rate, n_components=4, n_seconds=3)

    # varianza por componente
    plot_varianza_componentes(sources_baseline,
                             title=f"Varianza por Componente - {SUBJECT} (Baseline)")

    # tabla comparativa baseline vs tarea
    print("\n" + "=" * 60)
    print("  TABLA COMPARATIVA: BASELINE vs. TAREA")
    print("=" * 60)

    kurt_baseline = kurtosis(sources_baseline, fisher=True)
    kurt_task = kurtosis(sources_task_aligned, fisher=True)
    var_baseline = np.var(sources_baseline, axis=0)
    var_task = np.var(sources_task_aligned, axis=0)

    comparison_df = pd.DataFrame({
        'Componente': [f'IC {i+1}' for i in range(n_components)],
        'Curtosis Baseline': kurt_baseline.round(3),
        'Curtosis Tarea': kurt_task.round(3),
        'Delta Curtosis': (kurt_task - kurt_baseline).round(3),
        'Varianza Baseline': var_baseline.round(4),
        'Varianza Tarea': var_task.round(4),
        'Delta Varianza (%)': ((var_task - var_baseline) / var_baseline * 100).round(2)
    })
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('tabla_comparativa_ica.csv', index=False)
    print("Tabla guardada en tabla_comparativa_ica.csv")

    # resumen
    print(f"\nListo. Sujeto: {SUBJECT}, {n_components} canales/componentes, algoritmo FastICA (logcosh)")
    print("Graficas guardadas como PNG en el directorio actual.")

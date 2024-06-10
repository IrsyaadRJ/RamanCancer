import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_individual_spectra(data, labels, n_samples_to_plot=5):
    for i in range(n_samples_to_plot):
        plt.plot(data[i], label=f'Sample {i}, Cell Type: {labels[i]}')

    plt.xlabel('Raman Shift (Wavenumber)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Individual Raman Spectra')
    plt.show()


def plot_mean_spectra(data, labels):
    unique_cell_types = np.unique(labels)
    mean_spectra = {}

    for cell_type in unique_cell_types:
        mean_spectra[cell_type] = np.mean(data[labels == cell_type], axis=0)
        plt.plot(mean_spectra[cell_type],
                 label=f'Mean Spectrum for {cell_type}')

    plt.xlabel('Raman Shift (Wavenumber)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Mean Raman Spectra by Cell Type')
    plt.show()


def plot_spectral_distribution(data, labels, Raman_Shifts=[100, 500, 1000, 1500, 2000]):
    data_df = pd.DataFrame(data)
    data_df['Cell_Type'] = labels

    sns.violinplot(x='Cell_Type', y='Intensity', hue='Raman_Shift', data=data_df.melt(id_vars='Cell_Type',
                   var_name='Raman_Shift', value_name='Intensity').query('Raman_Shift in @Raman_Shifts'), inner='quartiles', cut=0)
    plt.title('Intensity Distribution by Cell Type and Raman Shift')
    plt.show()


def plot_heatmap(data):
    sns.heatmap(data, cmap='viridis', yticklabels=False)
    plt.xlabel('Raman Shift (Wavenumber)')
    plt.ylabel('Samples')
    plt.title('Heatmap of Raman Spectra Intensities')
    plt.show()

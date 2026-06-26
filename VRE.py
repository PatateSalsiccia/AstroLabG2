import pandas as pd
import numpy as np

file_path = "/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/VIS_SD20_four_quasar_average_spectra.csv"
df = pd.read_csv(file_path)

group_col = "group"
wavelength_col = "wavelength_nm"
intensity_col = "intensity"

red_min, red_max = 665, 685
nir_min, nir_max = 750, 800

def interp_intensity(group_data, wavelength):
    group_data = group_data.sort_values(wavelength_col)
    return np.interp(
        wavelength,
        group_data[wavelength_col],
        group_data[intensity_col]
    )

def calculate_all_metrics(group_data):
    group_data = group_data.sort_values(wavelength_col).copy()

    red_region = group_data[
        (group_data[wavelength_col] >= red_min) &
        (group_data[wavelength_col] <= red_max)
    ]

    nir_region = group_data[
        (group_data[wavelength_col] >= nir_min) &
        (group_data[wavelength_col] <= nir_max)
    ]

    red_mean = red_region[intensity_col].mean()
    nir_mean = nir_region[intensity_col].mean()

    ndvi_like = (nir_mean - red_mean) / (nir_mean + red_mean)

    i_680 = interp_intensity(group_data, 680)
    i_750 = interp_intensity(group_data, 750)

    red_edge_slope = (i_750 - i_680) / (750 - 680)
    red_edge_ratio = i_750 / i_680

    wavelength = group_data[wavelength_col].values
    intensity = group_data[intensity_col].values

    derivative = np.gradient(intensity, wavelength)

    mask = (wavelength >= 650) & (wavelength <= 780)

    wavelength_region = wavelength[mask]
    derivative_region = derivative[mask]

    max_index = np.argmax(derivative_region)

    max_slope_wavelength = wavelength_region[max_index]
    max_slope_value = derivative_region[max_index]

    return pd.Series({
        "Red mean 665-685 nm": red_mean,
        "NIR mean 750-800 nm": nir_mean,
        "NDVI-like index": ndvi_like,
        "I680": i_680,
        "I750": i_750,
        "Red-edge slope 680-750 nm": red_edge_slope,
        "Red-edge ratio I750/I680": red_edge_ratio,
        "Max slope wavelength nm": max_slope_wavelength,
        "Max slope value": max_slope_value
    })

results_df = df.groupby(group_col).apply(calculate_all_metrics).reset_index()

print(results_df)

results_df.to_csv(
    "/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/visible_spectral_indices_results_SD20.csv",
    index=False
)
import pandas as pd

input_path = "/Users/alessandrodelmonte/Desktop/IR_M2_P-vac_AVG_clean.csv"
output_path = "/Users/alessandrodelmonte/Desktop/IR_M2_P-vac_AVG_WL.csv"

# Read CSV using the first row as column names
df = pd.read_csv(input_path)

# Convert columns to numeric
df["wavenumber_cm-1"] = pd.to_numeric(df["wavenumber_cm-1"])
df["intensity"] = pd.to_numeric(df["intensity"])

# Convert wavenumber in cm^-1 to wavelength in nm
df["wavelength_nm"] = 10_000_000 / df["wavenumber_cm-1"]

# Keep only wavelength + intensity
df_wavelength = df[["wavelength_nm", "intensity"]]

# Save converted file
df_wavelength.to_csv(output_path, index=False)

print(df_wavelength)
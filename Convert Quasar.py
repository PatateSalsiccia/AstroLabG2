
import pandas as pd


# Load the exported CSV with no headers
raw = pd.read_csv("/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/QuasarFiles/FinalSD20_VIS_M2_P-vac_AVG.csv", header=None)
print (raw)

# First row = wavenumber
# Second row = intensity
df = pd.DataFrame({
    "wavenumber_cm-1": raw.iloc[0].values,
    "intensity": raw.iloc[3].values
})
print(df)
# Save the DataFrame to a new CSV file
output_path = "/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/FinalSD20_VIS_M2_P-vac_AVG_clean.csv"
df.to_csv(output_path, index=False)


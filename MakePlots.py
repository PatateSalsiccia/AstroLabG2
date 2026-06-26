import pandas as pd
import matplotlib.pyplot as plt

# Load your exported data
df = pd.read_csv("/Users/alessandrodelmonte/Desktop/IR_M2_P-vac_AVG_WL.csv")

x = df["wavelength_nm"]
y = df["intensity"]

plt.figure(figsize=(9, 5), dpi=300)
plt.plot(x, y, linewidth=1.5)

plt.xlabel("Wavelength / nm", fontsize=12)
plt.ylabel("Intensity / a.u.", fontsize=12)
plt.title("Infrared Spectrum: 85% Spirulina , 15% NaCl in vacuum", fontsize=14)

plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig("/Users/alessandrodelmonte/Desktop/Plot_IR_M2_P-vac_AVG_WL.png", dpi=300)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Input files: Quasar average spectra
# -----------------------------
files = {
    "M1 Ambiente": Path("/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/FinalSD20_VIS_M1_P-amb_AVG_clean.csv"),
    "M1 Vuoto": Path("/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/FinalSD20_VIS_M1_P-vac_AVG_clean.csv"),
    "M2 Ambiente": Path("/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/FinalSD20_VIS_M2_P-amb_AVG_clean.csv"),
    "M2 Vuoto": Path("/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/FinalSD20_VIS_M2_P-vac_AVG_clean.csv"),
}

# -----------------------------
# Load average spectra
# -----------------------------
average_spectra = []

for label, path in files.items():
    df = pd.read_csv(path)

    # Make sure the first two columns are named consistently
    df = df.rename(columns={
        df.columns[0]: "wavenumber_cm-1",
        df.columns[1]: "intensity"
    })

    df["group"] = label
    df["wavelength_nm"] = 10_000_000 / df["wavenumber_cm-1"]

    average_spectra.append(df[[
        "group",
        "wavenumber_cm-1",
        "wavelength_nm",
        "intensity"
    ]])

avg_all = pd.concat(average_spectra, ignore_index=True)

# Save combined data
avg_all.to_csv("/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/ConvertedQuasar/VIS_SD20_four_quasar_average_spectra.csv", index=False)
# -----------------------------
# Plot: Wavenumber with top wavelength axis
# -----------------------------
plt.figure(figsize=(9, 5), dpi=300)

for label in files.keys():
    sub = avg_all[avg_all["group"] == label]

    plt.plot(
        sub["wavenumber_cm-1"],
        sub["intensity"],
        linewidth=1.6,
        label=label
    )

plt.xlabel("Numero d'onda / cm$^{-1}$")
plt.ylabel("Riflettanza / u.a.")
plt.title("Spettri Visibile Medi: M1/M2 in Ambiente e in Vuoto")

ax = plt.gca()
ax.set_title(
    "Spettri Visibile Medi: M1/M2 in Ambiente e in Vuoto",
    fontweight="bold",
    pad=18
)
# Usual IR convention: high wavenumber on the left
ax.invert_xaxis()

# Conversion functions
def wn_to_nm(wn):
    return 10_000_000 / wn

def nm_to_wn(nm):
    return 10_000_000 / nm

# Secondary top axis: wavelength
secax = ax.secondary_xaxis(
    "top",
    functions=(wn_to_nm, nm_to_wn)
)

secax.set_xlabel("Lunghezza d'onda / nm")

plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    "/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/PLOTS/VIS_SD20_average_spectra_wavenumber_with_top_wavelength.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
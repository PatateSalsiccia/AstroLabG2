import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Input file
# -----------------------------
data_path = "/Users/alessandrodelmonte/Desktop/IR_all_samples_integrals_peaks_master_regions_only.csv"

df = pd.read_csv(data_path)

# -----------------------------
# Keep only integrals
# -----------------------------
integrals = df[df["measurement_type"] == "integral"].copy()

# Make sure values are numeric
integrals["value"] = pd.to_numeric(integrals["value"], errors="coerce")

# -----------------------------
# Calculate mean and SD
# -----------------------------
summary = (
    integrals
    .groupby(["group", "sample_id", "condition", "band", "range_cm-1"], as_index=False)
    .agg(
        mean_area=("value", "mean"),
        sd_area=("value", "std"),
        n=("value", "count")
    )
)

# Save summary table
#summary.to_csv("/Users/alessandrodelmonte/Desktop/IR_integral_mean_SD_summary.csv", index=False)
#print(summary)

# -----------------------------
# Plot mean ± SD for all bands
# -----------------------------
group_order = ["M1_ambient", "M1_vacuum", "M2_ambient", "M2_vacuum"]
band_order = [f"Band{i}" for i in range(1, 9)]

# More readable labels
label_map = {
    "M1_ambient": "M1 ambiente",
    "M1_vacuum": "M1 vuoto",
    "M2_ambient": "M2 ambiente",
    "M2_vacuum": "M2 vuoto",
}
# Create x-axis labels with band range
band_labels = (
    summary[["band", "range_cm-1"]]
    .drop_duplicates()
    .set_index("band")
    .loc[band_order]
)

x_labels = [
    f"{band}\n{band_labels.loc[band, 'range_cm-1']} cm$^{{-1}}$"
    for band in band_order
]
# Create one plot with all 8 bands
plt.figure(figsize=(12, 6), dpi=300)

x_positions = range(len(band_order))
bar_width = 0.18

for j, group in enumerate(group_order):
    sub = summary[summary["group"] == group].set_index("band").loc[band_order]

    x = [pos + (j - 1.5) * bar_width for pos in x_positions]

    plt.bar(
        x,
        sub["mean_area"],
        width=bar_width,
        yerr=sub["sd_area"],
        capsize=3,
        label=label_map[group]
    )

plt.xticks(list(x_positions), x_labels, rotation=45, ha="right")
plt.xlabel("Bande")
plt.ylabel("Area integrale / u. a.")
plt.title("Media ± DS dell'area integrale per M1 e M2 in Ambiente e in Vuoto.")

plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig("/Users/alessandrodelmonte/Desktop/DataAnalysis/Final/IR_integral_mean_SD_all_bands.png", dpi=300, bbox_inches="tight")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Input file
# -----------------------------
data_path = Path.home() / "Desktop" / "IR_all_samples_integrals_peaks_master_regions_only.csv"

# If the file is in the same folder as the script, you can use:
# data_path = "IR_all_samples_integrals_peaks_master_regions_only.csv"

df = pd.read_csv(data_path)

# -----------------------------
# Keep only peak depth data
# -----------------------------
df["value"] = pd.to_numeric(df["value"], errors="coerce")
data = df[df["measurement_type"] == "peak_depth"].copy()

# -----------------------------
# Output folder
# -----------------------------
output_dir = Path.home() / "Desktop" / "IR_heatmaps_peaks_allbands"
output_dir.mkdir(exist_ok=True)

# -----------------------------
# Order of groups and titles
# -----------------------------
groups = ["M1_ambient", "M1_vacuum", "M2_ambient", "M2_vacuum"]

group_titles = {
    "M1_ambient": "M1 ambient",
    "M1_vacuum": "M1 vacuum",
    "M2_ambient": "M2 ambient",
    "M2_vacuum": "M2 vacuum",
}

# -----------------------------
# Band order
# -----------------------------
band_order = [f"Band{i}" for i in range(1, 9)]

# Get one range per band
band_ranges = (
    data[["band", "range_cm-1"]]
    .drop_duplicates()
    .set_index("band")
)

# -----------------------------
# Make one figure per band
# -----------------------------
for band in band_order:
    band_range = band_ranges.loc[band, "range_cm-1"]

    band_data = data[data["band"] == band].copy()

    # Same color scale across all 4 panels for this band
    vmin = band_data["value"].min()
    vmax = band_data["value"].max()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=300, constrained_layout=True)

    for ax, group in zip(axes, groups):
        sub = band_data[band_data["group"] == group].copy()

        heatmap_data = (
            sub
            .pivot(index="grid_row", columns="grid_col", values="value")
            .sort_index(ascending=True)
        )

        im = ax.imshow(
            heatmap_data.values,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            origin="lower"
        )

        ax.set_title(group_titles[group], fontsize=10)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.set_yticklabels([1, 2, 3, 4, 5])
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")

        # Write values inside each square
        for r in range(5):
            for c in range(5):
                value = heatmap_data.values[r, c]
                ax.text(c, r, f"{value:.2g}", ha="center", va="center", fontsize=6)

    fig.suptitle(f"Peak Depth Heatmaps: {band} ({band_range} cm$^{{-1}}$)", fontsize=13)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Peak Depth / a.u.")

    filename = output_dir / f"IR_PeakDepth_heatmap_{band}_{band_range.replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

print(f"Peak Depth Heatmaps saved in:\n{output_dir}")
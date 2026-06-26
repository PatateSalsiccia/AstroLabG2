import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Input file
# -----------------------------
data_path = Path.home() / "Desktop" / "IR_all_samples_integrals_peaks_master_regions_only.csv"

df = pd.read_csv(data_path)

# -----------------------------
# Output folder
# -----------------------------
output_dir = Path.home() / "Desktop" / "IR_PCA_results"
output_dir.mkdir(exist_ok=True)

# -----------------------------
# Prepare data
# -----------------------------
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Use both integrals and peak depths
df["feature"] = df["band"] + "_" + df["measurement_type"]

# For PCA using only integrals instead, uncomment these two lines:
# df = df[df["measurement_type"] == "integral"].copy()
# df["feature"] = df["band"]

pca_data = df.pivot_table(
    index=[
        "sample_id",
        "condition",
        "composition",
        "spirulina_percent",
        "nacl_percent",
        "group",
        "region_number",
        "grid_row",
        "grid_col"
    ],
    columns="feature",
    values="value"
).reset_index()

pca_data.columns.name = None

feature_cols = [col for col in pca_data.columns if col.startswith("Band")]

X = pca_data[feature_cols].copy()

# Remove rows with missing values
valid_rows = X.dropna().index
X = X.loc[valid_rows]
pca_data_valid = pca_data.loc[valid_rows].reset_index(drop=True)

# -----------------------------
# Standardize manually
# -----------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)

X_scaled = (X - X_mean) / X_std

# -----------------------------
# PCA using SVD
# -----------------------------
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

scores = U[:, :2] * S[:2]
loadings_matrix = Vt[:2, :].T

explained_variance = (S ** 2) / (len(X_scaled) - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()
explained = explained_variance_ratio[:2] * 100

pca_data_valid["PC1"] = scores[:, 0]
pca_data_valid["PC2"] = scores[:, 1]

print(f"PC1 explains {explained[0]:.2f}% of variance")
print(f"PC2 explains {explained[1]:.2f}% of variance")

# Save PCA scores
scores_path = output_dir / "IR_PCA_scores.csv"
pca_data_valid.to_csv(scores_path, index=False)

# Save PCA loadings
loadings = pd.DataFrame(
    loadings_matrix[:, :2],
    index=feature_cols,
    columns=["PC1_loading", "PC2_loading"]
).reset_index()

loadings = loadings.rename(columns={"index": "feature"})

loadings_path = output_dir / "IR_PCA_loadings.csv"
loadings.to_csv(loadings_path, index=False)

# -----------------------------
# PCA score plot
# -----------------------------
groups = ["M1_ambient", "M1_vacuum", "M2_ambient", "M2_vacuum"]

label_map = {
    "M1_ambient": "M1 ambient",
    "M1_vacuum": "M1 vacuum",
    "M2_ambient": "M2 ambient",
    "M2_vacuum": "M2 vacuum",
}

plt.figure(figsize=(7, 6), dpi=300)

for group in groups:
    sub = pca_data_valid[pca_data_valid["group"] == group]

    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        label=label_map[group],
        s=45,
        alpha=0.8
    )

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
plt.title("PCA of regional IR band integrals and peak depths")

plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(output_dir / "IR_PCA_scores_by_group.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# PCA score plot with region numbers
# -----------------------------
plt.figure(figsize=(8, 6), dpi=300)

for group in groups:
    sub = pca_data_valid[pca_data_valid["group"] == group]

    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        label=label_map[group],
        s=45,
        alpha=0.8
    )

    for _, row in sub.iterrows():
        plt.text(
            row["PC1"],
            row["PC2"],
            str(int(row["region_number"])),
            fontsize=6,
            ha="center",
            va="center"
        )

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
plt.title("PCA of regional IR band values with region numbers")

plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(output_dir / "IR_PCA_scores_by_group_with_region_numbers.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# PCA loadings plot
# -----------------------------
plt.figure(figsize=(8, 6), dpi=300)

plt.scatter(
    loadings["PC1_loading"],
    loadings["PC2_loading"],
    s=45
)

for _, row in loadings.iterrows():
    plt.text(
        row["PC1_loading"],
        row["PC2_loading"],
        row["feature"],
        fontsize=7,
        ha="center",
        va="bottom"
    )

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("PC1 loading")
plt.ylabel("PC2 loading")
plt.title("PCA loadings: contribution of each band feature")

plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(output_dir / "IR_PCA_loadings.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved PCA files in:")
print(output_dir)
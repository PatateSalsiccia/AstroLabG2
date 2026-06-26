import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# -----------------------------
# Input files
# -----------------------------
amb_path = Path.home() / "Desktop" / "IR_gaussian_halfwidth" / "input"/ "IR_M1_P-amb_ALL_25_regions_only.csv"
vac_path = Path.home() / "Desktop" / "IR_gaussian_halfwidth" / "input"/ "IR_M1_P-vac_ALL_25_regions_only.csv"

# If the files are in the same folder as your script, use:
# amb_path = "IR_M1_P-amb_ALL_25_regions_only.csv"
# vac_path = "IR_M1_P-vac_ALL_25_regions_only.csv"

# -----------------------------
# Output folder
# -----------------------------
output_dir = Path.home() / "Desktop" / "IR_gaussian_halfwidth" / "output"/ "IR_avg_ambient_vs_vacuum"
output_dir.mkdir(exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
amb = pd.read_csv(amb_path)
vac = pd.read_csv(vac_path)

amb = amb.rename(columns={amb.columns[0]: "wavenumber_cm-1"})
vac = vac.rename(columns={vac.columns[0]: "wavenumber_cm-1"})

region_cols_amb = [c for c in amb.columns if c.startswith("region_")]
region_cols_vac = [c for c in vac.columns if c.startswith("region_")]

# Compute average and SD across the 25 regions
amb["mean_intensity"] = amb[region_cols_amb].mean(axis=1)
amb["sd_intensity"] = amb[region_cols_amb].std(axis=1)

vac["mean_intensity"] = vac[region_cols_vac].mean(axis=1)
vac["sd_intensity"] = vac[region_cols_vac].std(axis=1)

# -----------------------------
# Band definitions
# -----------------------------
bands = {
    "Band1": (4010, 4192),
    "Band2": (4192, 4488),
    "Band3": (4488, 4759),
    "Band4": (4759, 4956),
    "Band5": (4956, 5414),
    "Band6": (5414, 6065),
    "Band7": (6065, 7568),
    "Band8": (7568, 8980),
}

# -----------------------------
# Gaussian + linear baseline model
# -----------------------------
def gaussian_with_linear_baseline(x, m, b, A, center, sigma):
    return m * x + b + A * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def fit_gaussian_band(x, y, band_start, band_end):
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]

    width = band_end - band_start

    # Initial baseline estimate from edges
    n_edge = max(3, int(0.1 * len(x)))
    x_edges = np.concatenate([x[:n_edge], x[-n_edge:]])
    y_edges = np.concatenate([y[:n_edge], y[-n_edge:]])
    m0, b0 = np.polyfit(x_edges, y_edges, 1)

    baseline0 = m0 * x + b0
    residual = y - baseline0

    idx = np.argmax(np.abs(residual))
    A0 = residual[idx]
    center0 = x[idx]
    sigma0 = width / 6

    lower = [-np.inf, -np.inf, -np.inf, band_start, width / 200]
    upper = [ np.inf,  np.inf,  np.inf, band_end,   width]

    try:
        popt, _ = curve_fit(
            gaussian_with_linear_baseline,
            x,
            y,
            p0=[m0, b0, A0, center0, sigma0],
            bounds=(lower, upper),
            maxfev=20000
        )

        y_fit = gaussian_with_linear_baseline(x, *popt)
        residuals = y - y_fit

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        m, b, A, center, sigma = popt
        sigma = abs(sigma)
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        hwhm = fwhm / 2

        return {
            "success": True,
            "center_cm-1": center,
            "sigma_cm-1": sigma,
            "FWHM_cm-1": fwhm,
            "HWHM_cm-1": hwhm,
            "R2": r2,
            "x_fit": x,
            "y_fit": y_fit,
        }

    except Exception as e:
        return {
            "success": False,
            "center_cm-1": np.nan,
            "sigma_cm-1": np.nan,
            "FWHM_cm-1": np.nan,
            "HWHM_cm-1": np.nan,
            "R2": np.nan,
            "x_fit": x,
            "y_fit": np.full_like(x, np.nan, dtype=float),
        }

# -----------------------------
# Plot 1: full averaged spectra
# -----------------------------
plt.figure(figsize=(9, 5), dpi=300)

plt.plot(
    amb["wavenumber_cm-1"],
    amb["mean_intensity"],
    linewidth=1.8,
    label="Ambient average"
)

plt.plot(
    vac["wavenumber_cm-1"],
    vac["mean_intensity"],
    linewidth=1.8,
    label="Vacuum average"
)

# Optional shaded SD
plt.fill_between(
    amb["wavenumber_cm-1"],
    amb["mean_intensity"] - amb["sd_intensity"],
    amb["mean_intensity"] + amb["sd_intensity"],
    alpha=0.2
)

plt.fill_between(
    vac["wavenumber_cm-1"],
    vac["mean_intensity"] - vac["sd_intensity"],
    vac["mean_intensity"] + vac["sd_intensity"],
    alpha=0.2
)

plt.xlabel("Wavenumber / cm$^{-1}$")
plt.ylabel("Intensity / a.u.")
plt.title("M1 average spectra: ambient vs vacuum")
plt.gca().invert_xaxis()
plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(output_dir / "IR_M1_avg_full_ambient_vs_vacuum.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# Plot 2: band-by-band comparison
# -----------------------------
fig, axes = plt.subplots(4, 2, figsize=(10, 12), dpi=300)
axes = axes.flatten()

fit_summary = []

for ax, (band, (start, end)) in zip(axes, bands.items()):
    amb_mask = (amb["wavenumber_cm-1"] >= start) & (amb["wavenumber_cm-1"] <= end)
    vac_mask = (vac["wavenumber_cm-1"] >= start) & (vac["wavenumber_cm-1"] <= end)

    x_amb = amb.loc[amb_mask, "wavenumber_cm-1"].values
    y_amb = amb.loc[amb_mask, "mean_intensity"].values
    sd_amb = amb.loc[amb_mask, "sd_intensity"].values

    x_vac = vac.loc[vac_mask, "wavenumber_cm-1"].values
    y_vac = vac.loc[vac_mask, "mean_intensity"].values
    sd_vac = vac.loc[vac_mask, "sd_intensity"].values

    # Plot data
    order_amb = np.argsort(x_amb)
    order_vac = np.argsort(x_vac)

    ax.plot(x_amb[order_amb], y_amb[order_amb], linewidth=1.5, label="Ambient")
    ax.plot(x_vac[order_vac], y_vac[order_vac], linewidth=1.5, label="Vacuum")

    ax.fill_between(
        x_amb[order_amb],
        y_amb[order_amb] - sd_amb[order_amb],
        y_amb[order_amb] + sd_amb[order_amb],
        alpha=0.2
    )

    ax.fill_between(
        x_vac[order_vac],
        y_vac[order_vac] - sd_vac[order_vac],
        y_vac[order_vac] + sd_vac[order_vac],
        alpha=0.2
    )

    # Fit Gaussian to average curves
    fit_amb = fit_gaussian_band(x_amb, y_amb, start, end)
    fit_vac = fit_gaussian_band(x_vac, y_vac, start, end)

    ax.plot(fit_amb["x_fit"], fit_amb["y_fit"], linestyle="--", linewidth=1.2)
    ax.plot(fit_vac["x_fit"], fit_vac["y_fit"], linestyle="--", linewidth=1.2)

    ax.set_title(
        f"{band} ({start}-{end} cm$^{{-1}}$)\n"
        f"Amb FWHM={fit_amb['FWHM_cm-1']:.1f}, Vac FWHM={fit_vac['FWHM_cm-1']:.1f}",
        fontsize=9
    )

    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel("Intensity / a.u.")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()

    fit_summary.append({
        "band": band,
        "range_cm-1": f"{start}-{end}",
        "ambient_center_cm-1": fit_amb["center_cm-1"],
        "ambient_FWHM_cm-1": fit_amb["FWHM_cm-1"],
        "ambient_HWHM_cm-1": fit_amb["HWHM_cm-1"],
        "ambient_R2": fit_amb["R2"],
        "vacuum_center_cm-1": fit_vac["center_cm-1"],
        "vacuum_FWHM_cm-1": fit_vac["FWHM_cm-1"],
        "vacuum_HWHM_cm-1": fit_vac["HWHM_cm-1"],
        "vacuum_R2": fit_vac["R2"],
    })

axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(output_dir / "IR_M1_avg_bandwise_ambient_vs_vacuum.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# Save fit summary
# -----------------------------
fit_summary_df = pd.DataFrame(fit_summary)
fit_summary_df.to_csv(output_dir / "IR_M1_avg_ambient_vs_vacuum_gaussian_summary.csv", index=False)

print("Saved files in:")
print(output_dir)
print("\nFit summary:")
print(fit_summary_df)
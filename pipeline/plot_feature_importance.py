import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_coefficients(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"feature", "coefficient", "abs_coeff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"coefficients CSV missing columns: {missing}")
    df = df.sort_values("abs_coeff", ascending=False).reset_index(drop=True)
    return df


def plot_signed_bars(df: pd.DataFrame, out_path: Path, top: int = 0) -> None:
    """
    Plot signed coefficients with color and smart label placement:
    - Blue = negative (retention signal)
    - Red = positive (churn risk)
    - Label placed inside bar if |coef| >= 0.5, else outside
    """
    if top > 0:
        df = df.head(top)

    # Assign colors: red (positive → churn↑), blue (negative → churn↓)
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in df["coefficient"]]

    plt.figure(figsize=(8, max(4, 0.4 * len(df))))
    bars = plt.barh(df["feature"], df["coefficient"], color=colors)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="gray", linewidth=1)
    plt.xlabel("Coefficient (→ higher churn risk)")
    plt.title("Feature Impact on Churn Probability")
    plt.tight_layout()

    # Add labels for every bar with dynamic placement
    for i, (coef, bar) in enumerate(zip(df["coefficient"], bars)):
        label = f"{coef:.2f}"
        width = bar.get_width()
        inside = abs(coef) >= 0.3
        if coef > 0:
            # Positive side
            x = width - 0.02 if inside else width + 0.02
            ha = "right" if inside else "left"
            color = "white" if inside else "black"
        else:
            # Negative side
            x = width + 0.02 if inside else width - 0.02
            ha = "left" if inside else "right"
            color = "white" if inside else "black"
        plt.text(
            x, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha=ha, fontsize=9, color=color
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved feature importance → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot feature importance with sign-aware colors and smart labels.")
    parser.add_argument("--coef_csv", required=True, help="Path to coefficients.csv")
    parser.add_argument("--out", default="docs/feature_importance.png", help="Output PNG path")
    parser.add_argument("--top", type=int, default=0, help="Plot only top N features by |coef| (0 = all)")
    args = parser.parse_args()

    df = load_coefficients(Path(args.coef_csv))
    plot_signed_bars(df, Path(args.out), top=args.top)


if __name__ == "__main__":
    main()

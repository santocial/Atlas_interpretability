"""SHAP plot: top regions (bar) and distribution of all values (histogram)."""
import matplotlib

matplotlib.use("Agg")    # save to file, no display needed (e.g. on a cluster)
import matplotlib.pyplot as plt


def plot_region_shap(table, title, save_path, class_name="target class", top_n=15):
    """`table` is the DataFrame from explain_regions (sorted by |SHAP|)."""
    top = table.head(top_n)
    colors = ["blue" if v > 0 else "red" for v in top["shap_value"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(top["region_name"], top["shap_value"], color=colors, alpha=0.7)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"SHAP value (probability of {class_name})")
    axes[0].set_title(f"Top {top_n} regions\nBlue = pushes toward {class_name}, "
                      f"Red = pushes away")

    axes[1].hist(table["shap_value"], bins=30, color="steelblue", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel(f"SHAP value (probability of {class_name})")
    axes[1].set_ylabel("Number of regions")
    axes[1].set_title("All regions")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

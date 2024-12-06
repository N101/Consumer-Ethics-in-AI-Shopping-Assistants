import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


YLIM = (0, 5.9)


def make_graphs(
    df: pd.DataFrame, slices: list, labels: list, errors: pd.DataFrame, prefix: str
) -> list[plt.Figure]:
    images = []
    capsizes = {"downloading": 7, "passive": 3}
    default_capsize = 4

    def get_capsize(key):
        return capsizes.get(key, default_capsize)
    
    for sl, lbl in zip(slices, labels):
        fig, ax = plt.subplots()
        df.iloc[sl].plot(
            kind="bar",
            ylim=YLIM,
            yerr=errors,
            capsize=get_capsize(lbl),
            ecolor="darkred",
            color=["#2ca02c", "#4682b4", "#5a9bd4"],
            ax=ax,
            title=lbl,
            xlabel="Question",
            ylabel="Avg Score",
            figsize=(10, 5),
            rot=0,
        ).legend([f"{prefix}", "Students", "Non-students"])
        fig.tight_layout()
        images.append(fig)
    return images


def make_heatmap(df: pd.DataFrame, prefix: str) -> plt.Figure:
    fig, ax = plt.subplots()
    pivot_table = df.pivot_table(values="Response", index="Iteration", columns="#")
    sns.heatmap(pivot_table, cmap="viridis", cbar_kws={"label": "Responses"}, ax=ax)
    ax.set_title(f"{prefix} Heatmap")
    fig.tight_layout()
    return fig

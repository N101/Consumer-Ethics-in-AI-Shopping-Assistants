import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


YLIM = (0, 5.9)
SUFFIX = ""


def make_graphs(
    df: pd.DataFrame, slices: list, labels: list, errors: pd.DataFrame, suffix: str
) -> list[plt.Figure]:
    SUFFIX = suffix
    images = []
    for sl, lbl in zip(slices, labels):
        fig, ax = plt.subplots()
        df.iloc[sl].plot(
            kind="bar",
            ylim=YLIM,
            yerr=errors,
            capsize=3,
            ecolor="darkred",
            color=["#2ca02c", "#4682b4", "#5a9bd4"],
            ax=ax,
            title=lbl,
            xlabel="Question",
            ylabel="Avg Score",
            figsize=(10, 5),
            rot=0,
        ).legend([f"{SUFFIX}", "Students", "Non-students"])
        fig.tight_layout()
        images.append(fig)
    return images


def make_heatmap(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    pivot_table = df.pivot_table(values="Response", index="Iteration", columns="#")
    sns.heatmap(pivot_table, cmap="viridis", cbar_kws={"label": "Responses"}, ax=ax)
    ax.set_title(f"Heatmap {SUFFIX}")
    fig.tight_layout()
    return fig

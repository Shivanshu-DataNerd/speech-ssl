import os
import matplotlib.pyplot as plt


def save_plot(fig, filename, dpi=300, show=True):
    """
    Save matplotlib figure to graphs folder.

    Args:
        fig : matplotlib.figure.Figure
        filename : str
        dpi : int
        show : bool (display plot or not)
    """

    # project root = one level above src
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    graph_dir = os.path.join(PROJECT_ROOT, "graphs")

    os.makedirs(graph_dir, exist_ok=True)

    save_path = os.path.join(graph_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved graph → {save_path}")

    if show:
        plt.show()

    plt.close(fig)

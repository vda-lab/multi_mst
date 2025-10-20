import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from statsmodels.nonparametric.smoothers_lowess import lowess


def _draw(xs, ys, sources, targets, color, size=0.3, alpha=0.2, s=1.7438):
    plt.figure(figsize=(s, s))
    s = plt.scatter(
        xs,
        ys,
        c=color,
        s=size,
        edgecolors="none",
        linewidth=0,
        cmap="viridis",
        alpha=alpha,
    )
    lc = mc.LineCollection(
        list(zip(zip(xs[sources], ys[sources]), zip(xs[targets], ys[targets]))),
        linewidth=0.2,
        zorder=-1,
        alpha=0.5,
        color="k",
    )
    ax = plt.gca()
    ax.add_collection(lc)
    plt.subplots_adjust(0, 0, 1, 1)
    ax.autoscale()
    ax.set_aspect("equal")
    plt.axis("off")


def draw_graph(p, x, y, color=None, name="default", alg="umap"):
    coo = p.graph_.tocoo()
    sources = coo.row
    targets = coo.col
    _draw(x, y, sources, targets, color)
    plt.savefig(f"./images/{name}_top_{alg}.png", dpi=600, pad_inches=0)
    plt.show()


def draw_umap(p, color=None, name="default", alg="umap", size=0.3, alpha=0.2):
    coo = p.graph_.tocoo()
    sources = coo.row
    targets = coo.col
    _draw(
        p.embedding_[:, 0],
        p.embedding_[:, 1],
        sources,
        targets,
        color,
        size=size,
        alpha=alpha,
    )
    plt.savefig(f"./images/{name}_umap_{alg}.png", dpi=600, pad_inches=0)
    plt.show()


def draw_force(p, color=None, name="default", alg="umap"):
    coords, g = compute_force(p)
    sources, targets = [np.asarray(x) for x in zip(*g.edges())]
    _draw(coords[:, 0], coords[:, 1], sources, targets, color)
    plt.savefig(f"./images/{name}_force_{alg}.png", dpi=600, pad_inches=0)
    plt.show()


def compute_force(p):
    g = nx.Graph(p.graph_)
    pos = nx.nx_agraph.graphviz_layout(g, prog="sfdp")
    coords = np.nan * np.ones((p.graph_.shape[0], 2), dtype=np.float64)
    for k, v in pos.items():
        coords[k, :] = v
    return coords, g


def regplot_lowess_ci(
    data,
    x,
    y,
    *,
    ci_level,
    n_boot,
    lowess_frac=0.1,
    color="C0",
    scatter=True,
    line_kws=None,
    area_kws=None,
    scatter_kws=None,
):
    x_ = data[x].to_numpy()
    y_ = data[y].to_numpy()
    x_grid = np.linspace(start=x_.min(), stop=x_.max(), num=1000)

    def reg_func(_x, _y):
        return lowess(
            exog=_x, endog=_y, xvals=x_grid, is_sorted=False, frac=lowess_frac
        )

    beta_boots = sns.algorithms.bootstrap(
        x_,
        y_,
        func=reg_func,
        n_boot=n_boot,
    )
    err_bands = sns.utils.ci(beta_boots, ci_level, axis=0)
    y_plt = reg_func(x_, y_)

    ax = sns.lineplot(x=x_grid, y=y_plt, color=color, **(line_kws or {}))
    ax.fill_between(x_grid, *err_bands, alpha=0.15, color=color, **(area_kws or {}))
    if scatter:
        sns.scatterplot(x=x_, y=y_, ax=ax, color=color, **(scatter_kws or {}))
    return ax

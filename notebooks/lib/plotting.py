import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import dedent

# LaTeX font sizes on 10pt document:
# https://latex-tutorial.com/changing-font-size/
fontsize = dict(tiny=5, script=7, footnote=8, small=9, normal=10)


def configure_matplotlib():
    sns.set_style("white")
    sns.set_color_codes()

    mpl.rcParams.update(
        {
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "xtick.bottom": True,
            "ytick.left": True,
            "axes.titlesize": fontsize["normal"],
            "axes.labelsize": fontsize["small"],
            "xtick.labelsize": fontsize["small"],
            "ytick.labelsize": fontsize["small"],
            "font.size": fontsize["footnote"],
            "legend.title_fontsize": fontsize["footnote"],
            "legend.fontsize": fontsize["footnote"],
            "axes.unicode_minus": True,
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.bottom": False,
            "savefig.dpi": 300,
            "savefig.format": "png",
            "font.family": "serif",
            "text.usetex": True,
            "text.latex.preamble": dedent(
                """
                \\usepackage{libertine}
                \\renewcommand\\sfdefault{ppl}
                """
            ),
        }
    )

    return sns.color_palette("tab10", 10)


def sized_fig(width=0.5, aspect=0.618):
    """Create a figure with width as fraction of an A4 page."""
    page_width_cm = 13.9
    inch = 2.54
    w = width * page_width_cm
    h = aspect * w
    return plt.figure(figsize=(w / inch, h / inch), dpi=150)


def size_fig(width=0.5, aspect=0.618):
    page_width_cm = 13.9
    inch = 2.54
    w = width * page_width_cm
    h = aspect * w

    fig = plt.gcf()
    fig.set_dpi(150)
    fig.set_figwidth(w / inch)
    fig.set_figheight(h / inch)


def frame_off():
    """Disables frames and ticks, sets aspect ratio to 1."""
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect(1)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def adjust_legend_subtitles(legend):
    """
    Make invisible-handle "subtitles" entries look more like titles.
    Adapted from seaborn.utils.
    """
    # Legend title not in rcParams until 3.0
    font_size = plt.rcParams.get("legend.title_fontsize", None)
    vpackers = legend.findobj(mpl.offsetbox.VPacker)
    for vpacker in vpackers[:-1]:
        hpackers = vpacker.get_children()
        for hpack in hpackers:
            draw_area, text_area = hpack.get_children()
            handles = draw_area.get_children()
            if not all(artist.get_visible() for artist in handles):
                draw_area.set_width(0)
                for text in text_area.get_children():
                    if font_size is not None:
                        text.set_size(font_size)

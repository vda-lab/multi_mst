import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def configure_matplotlib():   
    sns.set_style('white')
    sns.set_color_codes()
    
    mpl.rcParams['text.color'] = 'black'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['ytick.left'] = True
    mpl.rcParams['ytick.major.size'] = mpl.rcParams['xtick.major.size']
    mpl.rcParams['ytick.major.width'] = mpl.rcParams['xtick.major.width']    
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['legend.title_fontsize'] = 6
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['axes.unicode_minus'] = True
    mpl.rcParams['axes.spines.left'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = False
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['savefig.format'] = 'png'
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = """
    \\usepackage{libertine}
    \\renewcommand\\sfdefault{ppl}
    """
    
    return sns.color_palette('tab10', 10)


def sized_fig(width=0.5, aspect=0.618):
    """Create a figure with width as fraction of an A4 page."""
    page_width_cm = 13.9
    inch = 2.54
    w = (width * page_width_cm)
    h = aspect * w
    return plt.figure(figsize=(w / inch, h / inch), dpi=150)


def size_fig(width=0.5, aspect=0.618):
    page_width_cm = 13.9
    inch = 2.54
    w = (width * page_width_cm)
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
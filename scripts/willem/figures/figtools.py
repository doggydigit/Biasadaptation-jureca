from datarep.matplotlibsettings import *


LABELS = {"rpw": [r""+"Task-specific \nnetwork", "None", "NA", r"$W_1$, $\mathbf{b}_1$,"+ "\n$\mathbf{w}_o$, $b_o$", "gradient descent \nsingle task"],
          "br": [r""+"Task-specific \nreadout", r"$W_1$, $\mathbf{b}_1$", "gradient descent \nmulti-task", r"$\mathbf{w}_o$, $b_o$", "gradient descent\nsingle task"],
          "code": [r"SC", r"$W_1$", "sparse dict code", r"$\mathbf{w}_o$, $b_o$", "gradient descent\nsingle task"],
          "sc": [r"SD", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ sparse dict " + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "pca": [r"PCA", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ PCA" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "rp": [r"RP", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ random" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "scd": [r"$\Delta$SD", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ sparse dict $\Delta \mathbf{x}$" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "pmdd": [r"$\Delta$PMD", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ " + "penalized matrix \n" + r"      decomposition $\Delta \mathbf{x}$" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "bpo": [r""+"Task-specific \nbiases", r"$W_1$, $\mathbf{w}_o$", "gradient descent \nmulti-task", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
         }

COLOURS = { "rpw": colours[8],
            "br": colours[9],
            "code": colours[1],
            "sc": colours[2],
            "pca": colours[3],
            "rp": colours[7],
            "scd": colours[4],
            "pmdd": colours[0],
            "bpo": colours[5],

}


def perfAx(ax, add_xticklabels=True, xlabel=None,
               add_yticklabels=True,
               ylim=(50.,100.), skip_yt=True, add_ylabel=True,
               nhs=[[10], [25], [50], [100], [250], [500]],
               xticks=None):
    ax = myAx(ax)
    # rect = Rectangle((0., 95.), len(nhs), 5., color='DarkGrey', alpha=.2, zorder=-1000)
    # ax.add_patch(rect)
    ax.axhline(100, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1000)
    ax.axhline(90, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1001)
    ax.axhline(80, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1002)
    ax.axhline(70, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1003)
    ax.axhline(60, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1004)

    ax.axhline(95, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1005)
    ax.axhline(85, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1006)
    ax.axhline(75, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1007)
    ax.axhline(65, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1008)

    ax.set_ylim(ylim)
    if xticks is None:
      ax.set_xticks(np.arange(len(nhs))+.5)
      ax.set_xlim((0., len(nhs)))
    else:
      ax.set_xticks(xticks)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)

    if add_xticklabels:
        xtl = ["[%s]"%(", ".join([str(nh) for nh in nh_])) for nh_ in nhs]
        ax.set_xticklabels(nhs, rotation=60, fontsize=ticksize)
    else:
        ax.set_xticklabels([])

    # y axis

    y0 = int(np.ceil(ylim[0] / 10.) * 10.)
    yt = np.arange(y0, ylim[1]+1e-4, 10.).astype(int)

    ax.set_yticks(yt)

    if add_ylabel:
        ax.set_ylabel(r'test perf (%)', fontsize=labelsize)

    if add_yticklabels:
        if skip_yt:
          ytl = [str(yt[ii]) if ii%2==0 else "" for ii in range(len(yt))]
        else:
          ytl = [str(yt[ii]) for ii in range(len(yt))]
        ax.set_yticklabels(ytl, fontsize=ticksize)
    else:
        ax.set_yticklabels([])

    return ax
from qutip import Qobj
import seaborn as sns
from numpy import real
from matplotlib import pyplot as plt

sns.set_theme(rc={"figure.figsize": (8, 6)})


def matrixplot(
    oper, annot=False, xlabels="auto", ylabels="auto", axs=None, figsize=(2 * 8.4, 6.72)
):
    if type(oper) is Qobj:
        oper = oper.full()
    if axs:
        [ax1, ax2] = axs
    else:
        fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=figsize, dpi=100)
        ax1.xaxis.tick_top()
    sns.heatmap(
        oper.real,
        annot=annot,
        cmap="RdBu",
        center=0,
        xticklabels=xlabels,
        yticklabels=ylabels,
        ax=ax1,
        square=not annot,
        annot_kws={"fontsize": "small", "fontstretch": "condensed"},
    )
    ax1.set_title("Real")
    ax2.xaxis.tick_top()
    sns.heatmap(
        oper.imag,
        annot=annot,
        cmap="PRGn",
        center=0,
        xticklabels=xlabels,
        yticklabels=ylabels,
        ax=ax2,
        square=not annot,
        annot_kws={"fontsize": "small", "fontstretch": "condensed"},
    )
    ax2.set_title("Imag")
    plt.tight_layout()
    return fig, [ax1, ax2]


def index_to_F_mF_string_D1(ind):
    if ind < 3:
        return rf"""$F=1, m={ind - 1}$"""
    elif ind < 8:
        return rf"""$F=2, m={ind - 2 - 3}$"""
    elif ind < 8 + 3:
        return rf"""$F'=1, m={ind - 1 - 8}$"""
    else:
        return rf"""$F'=2, m={ind - 2 - 8 - 3}$"""


def index_to_F_mF_string_D2(ind):
    if ind < 3:
        return rf"""$F=1, m={ind - 1}$"""
    elif ind < 8:
        return rf"""$F=2, m={ind - 2 - 3}$"""
    elif ind == 8:
        return rf"""$F'=0, m={ind - 8}$"""
    elif ind in (9, 10, 11):
        return rf"""$F'=1, m={ind - 10}$"""
    elif ind in (12, 13, 14, 15, 16):
        return f"""$F'=2, m={ind - 14}$"""
    else:
        return f"""$F'=3, m={ind - 20}$"""


def maplot(
    op: Qobj, std_xlabels=True, std_ylabels=True, annot=False, figsize=(2 * 8.4, 6.72)
):
    if op.shape[0] == 16:
        fig, axs = matrixplot(
            op,
            xlabels=[index_to_F_mF_string_D1(ind) for ind in range(op.shape[0])]
            if std_xlabels
            else "auto",
            ylabels=[index_to_F_mF_string_D1(ind) for ind in range(op.shape[0])]
            if std_ylabels
            else "auto",
            annot=annot,
            figsize=figsize,
        )
        return fig, axs
    else:
        fig, axs = matrixplot(
            op,
            xlabels=[index_to_F_mF_string_D2(ind) for ind in range(op.shape[0])]
            if std_xlabels
            else "auto",
            ylabels=[index_to_F_mF_string_D2(ind) for ind in range(op.shape[0])]
            if std_ylabels
            else "auto",
            annot=annot,
            figsize=figsize,
        )
        return fig, axs


def plot_ground_states_time(res_, axs=None):
    dimension = res_.states[0].shape[0]
    ground_exp_val_ = [
        [res_.states[t].diag()[k] for t in range(len(res_.times))] for k in range(8)
    ]
    if axs is not None:
        fig = axs[0, 0].get_figure()
    else:
        fig, axs = plt.subplots(
            ncols=5, nrows=2, figsize=(12, 6), sharex="all", sharey="all"
        )
    for i, e in enumerate(ground_exp_val_[:3]):
        axs[1, 1 + i].plot(res_.times, real(e))
    for i, e in enumerate(ground_exp_val_[3:8]):
        axs[0, i].plot(res_.times, real(e))
    fig.suptitle("Ground States")
    axs[1, 2].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig, axs


def plot_excited_states_time(res_, axs=None):
    dimension = res_.states[0].shape[0]
    excited_exp_val_ = [
        [res_.states[t].diag()[k] for t in range(len(res_.times))]
        for k in range(8, dimension)
    ]

    if dimension == 16:
        if axs is not None:
            fig = axs[0, 0].get_figure()
        else:
            fig, axs = plt.subplots(
                ncols=5, nrows=2, figsize=(12, 6), sharex="all", sharey="all"
            )
        for i, e in enumerate(excited_exp_val_[:3]):
            axs[1, 1 + i].plot(res_.times, real(e))
        for i, e in enumerate(excited_exp_val_[3:8]):
            axs[0, i].plot(res_.times, real(e))
        fig.suptitle("Excited States")
        axs[1, 2].set_xlabel("Time (s)")
        plt.tight_layout()
        return fig, axs
    elif dimension == 24:
        if axs is not None:
            fig = axs[0, 0].get_figure()
        else:
            fig, axs = plt.subplots(
                ncols=7, nrows=4, figsize=(12, 12), sharex="all", sharey="all"
            )
        for i, e in enumerate(excited_exp_val_[-7:]):
            axs[0, i].plot(res_.times, real(e))
        for i, e in enumerate(excited_exp_val_[-12:-7]):
            axs[1, 1 + i].plot(res_.times, real(e))
        for i, e in enumerate(excited_exp_val_[1:4]):
            axs[2, 2 + i].plot(res_.times, real(e))
        axs[3, 3].plot(res_.times, real(excited_exp_val_[0]))
        fig.suptitle("Excited States")
        axs[3, 3].set_xlabel("Time (s)")
        plt.tight_layout()
        return fig, axs
    else:
        raise ValueError


def plot_total_ground_pop(resa):
    fig, ax = plt.subplots()
    ax.plot(resa.times, [sum(elem.diag()[:8]) for elem in resa.states], label="Ground")
    ax.set_ylabel("Total Ground state population")
    ax.set_xlabel("time (s)")
    # ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_bar_ground_pop(rho):
    # populations plot
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharey="all", sharex="all")
    ax2.bar(list(range(-1, 2)), rho.diag().real[:3], color="tab:blue")
    ax1.set_title("F=2")
    ax1.bar(list(range(-2, 3)), rho.diag().real[3:8], color="tab:blue")
    ax2.set_title("F=1")
    fig.suptitle("Ground States")
    plt.tight_layout()
    return fig, [ax1, ax2]


def plot_bar_excited_pop_D1(rho):
    # populations plot
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharey="all", sharex="all")
    ax2.bar(list(range(-1, 2)), rho.diag().real[8 : 3 + 8], color="tab:blue")
    ax1.set_title("F=2")
    ax1.bar(list(range(-2, 3)), rho.diag().real[3 + 8 : 8 + 8], color="tab:blue")
    ax2.set_title("F=1")
    fig.suptitle("Excited States")
    plt.tight_layout()
    return fig, [ax1, ax2]

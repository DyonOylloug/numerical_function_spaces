import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar
if __name__ == '__main__':
    from norms import *  # kappa
else:
    from .norms import *  # kappa

# my_path = os.path.dirname(
#     os.path.abspath(__file__)
# )  # for save plots
# powyższe podaje ścieżkę do modułu czy do uruchomionego pliku?
# mpl.style.use("default")  # for work with plots on linux via ssh
# print('bieżący katalog: ', my_path)
# mpl.style.use("grayscale")  # do czarno białych wykresów
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["text.usetex"] = True  # for latex on plots


def plot_save(name: str = 'plot', p_norm: float = ''):
    """
    Saves the current figure in different formats (PNG, SVG, PDF) with a given name and p_norm.

    Parameters
    ----------
    name : str
        The name to be used for the saved files. Default is 'plot'.
    p_norm : float
        The p-norm value to be included in the file names. Default is an empty string.

    Returns
    -------
    The function saves the figure in folder 'plots' : None
    """
    my_path = os.getcwd()
    if not os.path.exists(my_path + '/plots'):
        os.makedirs(my_path + '/plots')
        print(" ! New folder \"plots\" - for plots ")
    else:
        print("Folder \"plots\" ")
    if not os.path.exists(my_path + '/plots/pdf'):
        os.makedirs(my_path + '/plots/pdf')
        print(" ! New folder \"plots\\pdf\" - for plots in pdf format")
    if not os.path.exists(my_path + '/plots/svg'):
        os.makedirs(my_path + '/plots/svg')
        print(" ! New folder \"plots\\svg\" - for plots in svg format")
    if not os.path.exists(my_path + '/plots/png'):
        os.makedirs(my_path + '/plots/png')
        print(" ! New folder \"plots\\png\" - for plots in png format ")

    plt.savefig(my_path + f"/plots/png/{name}{'_' if p_norm != '' else ''}{p_norm}.png", dpi=1200)
    plt.savefig(my_path + f"/plots/svg/{name}{'_' if p_norm != '' else ''}{p_norm}.svg")
    plt.savefig(my_path + f"/plots/pdf/{name}{'_' if p_norm != '' else ''}{p_norm}.pdf")


def description_for_plot(p_norm: float):
    """
    Set description for plots

    Parameters
    ----------
    p_norm : float
        The p-norm value to be included in plot description
    """
    if p_norm == 1:
        opis = (
            "$\\frac{1}{k}\\left(1+I_{\\Phi}(k\\, x) \\right)$"
        )
    elif p_norm == np.inf:
        opis = (
            "$\\frac{1}{k} \\max \\left(1,I_{\\Phi}(k\\, x)\\right)$"
        )
    else:
        opis = (
            "$\\frac{1}{k}\\left(1+I_{\\Phi}^p(k\\, x) \\right)^{1 / p}$"
        )
    return opis


def plot_kappa(
        Orlicz_function,
        x: np.ndarray,
        k_min: float,
        k_max: float,
        dk: float,
        p_norm: float,
        show_progress: bool = False,
        figsize: tuple = (5, 4),
        show: bool = True,
        save: bool = False,
        save_name: str = None,
        title: str = None
):
    # domain_k, array_k = array_for_infimum(  # tu można użyć funkcji kappa - będzie szybciej.
    #     Orlicz_function,
    #     x,
    #     # dt,
    #     k_min,
    #     k_max,
    #     dk,
    #     p_norm,
    #     show_progress
    # )
    domain_k = np.arange(k_min, k_max, dk)
    array_k = np.array([])
    with tqdm(
            total=len(domain_k),
            # desc="counting of  infimum in [" + str(k_min) + "," + str(k_max) + "]",
            # desc="counting of  $1/k*s_p(I_phi)(kx)$ in [" + str(k_min) + "," + str(k_max) + "]",
            desc=f"counting of  $\\kappa_{{p={p_norm},x}}(k)$ in [" + str(k_min) + "," + str(k_max) + "]",
            disable=not show_progress
    ) as pbar:
        for k in domain_k:
            array_k = np.append(array_k, kappa(Orlicz_function, x, k, p_norm))
            pbar.update(1)

    p_description = p_norm
    fig, ax = plt.subplots(figsize=figsize)
    opis = description_for_plot(p_description)
    norma_x = np.min(array_k)
    b_array_k = 0
    for b_array_k in range(len(array_k)):
        if array_k[b_array_k] == np.inf:
            break
    if p_description != np.inf:
        ax.scatter([], [], facecolors="none",
                   edgecolors="none",
                   label="$p=" + str(p_description) + "$",
                   )
    else:
        ax.scatter([], [], facecolors="none",
                   edgecolors="none",
                   label="$p=\\infty$"
                   )

    if b_array_k < len(array_k) - 1:
        ax.plot(
            domain_k[:b_array_k],
            array_k[:b_array_k],
            label=opis,
            linewidth=2,
        )
        ax.plot(
            domain_k[b_array_k: len(domain_k)],
            np.full(
                (len(domain_k[b_array_k: len(domain_k)])),
                1.3 * max(array_k[:b_array_k]),
            ),
            "--",
            label=opis + "$ = \\infty$",
            linewidth=2,
        )
    else:
        ax.plot(domain_k, array_k, label=opis, linewidth=2)

    accuracy = max((np.max(array_k[np.isfinite(array_k)])
                    - np.min(array_k[np.isfinite(array_k)])) * 0.000001, 1e-15)
    # accuracy = max((np.max(array_k) - np.min(array_k))*0.000001, 1e-15)
    osiagane_minimum = np.where(
        np.logical_and(
            # array_k < array_k[np.argmin(array_k)] + 0.00001,
            # array_k > array_k[np.argmin(array_k)] - 0.00001,
            array_k < array_k[np.argmin(array_k)] + accuracy,
            array_k > array_k[np.argmin(array_k)] - accuracy,
        )
    )

    # y_min, y_max = ax.get_ylim()
    # osiagane_minimum = np.where(
    #     np.logical_and(
    #         # array_k < array_k[np.argmin(array_k)] + 0.00001,
    #         # array_k > array_k[np.argmin(array_k)] - 0.00001,
    #         array_k < array_k[np.argmin(array_k)] + (y_max - y_min) * 0.00001,
    #         array_k > array_k[np.argmin(array_k)] - (y_max - y_min) * 0.00001,
    #     )
    # )

    if array_k[0] > norma_x + accuracy and array_k[-1] > norma_x + accuracy:
        ax.scatter(
            domain_k[osiagane_minimum],
            array_k[osiagane_minimum],
            label="$\\|x\\|\\approx$" + str(norma_x),
        )
    else:
        ax.scatter(
            domain_k[osiagane_minimum],
            array_k[osiagane_minimum],
            label="$\\|x\\|\\leq$" + str(norma_x),
        )
    if np.min(osiagane_minimum) == 0:
        ax.scatter(
            domain_k[np.min(osiagane_minimum)],
            array_k[np.min(osiagane_minimum)],
            facecolors="none",
            edgecolors="none",
            label="$k_{p}^*(x)\\leq$" + str(domain_k[np.min(osiagane_minimum)]),
        )
    elif np.min(osiagane_minimum) == len(domain_k) - 1:
        ax.scatter(
            domain_k[np.min(osiagane_minimum)],
            array_k[np.min(osiagane_minimum)],
            facecolors="none",
            edgecolors="none",
            label="$k_{p}^*(x)\\geq $" + str(domain_k[np.min(osiagane_minimum)]),
        )
    else:
        ax.scatter(
            domain_k[np.min(osiagane_minimum)],
            array_k[np.min(osiagane_minimum)],
            facecolors="none",
            edgecolors="none",
            label="$k_{p}^*(x)\\approx$" + str(domain_k[np.min(osiagane_minimum)]),
        )

    if np.max(osiagane_minimum) == len(domain_k) - 1:
        ax.scatter(
            domain_k[np.max(osiagane_minimum)],
            array_k[np.max(osiagane_minimum)],
            facecolors="none",
            edgecolors="none",
            label="$k_{p}^{**}(x)\\geq $" + str(domain_k[np.max(osiagane_minimum)]),
        )
    elif np.max(osiagane_minimum) == 0:
        ax.scatter(
            domain_k[np.max(osiagane_minimum)],
            array_k[np.max(osiagane_minimum)],
            facecolors="none",
            edgecolors="none",
            label="$k_{p}^{**}(x)\\leq $" + str(domain_k[np.max(osiagane_minimum)]),
        )
    else:
        ax.scatter(
            domain_k[np.max(osiagane_minimum)],
            array_k[np.max(osiagane_minimum)],
            facecolors="none",
            edgecolors="none",
            label="$k_{p}^{**}(x)\\approx $ " + str(domain_k[np.max(osiagane_minimum)]),
        )

    ax.locator_params(axis="x", nbins=10)
    ax.legend()
    # ax.annotate("$k$", xy=(0.98, 0.02), xycoords="axes fraction")
    # ax.annotate("$k$", xy=(1.03, -0.1), xycoords="axes fraction")
    ax.set_xlabel("$k$")
    ax.legend()
    # plt.title('$\\frac{1}{k}s_p\\left(I_{\\Phi}\\left(kx\\right)\\right)$')
    if title == None:
        plt.title('$\\kappa_{p,x}\\left(k\\right)$')
    else:
        plt.title(title)

    # fig.autofmt_xdate(rotation=0) # autorotation of the x-axis
    fig.tight_layout()
    # fig.savefig(my_path + "/plots/k_x.png", dpi=1200)
    if save == True:
        if save_name == None:
            plot_save(name='kappa', p_norm=p_norm)
        else:
            plot_save(save_name)

    if show is True:
        plt.show()
    plt.close()
    # fig.savefig(my_path + f"/plots/kappa_{p_norm}.png", dpi=1200)
    # fig.savefig(my_path + f"/plots/kappa_{p_norm}.svg")
    # fig.savefig(my_path + f"/plots/kappa_{p_norm}.pdf")


if __name__ == "__main__":
    import doctest  # import the doctest library

    doctest.testmod(verbose=True)  # run the tests and display all results (pass or fail)

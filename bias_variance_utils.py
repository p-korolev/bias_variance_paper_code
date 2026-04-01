"""
A toolkit for exploring the bias-variance tradeoff via polynomial regression.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Functions available for experimentation
NAMED_FUNCTIONS = {
    "sin":       (lambda x: np.sin(2 * np.pi * x),  r"$\sin(2\pi x)$"),
    "cos":       (lambda x: np.cos(2 * np.pi * x),  r"$\cos(2\pi x)$"),
    "cubic":     (lambda x: x ** 3 - x**2 + x,              r"$x^3 - x^2 + x$"),
    "step":      (lambda x: np.sign(x - 0.5),        r"$\mathrm{sign}(x-0.5)$"),
    "linear":    (lambda x: 2 * x - 1,               r"$2x - 1$"),
    "quadratic": (lambda x: 4 * x ** 2 - 4 * x + 1, r"$4x^2 - 4x + 1$"),
    "exp":       (lambda x: np.exp(2 * x) - 1,       r"$e^{2x} - 1$"),
    "abs":       (lambda x: np.abs(x - 0.5),         r"$|x - 0.5|$"),
    "quartic": (lambda x: x **4 - 4 * x + 1, r"$x^4 - 4x + 1$"),
    "quintic": (lambda x: x ** 5 - 4 * x + 1, r"$x^5 - 4x + 1$")
}

def resolve_func(true_func):
    """Return (callable, latex_label) for a string name or a callable."""
    if callable(true_func):
        return true_func, r"$f(x)$ (custom)"
    key = true_func.strip().lower()
    if key not in NAMED_FUNCTIONS:
        raise ValueError(
            f"Unknown true_func '{true_func}'. "
            f"Choose from: {list(NAMED_FUNCTIONS)} or pass a callable."
        )
    return NAMED_FUNCTIONS[key]

# Core experiment function to build experiment results dictionary for plotting and visualization
def run_experiment(
    true_func: str = "sin",
    n_train: int = 400,
    noise: float = 0.3,
    max_degree: int = 12,
    min_degree: int = 1,
    n_seeds: int = 200,
    n_test: int = 200,
    x_min: float = 0.0,
    x_max: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Run the full bias–variance simulation and return a results dictionary.

    **Params**
    ----------
    true_func : str or callable
        The ground-truth function f(x).  Use a string name (see module
        docstring) or any Python callable that accepts a NumPy array.
    n_train : int
        Number of training samples per random seed.
    noise : float
        Standard deviation of additive Gaussian noise  ε ~ N(0, noise²).
    max_degree : int
        Highest polynomial degree to evaluate.
    min_degree : int
        Lowest polynomial degree to evaluate (default 1).
    n_seeds : int
        Number of independent training datasets (= Monte-Carlo draws).
        Higher → more accurate bias/variance estimates.
    n_test : int
        Number of equally-spaced test points.
    x_min, x_max : float
        Domain of x.
    seed : int
        Base random seed for reproducibility.

    Returns dictionary with all necessary experiment metrics.
    """
    f, func_label = resolve_func(true_func)
    degrees = list(range(min_degree, max_degree + 1))

    x_test = np.linspace(x_min, x_max, n_test)
    y_test_true = f(x_test)

    all_preds   = {}
    bias2_list  = []
    var_list    = []
    test_mse_list = []
    train_mse_list = []

    for d in degrees:
        preds       = np.zeros((n_seeds, n_test))
        tr_mse_seed = []

        for s in range(n_seeds):
            rng = np.random.default_rng(seed + s * 1000)
            x_tr = rng.uniform(x_min, x_max, n_train)
            y_tr = f(x_tr) + rng.normal(0, noise, n_train)

            model = make_pipeline(
                PolynomialFeatures(degree=d, include_bias=False),
                LinearRegression(),
            )
            model.fit(x_tr.reshape(-1, 1), y_tr)
            preds[s]  = model.predict(x_test.reshape(-1, 1))
            y_hat_tr  = model.predict(x_tr.reshape(-1, 1))
            tr_mse_seed.append(mean_squared_error(y_tr, y_hat_tr))

        mean_pred = preds.mean(axis=0)
        bias2     = float(np.mean((mean_pred - y_test_true) ** 2))
        variance  = float(np.mean(np.var(preds, axis=0)))
        test_mse  = float(np.mean((preds - y_test_true[np.newaxis, :]) ** 2))

        all_preds[d]     = preds
        bias2_list.append(bias2)
        var_list.append(variance)
        test_mse_list.append(test_mse)
        train_mse_list.append(float(np.mean(tr_mse_seed)))

    return dict(
        degrees     = degrees,
        bias2       = np.array(bias2_list),
        variance    = np.array(var_list),
        test_mse    = np.array(test_mse_list),
        train_mse   = np.array(train_mse_list),
        all_preds   = all_preds,
        x_test      = x_test,
        y_test_true = y_test_true,
        func_label  = func_label,
        noise       = noise,
        n_train     = n_train,
        n_seeds     = n_seeds,
        x_min       = x_min,
        x_max       = x_max,
    )

def plot_data_generating_process(
    results: dict,
    n_samples_shown: int = 5,
    figsize: tuple = (9, 4),
) -> plt.Figure:
    """
    Plot the true function and several independent noisy training sets.
    """
    #f, _  = resolve_func(results["func_label"])
    # Re-resolve: func_label is a LaTeX string, so use the stored y_test_true
    x_test      = results["x_test"]
    y_true      = results["y_test_true"]
    noise       = results["noise"]
    n_train     = results["n_train"]
    x_min, x_max = results["x_min"], results["x_max"]
    n_seeds     = results["n_seeds"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_test, y_true, "k-", lw=2.2, zorder=5,
            label=f"True f(x) = {results['func_label']}")

    colours = plt.cm.tab10(np.linspace(0, 0.7, n_samples_shown))
    rng = np.random.default_rng(results.get("seed", 42))
    for i, c in enumerate(colours):
        x_tr = rng.uniform(x_min, x_max, n_train)
        # reconstruct y from all_preds is not possible; sample directly
        # stored all_preds but not raw training data — sample fresh
        y_tr = y_true[  # approximate: project onto test grid nearest points
            np.searchsorted(x_test, np.clip(x_tr, x_min, x_max - 1e-9))
        ] + rng.normal(0, noise, n_train)
        label = f"Training set {i + 1}" if i < 3 else ""
        ax.scatter(x_tr, y_tr, s=22, color=c, alpha=0.75, zorder=4,
                   label=label)

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(
        f"Data-Generating Process  "
        f"(n_train={n_train}, σ={noise})"
    )
    ax.legend(fontsize=8); ax.set_xlim(x_min, x_max)
    plt.tight_layout()
    return fig


def plot_model_showcase(
    results: dict,
    degrees_to_show: list[int] | None = None,
    figsize: tuple = (14, 4),
    y_lim: tuple | None = None,
) -> plt.Figure:
    """
    Overlay all fitted curves for selected degrees to visualise underfitting vs overfitting.
    """
    degrees  = results["degrees"]
    all_pred = results["all_preds"]
    x_test   = results["x_test"]
    y_true   = results["y_test_true"]
    n_seeds  = results["n_seeds"]
    n_train  = results["n_train"]
    noise    = results["noise"]

    if degrees_to_show is None:
        mid = degrees[len(degrees) // 2]
        degrees_to_show = [degrees[0], mid, degrees[-1]]

    # Validate
    degrees_to_show = [d for d in degrees_to_show if d in all_pred]

    titles = {d: _degree_title(d, results) for d in degrees_to_show}

    fig, axes = plt.subplots(1, len(degrees_to_show),
                             figsize=figsize, sharey=True)
    if len(degrees_to_show) == 1:
        axes = [axes]

    for ax, d in zip(axes, degrees_to_show):
        preds = all_pred[d]
        for s in range(n_seeds):
            ax.plot(x_test, preds[s], color="steelblue", alpha=0.04, lw=0.7)
        ax.plot(x_test, preds.mean(axis=0), color="royalblue",
                lw=2.2, label="Mean prediction")
        ax.plot(x_test, y_true, "r--", lw=2, label=f"True {results['func_label']}")
        ax.set_title(titles[d], fontsize=9)
        ax.set_xlabel("x")
        if y_lim:
            ax.set_ylim(*y_lim)
        ax.legend(fontsize=7)
        ax.set_xlim(results["x_min"], results["x_max"])
    axes[0].set_ylabel("y")
    plt.suptitle(
        f"{n_seeds} fitted curves per degree  "
        f"(n_train={n_train}, σ={noise})",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_train_vs_test_error(
    results: dict,
    show_noise_floor: bool = False,
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """
    Plot training error vs test error as a function of polynomial degree, producing the classic U-shaped test-error curve.
    """
    degrees  = results["degrees"]
    train_mse = results["train_mse"]
    test_mse  = results["test_mse"]
    noise     = results["noise"]

    best_idx = int(np.argmin(test_mse))
    best_d   = degrees[best_idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(degrees, train_mse, "o--", color="seagreen", lw=2,
            ms=6, label="Training MSE")
    ax.plot(degrees, test_mse, "s-", color="tomato", lw=2,
            ms=6, label="Test MSE (vs noiseless f)")
    if show_noise_floor:
        ax.axhline(noise ** 2, color="gray", ls=":", lw=1.5,
                   label=f"Irreducible noise  σ² = {noise**2:.3f}")
    ax.axvline(best_d, color="navy", ls="--", lw=1.2, alpha=0.6,
               label=f"Best degree  d = {best_d}")

    ax.set_xlabel("Polynomial Degree (model complexity)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(
        f"Training vs Test Error — U-shaped curve\n"
        f"f(x) = {results['func_label']},  "
        f"n_train = {results['n_train']},  σ = {noise}"
    )
    ax.legend(fontsize=9)
    ax.set_xticks(degrees)
    plt.tight_layout()
    return fig


def plot_bias_variance_decomposition(
    results: dict,
    show_stacked: bool = True,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot the bias^2, variance, and total test error curves together.
    """
    degrees  = results["degrees"]
    bias2    = results["bias2"]
    var      = results["variance"]
    test_mse = results["test_mse"]
    noise    = results["noise"]

    ncols = 2 if show_stacked else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    # Line plot 
    ax = axes[0]
    ax.plot(degrees, bias2, "o-", color="royalblue", lw=2, ms=6, label="Bias²")
    ax.plot(degrees, var, "s-", color="darkorange", lw=2, ms=6, label="Variance")
    ax.plot(degrees, bias2 + var, "^--", color="purple", lw=2, ms=6, label="Bias² + Variance")
    ax.plot(degrees, test_mse, "D:",  color="tomato", lw=2, ms=6, label="Actual Test MSE")
    ax.axhline(noise ** 2, color="gray", ls=":", lw=1.5, label=f"σ² = {noise**2:.3f}")
    ax.set_xlabel("Polynomial Degree"); ax.set_ylabel("Error")
    ax.set_title("Bias–Variance Decomposition")
    ax.legend(fontsize=8); ax.set_xticks(degrees)

    # Stacked area
    if show_stacked:
        ax2 = axes[1]
        ax2.stackplot(
            degrees,
            [bias2, var, np.full_like(bias2, noise ** 2)],
            labels=["Bias²", "Variance", f"Irreducible noise σ²"],
            colors=["royalblue", "darkorange", "lightgray"],
            alpha=0.8,
        )
        ax2.plot(degrees, test_mse, "D-", color="tomato",
                 lw=2, ms=5, label="Actual Test MSE")
        ax2.set_xlabel("Polynomial Degree")
        ax2.set_ylabel("Error (stacked)")
        ax2.set_title("Error Components — Stacked View")
        ax2.legend(fontsize=8); ax2.set_xticks(degrees)

    plt.suptitle(
        f"Bias–Variance Tradeoff  |  f(x) = {results['func_label']}  "
        f"|  n_train = {results['n_train']}  |  σ = {noise}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    return fig

def plot_pointwise_distribution(
    results: dict,
    x_point: float = 0.5,
    degrees_to_show: list[int] | None = None,
    figsize: tuple = (13, 4),
) -> plt.Figure:
    """
    Show the distribution of predictions at a single test point x_point across all random seeds, for selected polynomial degrees.
    """
    x_test   = results["x_test"]
    y_true   = results["y_test_true"]
    all_pred = results["all_preds"]
    degrees  = results["degrees"]
    n_seeds  = results["n_seeds"]

    # Find nearest test grid index
    idx = int(np.argmin(np.abs(x_test - x_point)))
    x0  = x_test[idx]
    f_x0 = y_true[idx]

    if degrees_to_show is None:
        mid = degrees[len(degrees) // 2]
        degrees_to_show = [degrees[0], mid, degrees[-1]]
    degrees_to_show = [d for d in degrees_to_show if d in all_pred]

    fig, axes = plt.subplots(1, len(degrees_to_show),
                             figsize=figsize, sharey=False)
    if len(degrees_to_show) == 1:
        axes = [axes]

    for ax, d in zip(axes, degrees_to_show):
        samples  = all_pred[d][:, idx]
        mean_hat = float(samples.mean())
        b2       = float((mean_hat - f_x0) ** 2)
        vr       = float(samples.var())

        ax.hist(samples, bins=30, color="steelblue", edgecolor="white",
                alpha=0.8, density=True)
        ax.axvline(f_x0,     color="red",  lw=2.5, ls="-",
                   label=f"True f(x₀) = {f_x0:.3f}")
        ax.axvline(mean_hat, color="navy", lw=2.5, ls="--",
                   label=f"Mean pred  = {mean_hat:.3f}")
        ax.set_title(
            f"Degree {d}\nBias² = {b2:.4f}   Var = {vr:.4f}",
            fontsize=9,
        )
        ax.set_xlabel(r"$\hat{f}_s(x_0)$")
        ax.legend(fontsize=7)

    plt.suptitle(
        f"Prediction distribution at x₀ ≈ {x0:.3f}  "
        f"across {n_seeds} seeds  "
        f"|  f(x) = {results['func_label']}",
        y=1.02,
    )
    plt.tight_layout()
    return fig

def plot_bias_variance_curves_at_point(
    results: dict,
    x_point: float = 0.5,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Plot pointwise bias^2, variance, and test MSE as a function of degree, evaluated at a single test location x_point.
    """
    x_test   = results["x_test"]
    y_true   = results["y_test_true"]
    all_pred = results["all_preds"]
    degrees  = results["degrees"]

    idx  = int(np.argmin(np.abs(x_test - x_point)))
    x0   = x_test[idx]
    f_x0 = y_true[idx]

    b2_pt, var_pt, mse_pt = [], [], []
    for d in degrees:
        samps = all_pred[d][:, idx]
        mu = samps.mean()
        b2_pt.append((mu - f_x0) ** 2)
        var_pt.append(samps.var())
        mse_pt.append(((samps - f_x0) ** 2).mean())

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(degrees, b2_pt,  "o-", color="royalblue",  lw=2, ms=6, label="Bias²")
    ax.plot(degrees, var_pt, "s-", color="darkorange",  lw=2, ms=6, label="Variance")
    ax.plot(degrees, mse_pt, "D:", color="tomato",      lw=2, ms=6, label="Test MSE")
    ax.set_xlabel("Polynomial Degree"); ax.set_ylabel("Error")
    ax.set_title(
        f"Pointwise Bias–Variance at x₀ ≈ {x0:.3f}\n"
        f"f(x) = {results['func_label']},  n_train = {results['n_train']}"
    )
    ax.legend(); ax.set_xticks(degrees)
    plt.tight_layout()
    return fig

def print_summary_table(results: dict) -> None:
    """
    Print a formatted table of Bias^2, Variance, Bias^2 + Var, and Test MSE
    for every degree in the sweep, including a decomposition check.
    """
    degrees   = results["degrees"]
    bias2     = results["bias2"]
    var       = results["variance"]
    test_mse  = results["test_mse"]
    train_mse = results["train_mse"]
    noise     = results["noise"]

    hdr = (f"{'Degree':>6}  {'Bias²':>8}  {'Variance':>10}  "
           f"{'B²+Var':>10}  {'TestMSE':>10}  {'TrainMSE':>10}  {'Residual':>10}")
    print(f"\nBias–Variance Summary  |  σ² = {noise**2:.4f}\n")
    print(hdr)
    print("─" * len(hdr))
    for i, d in enumerate(degrees):
        b   = bias2[i]; v = var[i]; t = test_mse[i]; tr = train_mse[i]
        res = t - (b + v)
        marker = " ← best" if t == test_mse.min() else ""
        print(f"{d:>6}  {b:>8.5f}  {v:>10.5f}  "
              f"{b+v:>10.5f}  {t:>10.5f}  {tr:>10.5f}  {res:>10.5f}{marker}")
    print("")

def plot_compare_experiments(
    results_a: dict,
    results_b: dict,
    label_a: str = "Experiment A",
    label_b: str = "Experiment B",
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Side-by-side comparison of bias–variance decompositions for two different experimental configurations.
    results_a, results_b : dict
        Outputs of run_experiment() for each configuration.
    label_a, label_b : str
        Readable string labels for each experiment.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, res, label in zip(axes, [results_a, results_b], [label_a, label_b]):
        degrees = res["degrees"]
        ax.plot(degrees, res["bias2"],    "o-", color="royalblue",  lw=2,
                ms=6, label="Bias²")
        ax.plot(degrees, res["variance"], "s-", color="darkorange",  lw=2,
                ms=6, label="Variance")
        ax.plot(degrees, res["test_mse"], "D:", color="tomato",      lw=2,
                ms=6, label="Test MSE")
        ax.axhline(res["noise"] ** 2, color="gray", ls=":", lw=1.2,
                   label=f"σ²={res['noise']**2:.3f}")
        ax.set_xlabel("Polynomial Degree"); ax.set_ylabel("Error")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8); ax.set_xticks(degrees)

    plt.suptitle("Experiment Comparison: Bias–Variance Decomposition",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_all(
    results: dict,
    x_point: float = 0.5,
    print_table: bool = True,
) -> list[plt.Figure]:
    """
    Generate every standard plot and optionally print the summary table.
    """
    if print_table:
        print_summary_table(results)

    figs = [
        plot_data_generating_process(results),
        plot_model_showcase(results),
        plot_train_vs_test_error(results),
        plot_bias_variance_decomposition(results),
        plot_pointwise_distribution(results, x_point=x_point),
        plot_bias_variance_curves_at_point(results, x_point=x_point),
    ]
    plt.show()
    return figs

# HELPER

def _degree_title(d: int, results: dict) -> str:
    """Return a descriptive title for a given degree based on its position."""
    degrees  = results["degrees"]
    test_mse = results["test_mse"]
    best_d   = degrees[int(np.argmin(test_mse))]
    if d == degrees[0]:
        return f"Degree {d} — Underfit (high bias)"
    if d == degrees[-1]:
        return f"Degree {d} — Overfit (high variance)"
    if d == best_d:
        return f"Degree {d} — Near-optimal"
    if d < best_d:
        return f"Degree {d} — Slight underfit"
    return f"Degree {d} — Slight overfit"


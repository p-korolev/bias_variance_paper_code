"""
A toolkit for exploring the bias-variance tradeoff via polynomial regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Functions available for experimentation
NAMED_FUNCTIONS = {
    "sin": (lambda x: np.sin(2 * np.pi * x), r"$\sin(2\pi x)$"),
    "cos": (lambda x: np.cos(2 * np.pi * x),  r"$\cos(2\pi x)$"),
    "cubic": (lambda x: x ** 3 - x**2 + x, r"$x^3 - x^2 + x$"),
    "step": (lambda x: np.sign(x - 0.5), r"$\mathrm{sign}(x-0.5)$"),
    "linear": (lambda x: 2 * x - 1, r"$2x - 1$"),
    "quadratic": (lambda x: 4 * x ** 2 - 4 * x + 1, r"$4x^2 - 4x + 1$"),
    "exp": (lambda x: np.exp(2 * x) - 1, r"$e^{2x} - 1$"),
    "abs":(lambda x: np.abs(x - 0.5), r"$|x - 0.5|$"),
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

    Params
    --------------
    true_func : true function f
    n_train : number of training samples per seed
    noise : Guassian noise
    max_degree : highest polynomial degree to evaluate
    min_degree : minimum polynomial degree to evaluate 
    n_seeds : number of independent training sets using monte-carlo draws
    n_test : number of test points
    x_min, x_max : x domain
    seed : seed for experiment reproduction

    Returns dictionary with all necessary experiment metrics.
    """
    f, func_label = resolve_func(true_func)
    degrees = list(range(min_degree, max_degree + 1))

    x_test = np.linspace(x_min, x_max, n_test)
    y_test_true = f(x_test)

    all_preds = {}
    bias2_list = []
    var_list = []
    test_mse_list = []
    train_mse_list = []

    for d in degrees:
        preds = np.zeros((n_seeds, n_test))
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
            preds[s] = model.predict(x_test.reshape(-1, 1))
            y_hat_tr = model.predict(x_tr.reshape(-1, 1))
            tr_mse_seed.append(mean_squared_error(y_tr, y_hat_tr))

        mean_pred = preds.mean(axis=0)
        bias2 = float(np.mean((mean_pred - y_test_true) ** 2))
        variance = float(np.mean(np.var(preds, axis=0)))
        test_mse = float(np.mean((preds - y_test_true[np.newaxis, :]) ** 2))

        all_preds[d] = preds
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
    x_test      = results["x_test"]
    y_true      = results["y_test_true"]
    noise       = results["noise"]
    n_train     = results["n_train"]
    x_min, x_max = results["x_min"], results["x_max"]
    n_seeds     = results["n_seeds"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_test, y_true, "k-", lw=2.2, zorder=5, label=f"True f(x) = {results['func_label']}")

    colours = plt.cm.tab10(np.linspace(0, 0.7, n_samples_shown))
    rng = np.random.default_rng(results.get("seed", 42))
    for i, c in enumerate(colours):
        x_tr = rng.uniform(x_min, x_max, n_train)
        y_tr = y_true[
            np.searchsorted(x_test, np.clip(x_tr, x_min, x_max - 1e-9))
        ] + rng.normal(0, noise, n_train)
        label = f"Training set {i + 1}" if i < 3 else ""
        ax.scatter(x_tr, y_tr, s=22, color=c, alpha=0.75, zorder=4, label=label)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    
    ax.set_title(
        f"Example Training Sample"
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

    # validate deg
    degrees_to_show = [d for d in degrees_to_show if d in all_pred]

    titles = {d: _degree_title(d, results) for d in degrees_to_show}

    fig, axes = plt.subplots(1, len(degrees_to_show), figsize=figsize, sharey=True)
    if len(degrees_to_show) == 1:
        axes = [axes]

    for ax, d in zip(axes, degrees_to_show):
        preds = all_pred[d]
        for s in range(n_seeds):
            ax.plot(x_test, preds[s], color="steelblue", alpha=0.04, lw=0.7)
        ax.plot(x_test, preds.mean(axis=0), color="royalblue", lw=2.2, label="Mean prediction")
        ax.plot(x_test, y_true, "r--", lw=2, label=f"True {results['func_label']}")
        ax.set_title(titles[d], fontsize=9)
        ax.set_xlabel("x")
        if y_lim:
            ax.set_ylim(*y_lim)
        ax.legend(fontsize=7)
        ax.set_xlim(results["x_min"], results["x_max"])
    axes[0].set_ylabel("y")
    
    plt.suptitle(
        f"Fitted Polynomials by Degree",
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
    ax.plot(degrees, train_mse, marker='o', color="seagreen", lw=2, ms=6, label="Training Error")
    ax.plot(degrees, test_mse, marker='o',color="tomato", lw=2, ms=6, label="Test Error")
    if show_noise_floor:
        ax.axhline(noise ** 2, color="gray", ls=":", lw=1.5, label=f"Irreducible noise  σ² = {noise**2:.3f}")
    ax.axvline(best_d, color="navy", ls="--", lw=1.2, alpha=0.6, label=f"Optimal Degree")
    ax.set_xlabel("Model Complexity")
    ax.set_ylabel("Error")
    ax.set_title(f"Training vs Test Error")
    ax.legend(fontsize=9)
    ax.set_xticks(degrees)
    plt.tight_layout()
    return fig


def plot_bias_variance_decomposition(
    results: dict,
    show_stacked: bool = True,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot the bias^2, variance, and total test error curves together.
    """
    degrees  = results["degrees"]
    bias2    = results["bias2"]
    var      = results["variance"]
    test_mse = results["test_mse"]
    noise    = results["noise"]
    bias_and_var = results["bias2"] + results["variance"]
    
    ncols = 2 if show_stacked else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    # decomposition parts
    ax = axes[0]
    ax.plot(degrees, bias2, "o-", color="royalblue", lw=2, ms=6, label="Bias^2")
    ax.plot(degrees, var, "s-", color="darkorange", lw=2, ms=6, label="Variance")
    ax.plot(degrees, bias_and_var, "^--", color="purple", lw=2, ms=6, label="Bias^2 + Variance + Noise")
    ax.plot(degrees, test_mse, "D:",  color="tomato", lw=2, ms=6, label="Test MSE")
    #ax.axhline(noise ** 2, color="gray", ls=":", lw=1.5, label=f"σ² = {noise**2:.3f}")
    ax.set_xlabel("Model Complexity"); ax.set_ylabel("Error")
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
    plt.tight_layout()
    return fig


# HELPER
def _degree_title(d: int, results: dict) -> str:
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
        return f"Degree {d} — Underfit (high bias)"
    return f"Degree {d} — Overfit (high variance)"


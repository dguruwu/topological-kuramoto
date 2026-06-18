"""
=========================================================
Ring Graphs and 1D Kuramoto Dynamics Visualization
=========================================================

This module provides specialized routines for constructing, analyzing,
and visualizing phase-locked states on 1D ring complexes (polygonal faces).

It serves as a simplified 1-dimensional case of the general topological
Kuramoto model, used to produce Figures 3, 4 and 5 of the manuscript.

Functions
---------
Core solver
- find_what_works_ring : 1D reduction of the n-D solver; all phase-locked
    states for a given ring partition.

Styling and label helpers
- generate_multiset_label : LaTeX (|S+|, |S-|) multiset-size label for a partition.
- arrays_equal           : np.allclose comparison used to group identical f-curves.
- add_2pi_lines          : Draw integer 2*pi reference lines and matching y-ticks.
- wrap_labels            : textwrap helper for long legend labels (currently unused).

Curve and range computation
- compute_zeta_range     : Valid zeta^[-] range giving consistent arcsin branches.
- get_plotting_data_ring : Evaluate the projected f-curve along zeta^[-].

Robust root-finding and stability
- find_two_robust_brackets        : Group sign changes into bracket intervals for brentq.
- robust_find_phase_locked_roots  : Intersections of projected f-curves with integer
    lines, robustly across omega_0.
- check_stable_jacobian_1d        : 1D ring Jacobian spectrum and per-root stability.

Ring time-series dynamics
- ring_theta_dynamics       : Projected lower-dimensional ring ODE (right-hand side).
- theta_star_from_ring_root : Reconstruct stationary theta^[-] from a scalar root.
- simulate_theta            : Integrate ring dynamics from a noisy perturbation of theta*.

Figure 3 (f-curves and branch time series)
- plot_ring_fcurves_grouped_stable : Panel a — distinct f-curves grouped by symmetry,
    annotated with Jacobian-based stability markers.
- plot_ring_timeseries_branches    : Panels b-d — time series for the three stable branches.
- make_figure_3                    : Assemble Figure 3.

Figure 4 (omega_0 dependence and bifurcation)
- plot_all_phase_locked_solutions_for_varying_w0_both : Panel a — evolution of f-curves
    with omega_0, stable/unstable intersections marked.
- plot_bifurcation_ring_branches : Panel b — bifurcation diagram of zeta^[-] branches vs omega_0.
- make_figure_4                  : Assemble Figure 4.

Figure 5 (scaling with ring size)
- plot_figure_5 : Number of stable all-plus omega-zero solutions vs ring size s.

Notes
-----
These routines are specifically optimized for the ring (1D cycle) topology.
They rely on the general Kuramoto dynamics utilities from `core.dynamics`.
"""


import numpy as np
from scipy.optimize import brentq, root_scalar, root as root_solver
from scipy.integrate import solve_ivp
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import textwrap 

from core.dynamics import get_psi_minus_sp, build_partitioned_root_problem, generate_z_vectors_from_C, f_n_minus_one, is_jacobian_stable


# === Aesthetics according to Nature guidelines ===

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
NATURE_FONT = (
    "Arial" if "Arial" in available_fonts
    else "Helvetica" if "Helvetica" in available_fonts
    else "DejaVu Sans"
)

mpl.rcParams.update({
    # Editable/vector PDF output
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,

    # Sans-serif text
    "font.family": "sans-serif",
    "font.sans-serif": [NATURE_FONT],
    "font.size": 6,

    # Sans-serif mathtext
    "mathtext.fontset": "custom",
    "mathtext.rm": NATURE_FONT,
    "mathtext.it": f"{NATURE_FONT}:italic",
    "mathtext.bf": f"{NATURE_FONT}:bold",

    # Nature-sized text at final display size
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,

    # Thin publication lines
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 0.75,
})

import logging

logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

def find_what_works_ring(S_plus, S_minus, B1, K1, C1, tol=1e-7, omega=None, z_range=None, verbose=False):
    """
    1D Ring wrapper around the n-D solver, allowing manual z-range override
    (default z-range is adjusted for stable solution search, here we look for all solutions, including unstable ones).
    """
    global generate_z_vectors_from_C

    # (n+1) layer absent: use explicit empties
    B_np1 = np.zeros((0, B1.shape[0]))   # (0 x num_edges) for shape safety
    K_np1 = np.eye(0)
    C_np1 = np.zeros((0, 0))

    # partition mapping
    S0_np1, S2_np1 = [], []
    S0_n, S2_n = S_plus, S_minus

    # allow manual z-range override
    if z_range is not None:
        def generate_z_override(C_n, C_np1):
            z_n_vecs = [np.array([z]) for z in z_range]
            z_np1_vecs = [()]  # none on upper layer
            if verbose:
                print(f"manual z_n_vecs = {list(z_range)}")
            return z_n_vecs, z_np1_vecs
        generate_func = generate_z_override
    else:
        generate_func = generate_z_vectors_from_C

    omega_used = np.zeros(B1.shape[1]) if omega is None else omega

    # temporarily swap generator
    original_gen = generate_z_vectors_from_C
    generate_z_vectors_from_C = generate_func

    try:
        # manually compute psi_sp for ring 
        psi_n_sp = get_psi_minus_sp(B1, K1, omega_used)
        psi_np1_sp = np.zeros((0,))   # empty vector

        # now use the same residual builder directly 
        z_n_vecs, z_np1_vecs = generate_func(C1, C_np1)

        roots = []
        for z_np1 in z_np1_vecs:
            for z_n in z_n_vecs:
                func, target, zeta0, mode = build_partitioned_root_problem(psi_np1_sp, psi_n_sp, z_np1, z_n, S0_np1, S0_n, S2_np1, S2_n,
                                                                           K1, K_np1, C1, C_np1)

                if mode == "none":
                    continue

                result = root_solver(func, x0=zeta0, method="lm", options={"xtol": 1e-10, "ftol": 1e-10})
                fun_norm = np.linalg.norm(result.fun) if result.success else np.nan

                if verbose:
                    print(f"z_n={z_n}, success={result.success}, ||f||={fun_norm:.2e}")

                if result.success and not np.isnan(result.fun).any() and fun_norm < tol:
                    roots.append((np.array([]), z_n, np.array([]), result.x))

    finally:
        generate_z_vectors_from_C = original_gen  # restore original

    return roots


# === Fig 3 related functions ===

# Styling the plots

def generate_multiset_label(S_plus, S_minus):
    return rf"$(|S_{{[0]}}^\bullet|, |S_{{[0]}}^\circ|)=({len(S_plus)},{len(S_minus)})$"

def arrays_equal(a, b, tol=1e-8):
    return np.allclose(a, b, atol=tol)

def add_2pi_lines(ax, range_z=(-1, 4)):
    for z in range(range_z[0], range_z[1] + 1):
        ax.axhline(y=z, color="0.7", linestyle="--", linewidth=0.45)
    ax.set_yticks(range(range_z[0], range_z[1] + 1))
    ax.set_yticklabels([str(z) for z in range(range_z[0], range_z[1] + 1)])

# Computing data for plots

def compute_zeta_range(psi_sp, K1, C1):
    C1 = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    d = np.linalg.solve(K1, C1.T).flatten()
    zeta_mins, zeta_maxs = [], []
    for psi_i, d_i in zip(psi_sp, d):
        if abs(d_i) < 1e-12:
            continue
        lo = (-1 - psi_i) / d_i
        hi = ( 1 - psi_i) / d_i
        zeta_mins.append(min(lo, hi))
        zeta_maxs.append(max(lo, hi))
    if not zeta_mins or not zeta_maxs:
        return -np.pi, np.pi
    eps = 1e-9
    return np.max(zeta_mins) + eps, np.min(zeta_maxs) - eps

def get_plotting_data_ring(psi_sp, S_plus, S_minus, K1, C1, resolution=500):
    # exact zeta range
    zmin, zmax = compute_zeta_range(psi_sp, K1, C1)
    zetas = np.linspace(zmin, zmax, resolution)

    # d = K1^{-1} C1^T : shape (n, r). For ring r=1.
    C1c = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    d = np.linalg.solve(K1, C1c.T) # (n, r)
    if d.shape[1] != 1:
        raise ValueError("Ring plotting expects rank(C1)=1.")

    d = d[:, 0] # (n,)
    psi_vals = psi_sp[None, :] + np.outer(zetas, d) # (res, n)
    psi_vals[(psi_vals > 1) | (psi_vals < -1)] = np.nan

    f_vals = np.zeros_like(psi_vals)
    f_vals[:, S_plus]  = np.arcsin(psi_vals[:, S_plus])
    f_vals[:, S_minus] = np.pi - np.arcsin(psi_vals[:, S_minus])

    # project via C1
    f_projected = (f_vals @ C1c.T) / (2 * np.pi) # (res, 1)
    return zetas, f_projected


# Final plotting function
def plot_ring_fcurves_grouped_stable(B1, K1, C1, all_partitions_ring, omega=None, resolution=600, levels=(-1, 4), ax = None, save_path=None):
    """
    Group identical f-curves (by multiset permutations), plot one representative per group,
    and mark each intersection with stability from Jacobian test.
    """
    omega = np.zeros(B1.shape[1]) if omega is None else omega
    psi_sp = get_psi_minus_sp(B1, K1, omega)

    # Group identical curves, store a representative partition per curve
    unique_curves = [] # list of dicts: {'zetas','f','S_plus','S_minus'}
    label_groups = defaultdict(list)

    for S_plus, S_minus in all_partitions_ring:
        zetas, f_vals = get_plotting_data_ring(psi_sp, S_plus, S_minus, K1, C1, resolution)
        found = False
        for j, cur in enumerate(unique_curves):
            if arrays_equal(f_vals, cur['f']):
                label_groups[j].append(generate_multiset_label(S_plus, S_minus))
                found = True
                break
        if not found:
            j = len(unique_curves)
            unique_curves.append({'zetas': zetas, 'f': f_vals, 'S_plus': S_plus, 'S_minus': S_minus})
            label_groups[j].append(generate_multiset_label(S_plus, S_minus))

    # Plotting
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(90 / 25.4, 100 / 25.4))
    else:
        fig = ax.figure

    add_2pi_lines(ax, range_z=levels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_curves)))

    # Precompute d for residuals/J 
    C1c = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    dir_vec = np.linalg.solve(K1, C1c.T)
    if dir_vec.shape[1] != 1:
        raise ValueError("Ring plotting expects rank(C1)=1.")
    d = dir_vec[:, 0] # (n,)

    for i, cur in enumerate(unique_curves):
        zetas, f_vals = cur['zetas'], cur['f']
        S_plus, S_minus = cur['S_plus'], cur['S_minus']
        f_flat = f_vals.ravel()
        label = generate_multiset_label(S_plus, S_minus)
        ax.plot(zetas, f_flat, color=colors[i], lw=0.8, label=label)

        # Intersections & stability using Jacobian test
        for k in range(levels[0], levels[1] + 1):
            g = f_flat - k
            # Find brackets where g changes sign and both endpoints are finite
            idxs = np.where(np.isfinite(g[:-1]) & np.isfinite(g[1:]) & (g[:-1] * g[1:] < 0))[0]
            for idx in idxs:
                a, b = zetas[idx], zetas[idx + 1]
                # Scalar residual for this (S_plus, S_minus)
                def residual(z):
                    psi = psi_sp + d * z
                    f = f_n_minus_one(psi, S_plus, S_minus)
                    # project via C1 and normalize by 2pi
                    return (C1c @ f)[0] / (2*np.pi) - k

                # Robust root in [a,b]
                try:
                    z_star = brentq(residual, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                except ValueError:
                    continue

                # Jacobian and stability
                psi_root = psi_sp + d * z_star
                theta = f_n_minus_one(psi_root, S_plus, S_minus)
                J1 = -B1.T @ K1 @ np.diag(np.cos(theta)) @ B1
                stable = is_jacobian_stable(J1)

                if stable:
                    ax.plot(z_star, k, "o", color=colors[i], markersize=5.0, markeredgewidth=0.75, markeredgecolor=colors[i], markerfacecolor=colors[i], zorder=3)
                else:
                    ax.plot(z_star, k, "o", color=colors[i], markersize=5.0, markeredgewidth=0.75, markeredgecolor=colors[i], markerfacecolor="white", zorder=3)

    # Cosmetics
    ax.set_xlabel(r"$\zeta^{[-]}$")
    ax.set_ylabel(r"$1^\top f_{0}(\psi^{[-]})/(2\pi)$")
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates while preserving order
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    ax.legend(uniq_handles, uniq_labels, bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=2, frameon=False)
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    
    #ax.text(-0.08, 0.97, "a", transform=ax.transAxes, ha="left", va="top", fontsize=16, weight="bold", clip_on=False)
    ax.text(0.01, 0.98, "a", transform=ax.transAxes, ha="left", va="top", fontsize=7, weight="bold")
    #plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    #plt.savefig(save_path, bbox_inches='tight')
    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300)
    #plt.show()


# === Fig 3 b, c, d, ===

def ring_theta_dynamics(t, theta):
    """
    Evaluate the projected lower-dimensional dynamics on a ring.
    """

    theta = np.asarray(theta)
    return np.roll(np.sin(theta), -1) - 2*np.sin(theta) + np.roll(np.sin(theta), 1)

def theta_star_from_ring_root(zeta_minus, S_plus, S_minus, n_edges=6):
    """
    Reconstruct a stationary ring branch from a scalar Kirchhoff root.
    Returns stationary lower interaction angles theta^[-] for the selected branch.
    """

    zeta = float(np.atleast_1d(zeta_minus)[0])
    psi = np.full(n_edges, zeta)

    theta = np.zeros(n_edges)
    theta[S_plus] = np.arcsin(psi[S_plus])
    theta[S_minus] = np.pi - np.arcsin(psi[S_minus])

    return theta

def simulate_theta(theta_star, tmax=40, npts=2000, noise=1e-2, seed=0, preserve_sum=True):
    """
    Simulate ring dynamics from a noisy perturbation of a stationary state.
    """
    
    rng = np.random.default_rng(seed)
    eta = noise * rng.standard_normal(len(theta_star))

    if preserve_sum:
        eta = eta - np.mean(eta) # preserve total circulation / winding numbers

    theta0 = theta_star + eta

    t_eval = np.linspace(0, tmax, npts)
    sol = solve_ivp(ring_theta_dynamics, (0, tmax), theta0, t_eval=t_eval, rtol=1e-9, atol=1e-11, method="RK45")
    return sol

def plot_ring_timeseries_branches(sims, theta_stars, axes = None, save_path=None):
    """
    Plot three stacked time-series panels for the three stable ring branches.

    """
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(90 / 25.4, 120 / 25.4), sharex=True)
    else:
        fig = axes[0].figure

    # six consistent edge colors
    edge_colors = plt.cm.viridis(np.linspace(0.12, 0.88, 6))
    panel_labels = ["b", "c", "d"]
    horizontal_lines = [np.pi/3, 0, -np.pi/3]

    for ax, sol, theta_star, plabel, hl in zip(axes, sims, theta_stars, panel_labels, horizontal_lines):
        z = int(round(np.sum(theta_star) / (2 * np.pi)))

        for j in range(sol.y.shape[0]):
            ax.plot(sol.t, sol.y[j], lw=0.65, color=edge_colors[j], label=rf"$i={j+1}$" if ax is axes[0] else None)

        ax.axhline(hl, ls="--", lw=0.5, color="0.5")

        ref_label = {-1: r"$-\pi/3$",0: r"$0$", 1: r"$\pi/3$"}.get(z, rf"${theta_star[0]:.2f}$")

        ax.annotate(ref_label, xy=(0.985, theta_star[0]), xycoords=("axes fraction", "data"), xytext=(-2, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=5.5, color="0.35")

        ax.set_ylabel(r"$\theta_i^{[-]}(t)$")
        ax.tick_params(axis="both")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.text(0.01, 0.96, plabel, transform=ax.transAxes, ha="left", va="bottom", fontsize=7, weight="bold", clip_on=False)
        ax.text(0.98, 0.94, rf"$z^{{[-]}}={z}$", transform=ax.transAxes, ha="right", va="top", fontsize=5.5)


    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=6, frameon=False, fontsize=5, handlelength=1.0,
                   columnspacing=0.6, handletextpad=0.25, borderaxespad=0.15)

    for ax in axes[:2]:
        ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xlabel(r"$t$")

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300)

# === Fig 3 combined ===

def make_figure_3(B1, K1, C1, all_partitions, sims, theta_stars, omega=None, save_path="Fig3.pdf"):
    MM = 1 / 25.4
    FIG3_WIDTH_MM = 180
    FIG3_HEIGHT_MM = 85

    fig = plt.figure(figsize=(FIG3_WIDTH_MM * MM, FIG3_HEIGHT_MM * MM), dpi=300)

    # Left panel
    ax_a = fig.add_axes([0.06, 0.32, 0.40, 0.63])

    # Right column
    right_x = 0.55
    right_w = 0.41
    right_top = 0.955
    right_bottom = 0.11
    right_gap = 0.055
    right_h = (right_top - right_bottom - 2 * right_gap) / 3

    y_d = right_bottom
    y_c = y_d + right_h + right_gap
    y_b = y_c + right_h + right_gap

    ax_b = fig.add_axes([right_x, y_b, right_w, right_h])
    ax_c = fig.add_axes([right_x, y_c, right_w, right_h], sharex=ax_b)
    ax_d = fig.add_axes([right_x, y_d, right_w, right_h], sharex=ax_b)
    axes_bcd = [ax_b, ax_c, ax_d]

    plot_ring_fcurves_grouped_stable(B1, K1, C1, all_partitions, omega=omega, ax=ax_a, save_path=None)
    plot_ring_timeseries_branches(sims, theta_stars, axes=axes_bcd, save_path=None)

    # a label position
    for txt in ax_a.texts:
        if txt.get_text() == "a":
            txt.set_position((-0.11, 1.00))
            txt.set_ha("left")
            txt.set_va("top")
            txt.set_clip_on(False)

    # b, c, d label position
    for ax, label in zip(axes_bcd, ["b", "c", "d"]):
        for txt in ax.texts:
            if txt.get_text() == label:
                txt.set_position((-0.11, 1.00))
                txt.set_ha("left")
                txt.set_va("bottom")
                txt.set_clip_on(False)

    fig.savefig(save_path, format="pdf", dpi=300)
    plt.show()
    plt.close(fig)

# === Fig 4 related functions ===

# Robust root-finding helpers

def find_two_robust_brackets(func, zeta_range, target, num=8000, merge_threshold=5e-4):
    """
    Find two robust bracket intervals where f(zeta) - target crosses zero.
    Groups nearby crossings into wider intervals (used for Fig. 3a).
    """
    zetas = np.linspace(zeta_range[0], zeta_range[1], num)
    f_vals = np.array([func(z) - target for z in zetas])

    small_brackets = []
    for i in range(len(f_vals) - 1):
        if np.isnan(f_vals[i]) or np.isnan(f_vals[i + 1]):
            continue
        if f_vals[i] * f_vals[i + 1] < 0:
            small_brackets.append((zetas[i], zetas[i + 1]))

    if not small_brackets:
        return []

    # Group nearby brackets into two wide ones
    grouped = []
    current = [small_brackets[0]]
    for (a, b) in small_brackets[1:]:
        if abs(a - current[-1][1]) < merge_threshold:
            current.append((a, b))
        else:
            grouped.append(current)
            current = [(a, b)]
    grouped.append(current)

    wide_brackets = []
    for group in grouped[:2]:
        start, end = group[0][0], group[-1][1]
        wide_brackets.append((start, end))

    return wide_brackets


def robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, z_vals, S_plus, S_minus, K1, C1):
    """
    Finds all phase-locked roots (z, zeta) for given partitions.
    Uses robust bracketing (brentq) across integer targets 2*pi*z.
    """
    roots = []
    for z in z_vals:
        target = 2 * np.pi * z
        func = lambda zeta: (C1 @ f_n_minus_one(psi_sp + np.linalg.solve(K1, C1.T)[:, 0] * zeta, S_plus, S_minus))[0]

        brackets = find_two_robust_brackets(func, (zeta_min, zeta_max), target)
        for a, b in brackets:
            try:
                result = root_scalar(lambda zeta: func(zeta) - target, bracket=[a, b], method='brentq')
                if result.converged:
                    roots.append((z, result.root))
            except ValueError:
                continue
    return roots


def check_stable_jacobian_1d(roots, psi_sp, B1, K1, C1, S_plus, S_minus):
    """
    Compute Jacobian and check stability for each root in 1D ring case.
    Returns list of indices of stable roots and corresponding eigenvalues.
    """
    stable_indices = []
    eigvals_all = []

    C1c = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    d = np.linalg.solve(K1, C1c.T)[:, 0]

    for idx, (_, zeta_root) in enumerate(roots):
        psi_root = psi_sp + d * zeta_root
        theta = f_n_minus_one(psi_root, S_plus, S_minus)
        J = -B1.T @ K1 @ np.diag(np.cos(theta)) @ B1
        eigvals = np.linalg.eigvalsh(J)
        eigvals_all.append(eigvals)
        
        nonzero = np.abs(eigvals) > 1e-10
        if np.all(eigvals[nonzero] < 0):
            stable_indices.append(idx)

    return stable_indices, eigvals_all


# === Figure 4a plotting ===

def plot_all_phase_locked_solutions_for_varying_w0_both(Sp0, Sm0, Sp1, Sm1, B1, K1, C1, w0_values_main, w0_values_inset, resolution_main=1500, resolution_inset=3500, ax=None, save_path=None):
    """
    Plot main and inset figures for varying omega_0 values (Fig. 4a).
    Shows two partitions (e.g. all-plus and one-minus) for a 1D ring.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(88/25.4, 66/25.4), dpi=250)
    else:
        fig = ax.figure

    all_w0_values = list(w0_values_main) + [w for w in w0_values_inset if w not in w0_values_main]
    colors_all = cm.viridis(np.linspace(0, 1, len(all_w0_values)))
    color_map = {w0: colors_all[i] for i, w0 in enumerate(all_w0_values)}

    lines, labels = [], []

    # Main curves
    for w0 in w0_values_main:
        omega = np.array([0, 0, 0, 0, +w0, -w0])
        psi_sp = get_psi_minus_sp(B1, K1, omega)
        zetas0, f_vals0 = get_plotting_data_ring(psi_sp, Sp0, Sm0, K1, C1, resolution_main)
        zetas1, f_vals1 = get_plotting_data_ring(psi_sp, Sp1, Sm1, K1, C1, resolution_main)

        color = color_map[w0]
        l1, = ax.plot(zetas0, f_vals0, color=color, lw=0.8)
        ax.plot(zetas1, f_vals1, '--', color=color, lw=0.8)

        # Roots and stability
        zeta_min, zeta_max = compute_zeta_range(psi_sp, K1, C1)
        roots_0 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [-1, 0, 1], Sp0, Sm0, K1, C1)
        roots_1 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [-1, 0, 1], Sp1, Sm1, K1, C1)
        stable_0, _ = check_stable_jacobian_1d(roots_0, psi_sp, B1, K1, C1, Sp0, Sm0)
        stable_1, _ = check_stable_jacobian_1d(roots_1, psi_sp, B1, K1, C1, Sp1, Sm1)

        for idx, (z, zeta) in enumerate(roots_0):
            ax.plot(zeta, z, 'o', markerfacecolor=color if idx in stable_0 else 'white', markeredgecolor=color, markersize=5.0, markeredgewidth=0.75, zorder=3)
        for idx, (z, zeta) in enumerate(roots_1):
            ax.plot(zeta, z, 'o', markerfacecolor=color if idx in stable_1 else 'white', markeredgecolor=color, markersize=5.0, markeredgewidth=0.75, zorder=3)

        lines.append(l1)
        labels.append(fr"$\omega_0 = {w0:.2f}$")

    # Horizontal lines
    for k in range(-1, 2):
        ax.axhline(y=k, linestyle=(0, (4, 3)), color='0.7', linewidth=0.45, zorder=0)

    ax.set_xlabel(r"$\zeta^{[-]}$")
    ax.set_ylabel(r"$C_1^\top f_0(\psi^{[-]})/(2\pi)$")
    ax.tick_params(axis='both')

    # Inset
    axins = ax.inset_axes([0.53, 0.13, 0.43, 0.37])  # slightly higher than before
    for w0 in w0_values_inset:
        color = color_map[w0]
        omega = np.array([0, 0, 0, 0, +w0, -w0])
        psi_sp = get_psi_minus_sp(B1, K1, omega)
        zetas0, f_vals0 = get_plotting_data_ring(psi_sp, Sp0, Sm0, K1, C1, resolution_inset)
        zetas1, f_vals1 = get_plotting_data_ring(psi_sp, Sp1, Sm1, K1, C1, resolution_inset)
        axins.plot(zetas0, f_vals0, color=color, lw=0.65)
        axins.plot(zetas1, f_vals1, '--', color=color, lw=0.65)

        # Find local roots in narrow window
        zeta_min, zeta_max = -0.2, -0.08
        roots_0 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [0], Sp0, Sm0, K1, C1)
        roots_1 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [0], Sp1, Sm1, K1, C1)
        stable_0, _ = check_stable_jacobian_1d(roots_0, psi_sp, B1, K1, C1, Sp0, Sm0)
        stable_1, _ = check_stable_jacobian_1d(roots_1, psi_sp, B1, K1, C1, Sp1, Sm1)

        for idx, (z, zeta) in enumerate(roots_0):
            axins.plot(zeta, z, 'o', markerfacecolor=color if idx in stable_0 else 'white', markeredgecolor=color, markersize=4, markeredgewidth=0.4, zorder=3)
        for idx, (z, zeta) in enumerate(roots_1):
            axins.plot(zeta, z, 'o', markerfacecolor=color if idx in stable_1 else 'white', markeredgecolor=color, markersize=4, markeredgewidth=0.4, zorder=3)

    axins.set_xlim([-0.175, -0.075])
    axins.set_ylim([-0.04, 0.03])
    axins.axhline(y=0, linestyle=(0, (4, 3)), color='0.7', linewidth=0.4, zorder=0)
    axins.tick_params(axis='both', labelsize=5)
    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.7")
    for artist in [pp, p1, p2]:
        artist.set_linewidth(0.3)
        artist.set_edgecolor("0.75")

    inset_only_w0 = [w for w in w0_values_inset if w not in w0_values_main]
    for w0 in inset_only_w0:
        color = color_map[w0]
        dummy_line, = ax.plot([], [], color=color, lw=0.8)
        lines.append(dummy_line)
        labels.append(fr"$\omega_0 = {w0:.3f}$")

    # Final legend
    ax.legend(lines, labels, loc='upper left', fontsize=5.2, ncol=2, frameon=True, borderpad=0.3, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)

    ax.text(-0.10, 1.01, "a", transform=ax.transAxes, ha="left", va="top", fontsize=7, weight="bold", clip_on=False)

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300)


def plot_bifurcation_ring_branches(Sp0, Sm0, Sp1, Sm1, B1, K1, C1, omega0_vals=np.linspace(1.307, 1.3275, 200), target_z=0, ax=None, save_path=None):
    """
    Plot the bifurcation diagram (Fig. 4b) showing how the phase-locked roots
    move as omega_0 varies for two selected partitions in the 1D ring.

    Parameters
    ----------
    Sp0, Sm0 : list[int]
        Partition defining the all-plus branch (S0_plus, S0_minus = emptyset).
    Sp1, Sm1 : list[int]
        Partition defining the mixed branch (+++++- etc.).
    B1, K1, C1 : np.ndarray
        Boundary, coupling and cycle matrices for the ring.
    omega0_vals : iterable
        Range of omega_0 values to sweep.
    target_z : int
        Integer winding number used in Eq. (22).
    save_path : str
        Optional file path for saving the figure.
    """
    z_vals = [target_z]
    results_partitioned = []

    for i, w0 in enumerate(omega0_vals):
        omega = np.array([0, 0, 0, 0, +w0, -w0])
        psi_sp = get_psi_minus_sp(B1, K1, omega)

        # Root finding for both partitions
        zeta_min, zeta_max = compute_zeta_range(psi_sp, K1, C1)
        roots_5_1 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, z_vals, Sp1, Sm1, K1, C1)
        roots_6_0 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, z_vals, Sp0, Sm0, K1, C1)

        results_partitioned.append({"omega0": w0, "roots_5_1": [r[1] for r in roots_5_1], "roots_6_0": [r[1] for r in roots_6_0]})

    # Extract branch data
    omega_list = [r["omega0"] for r in results_partitioned]
    root_5_1_a, root_5_1_b = [], []
    root_6_0_a, root_6_0_b = [], []

    for entry in results_partitioned:
        r51 = entry["roots_5_1"]
        r60 = entry["roots_6_0"]

        root_5_1_a.append(r51[0] if len(r51) > 0 else np.nan)
        root_5_1_b.append(r51[1] if len(r51) > 1 else np.nan)
        root_6_0_a.append(r60[0] if len(r60) > 0 else np.nan)
        root_6_0_b.append(r60[1] if len(r60) > 1 else np.nan)

    # Detect discontinuities (NaN appears after branch disappears)
    nan_idx_60 = np.isnan(root_6_0_a)
    first_nan_60 = np.argmax(nan_idx_60) if np.any(nan_idx_60) else None
    nan_idx_51 = np.isnan(root_5_1_a)
    first_nan_51 = np.argmax(nan_idx_51) if np.any(nan_idx_51) else None

    # Plotting 
    if ax is None:
        fig, ax = plt.subplots(figsize=(88/25.4, 50/25.4), dpi=250)
    else:
        fig = ax.figure

    # Partition S_{[0]}^{\circ} = emptyset
    ax.plot(omega_list, root_6_0_a, label=r'$\cos(\theta^{[-]}_{5})>0,\ \mathrm{stable}$', color="#E69F00", lw=0.8)

    # Partition S_{[0]}^{\circ} = [5,6]
    ax.plot(omega_list, root_5_1_b, label=r'$\cos(\theta^{[-]}_{5})<0,\ \mathrm{stable}$', color="#003366", lw=0.8)
    ax.plot(omega_list, root_5_1_a, label=r'$\cos(\theta^{[-]}_{5})<0,\ \mathrm{unstable}$', color="#003366", linestyle='--', lw=0.8)

    # Vertical dashed lines at bifurcations
    if first_nan_60 is not None and first_nan_60 > 0:
        ax.axvline(omega_list[first_nan_60 - 1], linestyle='--', color='k', lw=0.45)
    if first_nan_51 is not None:
        ax.axvline(omega_list[first_nan_51], linestyle='--', color='k', lw=0.45)

    ax.set_xlabel(r'$\omega_0$')
    ax.set_ylabel(r'$\zeta^{[-]}$')
    ax.tick_params(axis='both')

    # Legend
    leg = ax.legend(fontsize=5.2, loc='center left', bbox_to_anchor=(0.07, 0.55), frameon=True, handlelength=1.6, borderpad=0.3, labelspacing=0.45, handletextpad=0.5)
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("0.8")
    leg.get_frame().set_facecolor("white")

    ax.set_xlim(left=omega_list[0], right=omega_list[-1])
    ax.margins(x=0)

    # Nice x-ticks
    xticks = np.linspace(omega_list[0], omega_list[-1], 7)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.text(-0.10, 1.01, "b", transform=ax.transAxes, ha="left", va="top", fontsize=7, weight="bold", clip_on=False)

    # Saddle-node bifurcation annotation
    if first_nan_51 is not None:
        ax.annotate("saddle-node\nbifurcation", xy=(omega_list[first_nan_51], root_5_1_b[first_nan_51 - 1]), xycoords="data",
                    xytext=(omega_list[first_nan_51] + 0.0020, -0.104), textcoords="data", ha="left", va="center", fontsize=5.5,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300)


def make_figure_4(Sp0, Sm0, Sp1, Sm1, B1, K1, C1, w0_values_main, w0_values_inset, omega0_vals, save_path="Fig4.pdf"):
    MM = 1 / 25.4
    FIG4_WIDTH_MM = 88
    FIG4_HEIGHT_MM = 120

    fig = plt.figure(figsize=(FIG4_WIDTH_MM * MM, FIG4_HEIGHT_MM * MM), dpi=250)

    # Manual placement: reduced whitespace above and below panel a, and smaller gap to panel b
    ax_a = fig.add_axes([0.17, 0.50, 0.74, 0.41])
    ax_b = fig.add_axes([0.17, 0.11, 0.74, 0.25])

    plot_all_phase_locked_solutions_for_varying_w0_both(Sp0, Sm0, Sp1, Sm1, B1, K1, C1, w0_values_main, w0_values_inset, ax=ax_a, save_path=None)
    plot_bifurcation_ring_branches(Sp0, Sm0, Sp1, Sm1, B1, K1, C1, omega0_vals=omega0_vals, ax=ax_b, save_path=None)

    fig.savefig(save_path, format="pdf", dpi=300)
    plt.show()
    plt.close(fig)


# === Figure 5 plotting ===

def plot_figure_5(ring_sizes, sol_num, ring_sizes_calc, save_path="Fig5.pdf", show=True):
    """
    Plot the number of stable all-plus omega-zero ring solutions as a function of ring size.
    """

    MM = 1 / 25.4
    FIG5_WIDTH_MM = 88
    FIG5_HEIGHT_MM = 44

    fig, ax = plt.subplots(figsize=(FIG5_WIDTH_MM * MM, FIG5_HEIGHT_MM * MM), dpi=250)

    ax.scatter(ring_sizes, sol_num, s=14, marker='s', linewidth=0.4, label='Simulation')
    ax.step(ring_sizes, ring_sizes_calc, where='post', color='k', lw=0.8, label=r'$1 + 2 \left\lfloor \frac{s - 1}{4} \right\rfloor$')

    # Grid and ticks
    ax.grid(True, alpha=0.5, linewidth=0.35)
    y_min, y_max = ax.get_ylim()
    y_ticks = [y for y in range(int(y_min), int(y_max) + 1) if y % 2 == 1]
    ax.set_yticks(y_ticks)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which='minor', linewidth=0.25, alpha=0.3)

    # Labels and legend
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$N_{\mathrm{stable}}$')
    ax.legend(loc='upper left', fontsize=5.5, frameon=True, borderpad=0.3, handlelength=1.5, handletextpad=0.5)

    fig.savefig(save_path, format="pdf", dpi=300)

    if show:
        plt.show()

    plt.close(fig)
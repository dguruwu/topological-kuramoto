"""
=========================================================
Platonic Solid Generators for Topological Cell Complexes
=========================================================

This module provides helper routines for constructing oriented cell
complexes corresponding to the five Platonic solids (tetrahedron, cube,
octahedron, dodecahedron, icosahedron). The data is returned in a form
compatible with the Kuramoto dynamics pipeline (`find_all_plus_omega_zero_solutions`).

Functions
---------
Cell-complex construction
- extract_edges_from_faces   : Build unique edges from polygonal faces.
- build_face_adjacency       : Compute face adjacency via shared edges.
- edges_direction_in_face    : Determine local edge orientation in a face.
- orient_faces_adjacency     : Propagate orientation consistency across faces.
- get_platonic_solids        : Generate all five oriented Platonic solids (S0,S1,S2,S3).

Platonic-solid icon helpers (Fig. 6a insets)
- platonic_vertices          : Vertices for vector-rendered solid icons.
- convex_polyhedron_faces    : Group coplanar hull facets into true polygons.
- rotate_vertices            : Apply an Euler-angle rotation to a vertex array.
- icosahedron_geometry       : Vertices + faces for the icosahedron icon.
- tetrahedron_geometry       : Vertices + faces for a tetrahedron icon (currently unused).
- draw_platonic_solid        : Render a single shaded solid on a 3D inset axis.
- add_platonic_insets        : Place solid icons into a panel using data coords.

Figure 6 (stable counts and phase-locked solutions)
- plot_polyhedron_stable_counts : Panel a — N_stable per solid.
- plot_combined_ps_on_axes      : Panels b-g — F1/F2 curves and (zeta+, zeta-) scatters.
- make_figure_6                 : Assemble the full Figure 6 at publication size.

Figure 7 (omega_0 sweep)
- sweep_omega0_all_plus      : Sweep omega_0 and count stable phase-locked states.
- make_figure_7              : Assemble Figure 7 (N_stable curve + scatter panels).

Featured image and edge-dynamics utilities
- wrap_to_pi                       : Wrap angles to [-pi, pi).
- oriented_face_cycle              : Oriented 2-cycle C2 from ker(B2).
- projected_angles_from_windings   : Build projected angles from winding numbers.
- sparse_integer_shifts            : Sparse {-1,0,1} shift vectors with given sum.
- sparse_integer_shifts_weighted   : Sparse shifts with weights @ m = total.
- edge_dynamics_n1                 : n=1 topological Kuramoto edge dynamics.
- lift_equilibrium_from_windings   : Reconstruct an edge equilibrium from (z-, z+).
- simulate_edge_timeseries         : Integrate edge dynamics from a noisy equilibrium.
- make_intro_timeseries_pair       : Build two representative edge trajectories.
- plot_intro_edge_timeseries_stacked : Two stacked edge-phase time-series panels.
- rotation_matrix                  : 3D rotation matrix from Euler angles (degrees).
- edges_from_B1                    : Extract ordered edges from B1.
- cube_geometry_from_B1            : Cube vertices/edges/faces for the featured image.
- get_all_cube_stable_equilibria   : The 9 stable cube states over (z-, z+) in {-1,0,1}^2.
- boost_saturation                 : Increase saturation/brightness of an RGBA color.
- draw_colored_cube                : Draw one cube with phase-colored edges.
- make_featured_image_all_9_cube_states : 3x3 grid of all 9 stable cube states.
"""

import numpy as np
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm, ticker
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter
from matplotlib.colors import Normalize, rgb_to_hsv, hsv_to_rgb

from PIL import Image

from scipy.integrate import solve_ivp
from itertools import combinations, product

from core.dynamics import (
    compute_zeta_ranges_double,
    f_n_plus_one,
    f_n_minus_one,
    find_what_works_nd,
    check_stable_jacobians
)


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


def extract_edges_from_faces(faces):
    """
    Extract the unique set of edges from a list of polygonal faces.

    Each face is given as a cyclic list of vertex indices.
    The function enumerates all consecutive vertex pairs (including the closing edge),
    stores them as sorted tuples (u, v), and removes duplicates.

    Parameters
    ----------
    faces : list of list[int]
        List of polygonal faces, each represented by its ordered vertex indices.

    Returns
    -------
    edges : list of list[int]
        Sorted list of unique undirected edges present in the complex.

    Notes
    -----
    Edge orientation is ignored (stored as sorted vertex pairs).
    The output is suitable for building the 1-skeleton or incidence matrix B1.
    """

    edge_set = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            u, v = face[i], face[(i + 1) % n]
            edge = tuple(sorted((u, v)))
            edge_set.add(edge)
    return sorted([list(e) for e in edge_set])


def build_face_adjacency(faces):
    """
    Construct the adjacency relation between faces via shared edges.

    Each edge shared by exactly two faces defines an adjacency link.
    The function returns a mapping from each face index to its neighboring
    faces and their common edges.

    Parameters
    ----------
    faces : list of list[int]
        List of polygonal faces (each as an ordered list of vertex indices).

    Returns
    -------
    adjacency : dict[int, list[tuple[int, tuple[int,int]]]]
        Dictionary mapping each face index to a list of tuples
        (neighbor_index, shared_edge_vertices).

    Notes
    -----
    Faces sharing an edge are considered adjacent only if that edge
    occurs exactly twice in the complex.
    This structure supports propagation of orientation across faces.
    """
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            u, v = face[i], face[(i + 1) % n]
            edge_to_faces[tuple(sorted((u, v)))].append(fi)
    adjacency = defaultdict(list)
    for edge, f_list in edge_to_faces.items():
        if len(f_list) == 2:
            f1, f2 = f_list
            adjacency[f1].append((f2, edge))
            adjacency[f2].append((f1, edge))
    return adjacency


def edges_direction_in_face(face):
    """
    Compute the orientation of each edge within a single face.

    For each edge (u, v) in the ordered vertex list of the face,
    the direction is assigned +1 if (u < v) matches the traversal order,
    and -1 otherwise. The edge is stored as a sorted tuple (min, max).

    Parameters
    ----------
    face : list[int]
        Ordered list of vertex indices representing one face.

    Returns
    -------
    dir_map : dict[tuple[int,int], int]
        Mapping from edge (min, max) → orientation (+1 or -1).

    Notes
    -----
    Used internally to compare local edge orientations between neighboring faces.
    Assumes the face vertex order is consistent (counterclockwise or clockwise).
    """
    n = len(face)
    dir_map = {}
    for i in range(n):
        u, v = face[i], face[(i + 1) % n]
        edge = tuple(sorted((u, v)))
        # Direction is +1 if edge is (u->v) matches (min->max), else -1
        dir_map[edge] = 1 if (u < v) else -1
    return dir_map

def orient_faces_adjacency(faces, vertices):
    """
    Orient all faces of a polyhedral surface consistently.

    Starting from one arbitrarily oriented face, this function propagates
    orientation to all neighboring faces so that shared edges have opposite
    traversal directions (ensuring coherent global orientation).

    Parameters
    ----------
    faces : list of list[int]
        List of polygonal faces (each as a list of vertex indices).
    vertices : list[list[float]]
        Vertex coordinates (only used to define total count; geometry unused).

    Returns
    -------
    oriented : list[list[int]]
        List of faces with vertex orders adjusted for consistent orientation.

    Notes
    -----
    Uses breadth-first traversal of the face adjacency graph.
    Ensures each face's orientation is compatible with its neighbors
    (shared edges are traversed in opposite directions).
    """
    adjacency = build_face_adjacency(faces)
    oriented = [None] * len(faces)
    oriented[0] = faces[0]  # seed orientation with first face

    queue = deque([0])
    while queue:
        f_idx = queue.popleft()
        current_face = oriented[f_idx]
        current_dir = edges_direction_in_face(current_face)

        for neighbor_idx, shared_edge in adjacency[f_idx]:
            if oriented[neighbor_idx] is not None:
                continue

            neighbor_face = faces[neighbor_idx]
            neighbor_dir = edges_direction_in_face(neighbor_face)

            # If shared edge direction same in both faces, flip neighbor face
            if current_dir[tuple(sorted(shared_edge))] == neighbor_dir[tuple(sorted(shared_edge))]:
                oriented[neighbor_idx] = list(reversed(neighbor_face))
            else:
                oriented[neighbor_idx] = neighbor_face

            queue.append(neighbor_idx)

    return oriented

def get_platonic_solids():
    """
    Construct the complete topological data for all five Platonic solids.

    Each solid is returned as a tuple (S0, S1, S2, S3), where:
        S0: list of vertices,
        S1: list of edges (2-cells),
        S2: list of faces (3-cells),
        S3: list containing the single 3D volume (4-cell equivalent).

    The orientation of all faces is made globally consistent using
    `orient_faces_adjacency`.

    Returns
    -------
    solids : dict[str, tuple[list, list, list, list]]
        Dictionary mapping solid name → (vertices, edges, faces, volume).

    All 5 Platonic Solids included: 'tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron'

    Notes
    -----
    Vertex numbering starts at 1.
    """
    solids = {}

    # Tetrahedron
    tetra_vertices = [
        [1, 1, 1], [-1, -1, 1],
        [-1, 1, -1], [1, -1, -1]
    ]
    tetra_faces = [
        [1, 2, 3], [1, 4, 2],
        [3, 2, 4], [1, 3, 4]
    ]
    tetra_faces = orient_faces_adjacency(tetra_faces, tetra_vertices)
    tetra_edges = extract_edges_from_faces(tetra_faces)
    tetra_volume = [[1, 2, 3, 4]]
    solids['tetrahedron'] = (
        [[i + 1] for i in range(len(tetra_vertices))],
        tetra_edges,
        tetra_faces,
        tetra_volume
    )

    # Cube
    cube_vertices = [
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]
    ]
    cube_faces = [
        [1, 2, 3, 4], [5, 6, 7, 8],
        [1, 5, 6, 2], [4, 3, 7, 8],
        [1, 4, 8, 5], [2, 6, 7, 3]
    ]
    cube_faces = orient_faces_adjacency(cube_faces, cube_vertices)
    cube_edges = extract_edges_from_faces(cube_faces)
    cube_volume = [[1, 2, 3, 4, 5, 6, 7, 8]]
    solids['cube'] = (
        [[i + 1] for i in range(len(cube_vertices))],
        cube_edges,
        cube_faces,
        cube_volume
    )

    # Octahedron
    octa_vertices = [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    octa_faces = [
        [1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
        [3, 1, 6], [2, 3, 6], [4, 2, 6], [1, 4, 6]
    ]
    octa_faces = orient_faces_adjacency(octa_faces, octa_vertices)
    octa_edges = extract_edges_from_faces(octa_faces)
    octa_volume = [[1, 2, 3, 4, 5, 6]]
    solids['octahedron'] = (
        [[i + 1] for i in range(len(octa_vertices))],
        octa_edges,
        octa_faces,
        octa_volume
    )

    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Dodecahedron
    phi_inv = 1 / phi
    dodeca_vertices = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_inv, phi], [0, -phi_inv, phi], [0, phi_inv, -phi], [0, -phi_inv, -phi],
        [phi_inv, phi, 0], [-phi_inv, phi, 0], [phi_inv, -phi, 0], [-phi_inv, -phi, 0],
        [phi, 0, phi_inv], [-phi, 0, phi_inv], [phi, 0, -phi_inv], [-phi, 0, -phi_inv]
    ]
    dodeca_faces = [
        [1, 2, 3, 4, 5],
        [1, 6, 11, 12, 2],
        [2, 12, 17, 18, 3],
        [3, 18, 19, 7, 4],
        [4, 7, 8, 9, 5],
        [1, 5, 9, 10, 6],
        [6, 10, 14, 15, 11],
        [12, 11, 15, 16, 17],
        [18, 17, 16, 20, 19],
        [7, 19, 20, 13, 8],
        [9, 8, 13, 14, 10],
        [20, 16, 15, 14, 13]
    ]
    dodeca_faces = orient_faces_adjacency(dodeca_faces, dodeca_vertices)
    dodeca_edges = extract_edges_from_faces(dodeca_faces)
    dodeca_volume = [[i + 1 for i in range(len(dodeca_vertices))]]
    solids['dodecahedron'] = (
        [[i + 1] for i in range(len(dodeca_vertices))],
        dodeca_edges,
        dodeca_faces,
        dodeca_volume
    )

    # Icosahedron
    icosa_vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    icosa_faces = [
        [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 2],
        [2, 7, 3], [3, 8, 4], [4, 9, 5], [5, 10, 6], [6, 11, 2],
        [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 11, 6], [11, 7, 2],
        [7, 12, 8], [8, 12, 9], [9, 12, 10], [10, 12, 11], [11, 12, 7]
    ]
    icosa_faces = orient_faces_adjacency(icosa_faces, icosa_vertices)
    icosa_edges = extract_edges_from_faces(icosa_faces)
    icosa_volume = [[i + 1 for i in range(len(icosa_vertices))]]
    solids['icosahedron'] = (
        [[i + 1] for i in range(len(icosa_vertices))],
        icosa_edges,
        icosa_faces,
        icosa_volume
    )

    return solids


# === Platonic solid helpers for panel a ===

def platonic_vertices(name):
    """
    Return vertices for simple vector-rendered Platonic solids.
    These are used only for the small schematic icons in Fig. 5a.
    """
    phi = (1 + np.sqrt(5)) / 2

    if name == "Tetrahedron":
        V = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float)

    elif name == "Cube":
        V = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)

    elif name == "Octahedron":
        V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)

    elif name == "Dodecahedron":
        V = []
        V += [[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
        V += [[0, y / phi, z * phi] for y in [-1, 1] for z in [-1, 1]]
        V += [[x / phi, y * phi, 0] for x in [-1, 1] for y in [-1, 1]]
        V += [[x * phi, 0, z / phi] for x in [-1, 1] for z in [-1, 1]]
        V = np.array(V, dtype=float)

    elif name == "Icosahedron":
        V = []
        V += [[0, y, z * phi] for y in [-1, 1] for z in [-1, 1]]
        V += [[x, y * phi, 0] for x in [-1, 1] for y in [-1, 1]]
        V += [[x * phi, 0, z] for x in [-1, 1] for z in [-1, 1]]
        V = np.array(V, dtype=float)

    else:
        raise ValueError(f"Unknown solid name: {name}")

    V = V / np.max(np.linalg.norm(V, axis=1))
    return V


def convex_polyhedron_faces(V, tol=1e-7):
    """
    Build true polygonal faces from ConvexHull by grouping coplanar triangular facets.
    This avoids drawing artificial diagonals on cube and dodecahedron faces.
    """
    hull = ConvexHull(V)
    groups = []

    for simplex, eq in zip(hull.simplices, hull.equations):
        n = eq[:3]
        d = eq[3]
        n = n / np.linalg.norm(n)
        d = d / np.linalg.norm(eq[:3])

        found = False
        for group in groups:
            n0, d0, simplices = group
            if abs(np.dot(n, n0) - 1) < tol and abs(d - d0) < tol:
                simplices.append(simplex)
                found = True
                break
            if abs(np.dot(n, n0) + 1) < tol and abs(d + d0) < tol:
                simplices.append(simplex)
                found = True
                break

        if not found:
            groups.append([n, d, [simplex]])

    faces = []
    for n, d, simplices in groups:
        idxs = sorted(set(int(i) for simplex in simplices for i in simplex))
        pts = V[idxs]

        center = pts.mean(axis=0)

        # Build a local 2D basis in the face plane
        u = pts[0] - center
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        angles = np.arctan2((pts - center) @ v, (pts - center) @ u)
        ordered = [idxs[i] for i in np.argsort(angles)]

        faces.append([V[i] for i in ordered])

    return faces


def rotate_vertices(V, rx=0, ry=0, rz=0):
    rx, ry, rz = np.deg2rad([rx, ry, rz])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])

    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])

    return V @ (Rz @ Ry @ Rx).T


def icosahedron_geometry():
    phi = (1 + np.sqrt(5)) / 2

    V = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1]
    ], dtype=float)

    V = V / np.max(np.linalg.norm(V, axis=1))

    faces_idx = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    # a small built-in rotation usually makes the icon look nicer
    V = rotate_vertices(V, rx=8, ry=18, rz=-12)

    faces = [[V[i] for i in f] for f in faces_idx]
    return V, faces

def draw_platonic_solid(ax3d, name, elev=18, azim=-55):
    """
    Draw a small vector Platonic solid icon on a 3D inset axis.
    Faces are drawn as true polygons, not triangulated hull facets.
    """
    if name == "Tetrahedron":
        V = platonic_vertices(name)

        # stronger pre-rotation so it is not seen nearly face-on
        V = rotate_vertices(V, rx=35, ry=28, rz=18)

        faces = convex_polyhedron_faces(V)
        edgecolor = "0.25"
        lw = 0.22
    elif name == "Icosahedron":
        V, faces = icosahedron_geometry()
        edgecolor = "0.22"
        lw = 0.14
    else:
        V = platonic_vertices(name)
        faces = convex_polyhedron_faces(V)
        edgecolor = "0.25"
        lw = 0.22

    light = np.array([-0.4, -0.5, 1.0])
    light = light / np.linalg.norm(light)

    face_colors = []
    for face in faces:
        pts = np.array(face)
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n = n / np.linalg.norm(n)
        shade = 0.46 + 0.28 * max(0, abs(np.dot(n, light)))
        face_colors.append(str(min(shade, 0.78)))

    poly = Poly3DCollection(faces, facecolors=face_colors, edgecolor=edgecolor, linewidth=lw, alpha=1.0, zsort='average')
    ax3d.add_collection3d(poly)

    ax3d.set_xlim(-1.1, 1.1)
    ax3d.set_ylim(-1.1, 1.1)
    ax3d.set_zlim(-1.1, 1.1)
    ax3d.set_box_aspect([1, 1, 1])

    if name == "Tetrahedron":
        ax3d.view_init(elev=22, azim=-40)
        ax3d.set_proj_type("persp", focal_length=0.9)
    elif name == "Icosahedron":
        ax3d.view_init(elev=16, azim=-28)
        ax3d.set_proj_type("ortho")
    else:
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_proj_type("ortho")

    ax3d.set_proj_type("ortho")
    ax3d.axis("off")
    ax3d.patch.set_alpha(0)


def add_platonic_insets(fig, ax, solid_names, x_positions, y_positions, icon_size_mm=13):
    """
    Add vector-rendered Platonic solid icons into panel a.
    Icons are positioned using the data coordinates of the main axis.
    """
    fig.canvas.draw()
    fig_w_mm = fig.get_size_inches()[0] * 25.4
    fig_h_mm = fig.get_size_inches()[1] * 25.4
    icon_w = icon_size_mm / fig_w_mm
    icon_h = icon_size_mm / fig_h_mm

    orientations = {
        "Tetrahedron": (24, 18),
        "Cube": (20, -55),
        "Octahedron": (18, -40),
        "Dodecahedron": (18, -35),
        "Icosahedron": (18, -35)
    }

    for name, x, y in zip(solid_names, x_positions, y_positions):
        x_disp, y_disp = ax.transData.transform((x, y))
        x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
        ax_icon = fig.add_axes([x_fig - icon_w / 2, y_fig - icon_h / 2, icon_w, icon_h], projection="3d")
        elev, azim = orientations[name]
        draw_platonic_solid(ax_icon, name, elev=elev, azim=azim)


# === Figure 6a plotting ===

def plot_polyhedron_stable_counts(roots_ps_n1, ax=None, save_path=None, add_solids=True):
    """
    Plot the number of stable roots for the Platonic solids.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(180/25.4, 28/25.4), dpi=250)
    else:
        fig = ax.figure

    x = [1, 2, 3, 4, 5]
    y = [len(roots) for roots in roots_ps_n1]   # number of stable roots per solid

    solid_names = ['Tetrahedron', 'Cube', 'Octahedron', 'Dodecahedron', 'Icosahedron']

    ax.plot(x, y, marker='o', linestyle='-', color='black', markersize=2.0, linewidth=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(solid_names, rotation=0)
    ax.set_ylabel(r"$N_{\mathrm{stable}}$")

    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(-1.5, 47)

    ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    ax.minorticks_on()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis='both')

    ax.text(-0.055, 1.00, "a", transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold", clip_on=False)

    if add_solids:
        add_platonic_insets(fig, ax, solid_names, x, [30, 31, 31, 30, 30], icon_size_mm=13)

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300)


# === Figure 6b-g plotting ===

def plot_combined_ps_on_axes(ps_data_list, ax1, ax2, axes2, resolution=1500):
    """
    Plot panels b-g for the Platonic solid phase-locked solution data.
    """
    
    # Colors from viridis
    viridis = cm.get_cmap('viridis', 256)
    colors = [
        viridis(1.0),   # yellow -> tetrahedron
        viridis(0.5),   # middle -> cube & octahedron
        viridis(0.2)    # dark -> dodecahedron & icosahedron
    ]

    line_styles = [
        '-',          # tetrahedron
        '-', '--',    # cube & octahedron
        '-', '--'     # dodecahedron & icosahedron
    ]

    # Flatten colors list to match solids
    solid_colors = [
        colors[0],            # tetrahedron
        colors[1], colors[1], # cube, octahedron
        colors[2], colors[2]  # dodeca, icosahedron
    ]

    # Row 1: F1 and F2 side by side
    for idx, ps_data in enumerate(ps_data_list):
        color = solid_colors[idx]
        ls = line_styles[idx]
        
        # Compute zeta ranges
        (zeta_plus_range, zeta_minus_range) = compute_zeta_ranges_double(
            ps_data['psi_plus_sp'], ps_data['psi_minus_sp'],
            ps_data['K1'], ps_data['K2'],
            ps_data['C1'], ps_data['C2']
        )
        zp_min, zp_max = zeta_plus_range
        zm_min, zm_max = zeta_minus_range
        zeta_plus_vals = np.linspace(zp_min, zp_max, resolution)
        zeta_minus_vals = np.linspace(zm_min, zm_max, resolution)
        
        A_plus = (np.linalg.inv(ps_data['K2']) @ ps_data['C2']).flatten()
        A_minus = (np.linalg.inv(ps_data['K1']) @ ps_data['C1'].T).flatten()

        # Evaluate F1
        F1_vals = []
        for zp in zeta_plus_vals:
            psi_plus = ps_data['psi_plus_sp'].flatten() + A_plus * zp
            f_plus = f_n_plus_one(psi_plus, ps_data['S2_plus'], ps_data['S2_minus'])
            F1_vals.append(float(np.sum(ps_data['C2'].flatten() * f_plus.flatten())))
        F1_vals = np.array(F1_vals)/(2*np.pi)

        # Evaluate F2
        F2_vals = []
        for zm in zeta_minus_vals:
            psi_minus = ps_data['psi_minus_sp'].flatten() + A_minus * zm
            f_minus = f_n_minus_one(psi_minus, ps_data['S0_plus'], ps_data['S0_minus'])
            F2_vals.append(float(np.sum(ps_data['C1'].flatten() * f_minus.flatten())))
        F2_vals = np.array(F2_vals)/(2*np.pi)

        # Plot F1 and F2
        ax1.plot(zeta_plus_vals, F1_vals, color=color, linestyle=ls, label=ps_data['label'], lw=0.65)
        ax2.plot(zeta_minus_vals, F2_vals, color=color, linestyle=ls, label=ps_data['label'], lw=0.65)

        # Plot root intersections as dots
        for idx_r, (z_plus, z_minus, zeta_plus, zeta_minus) in enumerate(ps_data['roots']):
            is_stable = ps_data['stable_indices'] is not None and idx_r in ps_data['stable_indices']
            
            F1_val = float(np.sum(ps_data['C2'].flatten() * f_n_plus_one(
                ps_data['psi_plus_sp'].flatten() + A_plus * zeta_plus,
                ps_data['S2_plus'], ps_data['S2_minus']
            ).flatten()))/(2*np.pi)
            
            F2_val = float(np.sum(ps_data['C1'].flatten() * f_n_minus_one(
                ps_data['psi_minus_sp'].flatten() + A_minus * zeta_minus,
                ps_data['S0_plus'], ps_data['S0_minus']
            ).flatten()))/(2*np.pi)

            ax1.plot(zeta_plus, F1_val, 'o', color='black' if is_stable else 'white', markeredgecolor='black', ms=2.0, markeredgewidth=0.4)
            ax2.plot(zeta_minus, F2_val, 'o', color='black' if is_stable else 'white', markeredgecolor='black', ms=2.0, markeredgewidth=0.4)

    # Dashed horizontal lines at integer y
    for z in range(-5, 6):
        ax1.axhline(z, color='0.65', linestyle='dashed', linewidth=0.35, zorder=0)
    for z in range(-5, 6):
        ax2.axhline(z, color='0.65', linestyle='dashed', linewidth=0.35, zorder=0)

    ax1.set_xlabel(r'$\zeta^{[+]}$')
    ax1.set_ylabel(r'$C_2^\top f_2(\psi^{[+]})/(2\pi)$')
    ax2.set_xlabel(r'$\zeta^{[-]}$')
    ax2.set_ylabel(r'$C_1 f_0(\psi^{[-]})/(2\pi)$')

    ax1.tick_params(axis='both')
    ax2.tick_params(axis='both')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.text(-0.12, 1.03, "b", transform=ax1.transAxes, ha="left", va="bottom", fontsize=7, fontweight="bold", clip_on=False)
    ax2.text(-0.12, 1.03, "c", transform=ax2.transAxes, ha="left", va="bottom", fontsize=7, fontweight="bold", clip_on=False)

    ax1.legend(fontsize=5.2, frameon=True, borderpad=0.3, handlelength=1.4, handletextpad=0.45)

    # Row 2: scatter plots of zeta_plus vs zeta_minus for last 4 solids (skip tetrahedron since there is no multistability)
    panel_labels = ["d", "e", "f", "g"]
    for idx, ps_data in enumerate(ps_data_list[1:]):  # skip tetrahedron
        ax = axes2[idx]
        for idx_r, (z_plus, z_minus, zeta_plus, zeta_minus) in enumerate(ps_data['roots']):
            is_stable = ps_data['stable_indices'] is not None and idx_r in ps_data['stable_indices']
            ax.plot(zeta_plus, zeta_minus, 'o', color='black' if is_stable else 'white', markeredgecolor='black', ms=1.8, markeredgewidth=0.35)

        ax.set_title(ps_data['label'], fontsize=6)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$\zeta^{[+]}$')

        if idx == 0:
            ax.set_ylabel(r'$\zeta^{[-]}$')

        ax.set_xticks(np.arange(-1, 1.01, 0.5))
        ax.set_yticks(np.arange(-1, 1.01, 0.5))
        ax.grid(True, which='major', linewidth=0.35)
        ax.minorticks_on()
        ax.grid(True, which='minor', linewidth=0.18)
        ax.tick_params(axis='both')
        ax.text(-0.18, 1.03, panel_labels[idx], transform=ax.transAxes, ha="left", va="bottom", fontsize=7, fontweight="bold", clip_on=False)


# === Combined Figure 6 ===

def make_figure_6(roots_ps_n1, ps_data_list, resolution=1500, save_path="Fig6final.pdf", show=True):
    """
    Make the final combined Figure 5 at publication size.
    """
    MM = 1 / 25.4
    FIG5_WIDTH_MM = 180
    FIG5_HEIGHT_MM = 132

    fig = plt.figure(figsize=(FIG5_WIDTH_MM * MM, FIG5_HEIGHT_MM * MM), dpi=250)

    ax_a = fig.add_axes([0.065, 0.805, 0.91, 0.16])

    ax_b = fig.add_axes([0.095, 0.455, 0.39, 0.27])
    ax_c = fig.add_axes([0.585, 0.455, 0.39, 0.27])

    bottom_y = 0.085
    bottom_w = 0.18
    bottom_h = bottom_w * FIG5_WIDTH_MM / FIG5_HEIGHT_MM
    bottom_gap = 0.053

    ax_d = fig.add_axes([0.080, bottom_y, bottom_w, bottom_h])
    ax_e = fig.add_axes([0.080 + bottom_w + bottom_gap, bottom_y, bottom_w, bottom_h])
    ax_f = fig.add_axes([0.080 + 2 * (bottom_w + bottom_gap), bottom_y, bottom_w, bottom_h])
    ax_g = fig.add_axes([0.080 + 3 * (bottom_w + bottom_gap), bottom_y, bottom_w, bottom_h])
    axes_bottom = [ax_d, ax_e, ax_f, ax_g]

    plot_polyhedron_stable_counts(roots_ps_n1, ax=ax_a, save_path=None, add_solids=True)
    plot_combined_ps_on_axes(ps_data_list, ax_b, ax_c, axes_bottom, resolution=resolution)

    fig.savefig(save_path, format="pdf", dpi=300)

    if show:
        plt.show()

    plt.close(fig)




# === Figure 7 ===

def sweep_omega0_all_plus(B1, B2, K1, K2, C1, C2, omega0_vals, S0_plus, S2_plus, S0_minus=None, S2_minus=None, tol=1e-7):
    """
    Sweep over omega_0 for an 'all-plus' partition on any Platonic solid.
    Uses the general `find_what_works_nd` solver to identify all phase-locked
    states and then checks stability for each omega_0.
    """
    S0_minus = S0_minus or []
    S2_minus = S2_minus or []

    stable_counts = []
    all_results = []
    all_omegas = []

    for w0 in omega0_vals:
        print(fr"$\omega_0$ = {w0:.3f}")

        # Define inhomogeneous omega distribution
        omega = np.zeros(B1.shape[1])
        #omega[-2:] = [+w0, -w0]
        omega[-1] = + w0
        omega[-4] = + w0
        all_omegas.append(omega)

        # Find all candidate roots
        roots = find_what_works_nd(S0_plus, S0_minus, S2_plus, S2_minus, B1, B2, K1, K2, C1, C2, tol=tol, omega=omega)

        if not roots:
            stable_counts.append(0)
            all_results.append(([], []))
            continue

        # Stability check
        stable_indices, *_ = check_stable_jacobians(roots, omega, B1, B2, K1, K2, C1, C2, S0_plus, S0_minus, S2_plus, S2_minus)

        stable_counts.append(len(stable_indices))
        all_results.append((roots, stable_indices))

    return stable_counts, all_results, all_omegas

def make_figure_7(omega0_vals, stable_counts, all_results, all_omegas, selected_omegas, save_path="Fig7.pdf", show=True):
    """
    Styled composite plot for fixed-point analysis.

    Top plot:
        - N_stable(omega_0) with discontinuities when count drops
        - Blue squares connected by solid segments otherwise

    Bottom plots:
        - (zeta_plus, zeta_minus) scatter diagrams for manually selected omega_0
        - Grid: solid gray lines
        - omega_0 label in bordered box, upper right corner
    """

    MM = 1 / 25.4
    FIG7_WIDTH_MM = 180
    FIG7_HEIGHT_MM = 90

    fig = plt.figure(figsize=(FIG7_WIDTH_MM * MM, FIG7_HEIGHT_MM * MM), dpi=250)

    # === TOP PLOT ===
    ax_top = fig.add_axes([0.070, 0.690, 0.905, 0.250])

    x = np.asarray(omega0_vals, dtype=float)
    y = np.asarray(stable_counts, dtype=float)

    # sort by omega_0 in case the array isn't ordered
    order = np.argsort(x)
    x, y = x[order], y[order]
    all_results_sorted = [all_results[j] for j in order]

    # Discontinuity detection. break the line when there's a drop
    discontinuities = np.where(np.diff(y) < 0)[0]
    start_idx = 0
    for d in discontinuities:
        ax_top.plot(x[start_idx:d+1], y[start_idx:d+1], color='C0', marker='s', markersize=1.8, markeredgewidth=0.0, linewidth=0.8, zorder=3)
        start_idx = d + 1

    # last segment
    ax_top.plot(x[start_idx:], y[start_idx:], color='C0', marker='s', markersize=1.8, markeredgewidth=0.0, linewidth=0.8, zorder=3)

    ax_top.set_xlabel(r'$\omega_0$')
    ax_top.set_ylabel(r"$N_{\mathrm{stable}}$")
    ax_top.grid(True, linewidth=0.35, alpha=0.65)
    ax_top.tick_params(axis='both')
    ax_top.yaxis.set_major_locator(MultipleLocator(2))
    ax_top.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax_top.yaxis.set_minor_locator(MultipleLocator(1))
    ax_top.grid(which='minor', linewidth=0.20, alpha=0.35)

    ax_top.text(-0.055, 1.00, "a", transform=ax_top.transAxes, ha="left", va="top", fontsize=7, fontweight="bold", clip_on=False)

    # === BOTTOM SCATTER PLOTS ===
    panel_labels = ["b", "c", "d", "e"]

    bottom_y = 0.110
    bottom_w = 0.185
    bottom_h = bottom_w * FIG7_WIDTH_MM / FIG7_HEIGHT_MM
    bottom_gap = 0.055
    bottom_x0 = 0.070

    axes = []
    for i in range(len(selected_omegas)):
        ax = fig.add_axes([bottom_x0 + i * (bottom_w + bottom_gap), bottom_y, bottom_w, bottom_h])
        axes.append(ax)

        # find nearest index for this omega_0
        w_target = selected_omegas[i]
        idx = np.argmin(np.abs(x - w_target))
        roots, stable_indices = all_results_sorted[idx] if idx < len(all_results_sorted) else ([], [])
        omega_val = x[idx]

        if not roots:
            ax.text(0.5, 0.5, 'No roots', ha='center', va='center', fontsize=6)
            ax.axis('off')
            continue

        # scatter roots
        for r_idx, (z_plus, z_minus, zeta_plus, zeta_minus) in enumerate(roots):
            is_stable = r_idx in stable_indices
            ax.plot(float(zeta_plus), float(zeta_minus), marker='o', markersize=2.4, markerfacecolor='black' if is_stable else 'white', markeredgecolor='black', markeredgewidth=0.45, linestyle='None')

        # omega_0 label in bordered box (upper right)
        ax.plot([], [], ' ', label=fr'$\omega_0 = {omega_val:.2f}$')
        leg = ax.legend(fontsize=5.5, loc='upper right', bbox_to_anchor=(1, 1), borderpad=0.2, handlelength=0, handletextpad=0, borderaxespad=0.1, frameon=True, fancybox=False, edgecolor='black', facecolor='white')
        leg.get_frame().set_linewidth(0.4)

        # axis style
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal')
        ax.grid(True, color='gray', linestyle='-', linewidth=0.35, alpha=0.7)
        ax.tick_params(axis='both')
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        # labels
        if i == 0:
            ax.set_ylabel(r'$\zeta_-$')
        else:
            ax.set_ylabel('')
        ax.set_xlabel(r'$\zeta_+$')

        ax.text(-0.14, 1.02, panel_labels[i], transform=ax.transAxes, ha="left", va="bottom", fontsize=7, fontweight="bold", clip_on=False)

    fig.savefig(save_path, format="pdf", dpi=300)

    if show:
        plt.show()

    plt.close(fig)



# === Featured image ===

def wrap_to_pi(x):
    """
    Wrap angles to the interval [-pi, pi).

    Parameters
    ----------
    x : array_like
        Angle or array of angles in radians.

    Returns
    -------
    ndarray
        Wrapped angles with values in [-pi, pi).
    """

    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2 * np.pi) - np.pi

def oriented_face_cycle(B2, tol=1e-10):
    """
    Return the oriented face-cycle vector C2 satisfying B2 @ C2 = 0.
    For the hand-defined cube this is [1, -1, 1, -1, -1, 1].
    """
    _, _, Vt = np.linalg.svd(B2.astype(float), full_matrices=True)
    c = Vt[-1]
    c = c / np.max(np.abs(c))
    c[np.abs(c) < tol] = 0
    c = np.round(c).astype(int)

    nz = np.flatnonzero(c)
    if len(nz) > 0 and c[nz[0]] < 0:
        c = -c

    return c

def projected_angles_from_windings(z_minus, z_plus, N0, N2, C2=None):
    """
    Construct all-normal projected angles from lower and upper winding numbers.

    Parameters
    ----------
    z_minus : int
        Lower-dimensional winding number associated with node-side interactions.
    z_plus : int
        Higher-dimensional winding number associated with face-side interactions.
    N0 : int
        Number of 0-cells.
    N2 : int
        Number of 2-cells.
    C2 : ndarray, optional
        Oriented face-cycle vector. If omitted, the all-ones orientation is used.

    Returns
    -------
    theta_minus : ndarray
        Uniform lower projected angles theta^[-].
    theta_plus : ndarray
        Oriented upper projected angles theta^[+].
    """
    if C2 is None:
        C2 = np.ones(N2)

    C2 = np.asarray(C2, dtype=float)

    theta_minus = (2 * np.pi * z_minus / N0) * np.ones(N0)
    theta_plus  = (2 * np.pi * z_plus  / N2) * C2

    return theta_minus, theta_plus

def sparse_integer_shifts(n, total):
    """
    Generate sparse integer shift vectors with prescribed total sum.

    Parameters
    ----------
    n : int
        Length of each shift vector.
    total : int
        Desired sum of the vector entries.

    Returns
    -------
    list of ndarray
        Integer vectors with entries in {-1, 0, 1}, sum equal to `total`,
        and the minimal number of nonzero entries.
    """

    if total == 0:
        return [np.zeros(n, dtype=int)]

    sign = 1 if total > 0 else -1
    k = abs(total)

    out = []
    for idx in combinations(range(n), k):
        m = np.zeros(n, dtype=int)
        m[list(idx)] = sign
        out.append(m)
    return out

def sparse_integer_shifts_weighted(weights, total):
    """
    Generate sparse integer shift vectors m with entries in {-1, 0, 1}
    satisfying weights @ m = total.
    """
    weights = np.asarray(weights, dtype=int)
    total = int(total)
    n = len(weights)

    if total == 0:
        return [np.zeros(n, dtype=int)]

    out = []
    for nnz in range(1, n + 1):
        for idx in combinations(range(n), nnz):
            for signs in product([-1, 1], repeat=nnz):
                m = np.zeros(n, dtype=int)
                m[list(idx)] = signs

                if int(weights @ m) == total:
                    out.append(m)

        if out:
            return out

    return out

def edge_dynamics_n1(t, theta, B1, B2):
    """
    Evaluate the topological Kuramoto edge dynamics for n = 1.

    Parameters
    ----------
    t : float
        Current integration time. Included for compatibility with `solve_ivp`;
        the dynamics are autonomous and do not depend explicitly on time.
    theta : array_like
        Phase variables defined on edges.
    B1 : ndarray
        Node-edge incidence matrix.
    B2 : ndarray
        Edge-face incidence matrix.

    Returns
    -------
    ndarray
        Time derivative of the edge phase variables.
    """
    theta = np.asarray(theta, dtype=float)
    return -B1.T @ np.sin(B1 @ theta) - B2 @ np.sin(B2.T @ theta)


# Reconstruct edge equilibria from winding numbers
def lift_equilibrium_from_windings(B1, B2, z_minus, z_plus, linear_tol=1e-10, dyn_tol=1e-9):
    """
    Reconstruct an edge-phase equilibrium from prescribed winding numbers,
    i. e., find an edge-phase equilibrium theta such that
        B1 theta   = theta^- - 2 pi m^-
        B2^T theta = theta^+ - 2 pi m^+
    for some integer shift vectors m^-, m^+,
    and verify that theta is an equilibrium of the full edge ODE.

    Parameters
    ----------
    B1 : ndarray
        Node-edge incidence matrix.
    B2 : ndarray
        Edge-face incidence matrix.
    z_minus : int
        Lower-dimensional winding number.
    z_plus : int
        Higher-dimensional winding number.
    linear_tol : float, optional
        Maximum allowed residual for the lifted linear consistency equations.
    dyn_tol : float, optional
        Maximum allowed residual for the full edge dynamics.

    Returns
    -------
    theta_star : ndarray
        Edge-phase equilibrium wrapped to [-pi, pi).
    info : dict
        Diagnostic information, including chosen integer shifts, residuals,
        projected angles, and auxiliary sine variables.

    Raises
    ------
    RuntimeError
        If no lifted equilibrium satisfies the requested residual tolerances.
    """
    N0 = B1.shape[0]
    N2 = B2.shape[1]

    C2 = oriented_face_cycle(B2)

    theta_minus, theta_plus = projected_angles_from_windings(z_minus, z_plus, N0, N2, C2=C2)
    A = np.vstack([B1, B2.T])

    minus_shifts = sparse_integer_shifts(N0, z_minus)
    plus_shifts = sparse_integer_shifts_weighted(C2, z_plus)

    best = None
    best_score = np.inf

    for m_minus in minus_shifts:
        rhs_minus = theta_minus - 2 * np.pi * m_minus

        for m_plus in plus_shifts:
            rhs_plus = theta_plus - 2 * np.pi * m_plus
            b = np.concatenate([rhs_minus, rhs_plus])

            theta_star, *_ = np.linalg.lstsq(A, b, rcond=None)
            theta_star = wrap_to_pi(theta_star)

            lin_res = np.linalg.norm(A @ theta_star - b)
            dyn_res = np.linalg.norm(edge_dynamics_n1(0.0, theta_star, B1, B2))

            score = lin_res + dyn_res
            if score < best_score:
                best_score = score
                best = (theta_star, m_minus, m_plus, lin_res, dyn_res)

    theta_star, m_minus, m_plus, lin_res, dyn_res = best

    if lin_res > linear_tol or dyn_res > dyn_tol:
        raise RuntimeError(
            f"No consistent lifted equilibrium found for "
            f"(z_minus,z_plus)=({z_minus},{z_plus}); "
            f"best linear residual={lin_res:.3e}, dynamics residual={dyn_res:.3e}"
        )

    info = {
        "z_minus": z_minus,
        "z_plus": z_plus,
        "m_minus": m_minus,
        "m_plus": m_plus,
        "linear_residual": lin_res,
        "dynamics_residual": dyn_res,
        "theta_minus": B1 @ theta_star,
        "theta_plus": B2.T @ theta_star,
        "psi_minus": np.sin(B1 @ theta_star),
        "psi_plus": np.sin(B2.T @ theta_star),
    }
    return theta_star, info

def simulate_edge_timeseries(theta_star, B1, B2, tmax=25, npts=2000, noise=1e-2, seed=0):
    """
    Simulate n = 1 edge dynamics from a noisy perturbation of an equilibrium.

    Parameters
    ----------
    theta_star : array_like
        Edge-phase equilibrium used as the initial reference state.
    B1 : ndarray
        Node-edge incidence matrix.
    B2 : ndarray
        Edge-face incidence matrix.
    tmax : float, optional
        Final integration time.
    npts : int, optional
        Number of time points at which to store the solution.
    noise : float, optional
        Standard deviation of the Gaussian perturbation added to `theta_star`.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    scipy.integrate.OdeResult
        Numerical solution returned by `solve_ivp`.
    """
    rng = np.random.default_rng(seed)
    theta0 = theta_star + noise * rng.standard_normal(len(theta_star))

    t_eval = np.linspace(0, tmax, npts)
    sol = solve_ivp(lambda t, th: edge_dynamics_n1(t, th, B1, B2), (0, tmax), theta0, t_eval=t_eval, rtol=1e-9, atol=1e-11, method="RK45")
    return sol

def make_intro_timeseries_pair(B1, B2, branch1, branch2, tmax=25, npts=2000, noise=1e-2, seed1=1, seed2=2):
    """
    Build two representative edge-dynamics trajectories for introductory figures.

    Parameters
    ----------
    B1 : ndarray
        Node-edge incidence matrix.
    B2 : ndarray
        Edge-face incidence matrix.
    branch1 : tuple of int
        Winding-number pair `(z_minus, z_plus)` for the first trajectory.
    branch2 : tuple of int
        Winding-number pair `(z_minus, z_plus)` for the second trajectory.
    tmax : float, optional
        Final integration time for both simulations.
    npts : int, optional
        Number of time points stored for each simulation.
    noise : float, optional
        Standard deviation of the Gaussian perturbation added to each equilibrium.
    seed1 : int, optional
        Random seed for the first trajectory.
    seed2 : int, optional
        Random seed for the second trajectory.

    Returns
    -------
    dict
        Dictionary containing the two equilibria, diagnostic dictionaries,
        and simulated trajectories.
    """
    theta_star_1, info1 = lift_equilibrium_from_windings(B1, B2, *branch1)
    theta_star_2, info2 = lift_equilibrium_from_windings(B1, B2, *branch2)

    sol1 = simulate_edge_timeseries(theta_star_1, B1, B2, tmax=tmax, npts=npts, noise=noise, seed=seed1)
    sol2 = simulate_edge_timeseries(theta_star_2, B1, B2, tmax=tmax, npts=npts, noise=noise, seed=seed2)

    return {
        "theta_star_1": theta_star_1,
        "theta_star_2": theta_star_2,
        "info1": info1,
        "info2": info2,
        "sol1": sol1,
        "sol2": sol2,
    }

def plot_intro_edge_timeseries_stacked(data, save_path="Fig_example_timeseries.pdf", figsize=(5.6, 4.0), line_color="#003366"):
    """
    Plot two stacked edge-phase time series panels.

    Parameters
    ----------
    data : dict
        Output of `make_intro_timeseries_pair`, containing `sol1`, `sol2`,
        `info1`, and `info2`.
    save_path : str, optional
        Path where the figure is saved.
    figsize : tuple of float, optional
        Figure size in inches.
    line_color : str, optional
        Color used for all edge trajectories.

    Returns
    -------
    None
        Saves the figure to `save_path` and displays it.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0})

    sol1, sol2 = data["sol1"], data["sol2"]
    info1, info2 = data["info1"], data["info2"]

    n_edges = sol1.y.shape[0]

    panels = [(axes[0], sol1, info1, "IC 1"), (axes[1], sol2, info2, "IC 2")]

    for ax, sol, info, ic_label in panels:
        for j in range(n_edges):
            ax.plot(sol.t, sol.y[j], color=line_color, lw=1.3)

        ax.set_ylabel(r"$\theta_{[j,k]}(t)$", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


    axes[1].set_xlabel(r"$t$", fontsize=16)

    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(axis="x", pad=6)

    if np.min(sol.y) < -1.8:
        axes[1].set_ylim([-2.5, 2.5])

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def rotation_matrix(rx=0, ry=0, rz=0):
    """
    Build a 3D rotation matrix from Euler angles in degrees.
    """
    rx, ry, rz = np.deg2rad([rx, ry, rz])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])

    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])

    return Rz @ Ry @ Rx


def edges_from_B1(B1):
    """
    Extract ordered cube edges directly from the incidence matrix B1.
    Column j gives edge j.
    """
    edges = []
    for j in range(B1.shape[1]):
        tail = np.where(B1[:, j] == -1)[0]
        head = np.where(B1[:, j] == +1)[0]

        if len(tail) != 1 or len(head) != 1:
            raise ValueError(f"Column {j} of B1 is not a valid oriented edge.")

        edges.append((tail[0], head[0]))   # 0-based vertex indices
    return edges

def cube_geometry_from_B1(B1, scale=1.0):
    """
    Return cube vertices, faces, and edges.
    The edge ordering is taken directly from B1, then edges 7 and 10 are
    swapped for drawing so the phase colors land on the intended cube edges.
    """

    # Vertex positions: row i corresponds to vertex i+1 in B1
    V = np.array([
        [0, 0, 1],  # vertex 1
        [1, 0, 1],  # vertex 2
        [1, 1, 1],  # vertex 3
        [0, 1, 1],  # vertex 4
        [0, 0, 0],  # vertex 5
        [1, 0, 0],  # vertex 6
        [1, 1, 0],  # vertex 7
        [0, 1, 0],  # vertex 8
    ], dtype=float)

    V = V - np.array([0.5, 0.5, 0.5])
    V *= scale

    # Faces are still defined by vertices
    faces = [
        [0, 1, 2, 3],  # top
        [4, 5, 6, 7],  # bottom
        [0, 1, 5, 4],  # front
        [1, 2, 6, 5],  # right
        [2, 3, 7, 6],  # back
        [3, 0, 4, 7],  # left
    ]

    edges = edges_from_B1(B1)

    # Drawing-only correction: swap displayed edges 7 and 10.
    # Edge numbers are 1-based in the debug plot; Python indices are 0-based.
    edge_perm = list(range(12))
    edge_perm[6], edge_perm[9] = edge_perm[9], edge_perm[6]
    edges = [edges[k] for k in edge_perm]

    return V, edges, faces


def get_all_cube_stable_equilibria(B1, B2):
    """
    Construct the 9 stable cube equilibria corresponding to
    (z_-, z_+) in {-1,0,1} x {-1,0,1}.
    """
    branches = [
        (-1,  1), (0,  1), (1,  1),
        (-1,  0), (0,  0), (1,  0),
        (-1, -1), (0, -1), (1, -1),
    ]

    states = []
    for z_minus, z_plus in branches:
        theta_star, info = lift_equilibrium_from_windings(B1, B2, z_minus, z_plus)
        states.append({
            "branch": (z_minus, z_plus),
            "theta_star": wrap_to_pi(theta_star),
            "info": info,
        })

    return states


def boost_saturation(color, sat_factor=1.35, val_factor=1.08):
    """
    Increase saturation and slightly increase brightness of an RGBA color.
    """
    rgba = np.array(color)
    rgb = rgba[:3]
    hsv = rgb_to_hsv(rgb.reshape(1, 1, 3)).reshape(3)
    hsv[1] = np.clip(hsv[1] * sat_factor, 0, 1)
    hsv[2] = np.clip(hsv[2] * val_factor, 0, 1)
    rgb_new = hsv_to_rgb(hsv.reshape(1, 1, 3)).reshape(3)
    return (rgb_new[0], rgb_new[1], rgb_new[2], rgba[3])


def draw_colored_cube(ax, theta, B1, center=(0, 0, 0), scale=1.0, rotation=(22, -24, 8), cmap_name="twilight_shifted"):
    """
    Draw one cube with edges colored according to edge phases.
    Edge ordering is taken directly from B1.
    """
    theta = wrap_to_pi(np.asarray(theta, dtype=float))
    V, edges, faces = cube_geometry_from_B1(B1, scale=scale)

    R = rotation_matrix(*rotation)
    V = V @ R.T
    V = V + np.asarray(center)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    raw_edge_colors = [cmap(norm(val)) for val in theta]
    edge_colors = [boost_saturation(c, sat_factor=1.45, val_factor=1.10) for c in raw_edge_colors]

    # very subtle faces, mostly there to give depth
    face_polys = [[V[i] for i in face] for face in faces]
    face_collection = Poly3DCollection(
        face_polys,
        facecolors=(0.88, 0.91, 0.98, 0.10),
        edgecolor=(0.70, 0.76, 0.84, 0.04),
        linewidth=0.20
    )
    ax.add_collection3d(face_collection)

    segments = [[V[i], V[j]] for i, j in edges]

    # layered glow for a real bloom effect
    glow1_colors = [(r, g, b, 0.06) for r, g, b, _ in edge_colors]
    glow2_colors = [(r, g, b, 0.12) for r, g, b, _ in edge_colors]
    glow3_colors = [(r, g, b, 0.22) for r, g, b, _ in edge_colors]

    glow1 = Line3DCollection(segments, colors=glow1_colors, linewidths=13.0)
    glow2 = Line3DCollection(segments, colors=glow2_colors, linewidths=8.5)
    glow3 = Line3DCollection(segments, colors=glow3_colors, linewidths=5.5)

    ax.add_collection3d(glow1)
    ax.add_collection3d(glow2)
    ax.add_collection3d(glow3)

    # bright core edge
    lines = Line3DCollection(segments, colors=edge_colors, linewidths=3.4)
    ax.add_collection3d(lines)

def make_featured_image_all_9_cube_states(B1, B2, save_path="featured_image_suggestion.png", background="#07122b"):
    """
    Create a 1200x675 RGB featured image showing all 9 stable cube states
    in a regular 3x3 grid.
    """
    states = get_all_cube_stable_equilibria(B1, B2)

    fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=background)  # 1200 x 675 px

    # tighter 3x3 layout
    left = 0.045
    right = 0.045
    bottom = 0.07
    top = 0.07
    hgap = 0.012
    vgap = 0.030

    cell_w = (1 - left - right - 2 * hgap) / 3
    cell_h = (1 - top - bottom - 2 * vgap) / 3

    k = 0
    for row in range(3):
        for col in range(3):
            x0 = left + col * (cell_w + hgap)
            y0 = 1 - top - (row + 1) * cell_h - row * vgap

            ax = fig.add_axes([x0, y0, cell_w, cell_h], projection="3d", facecolor=background)

            draw_colored_cube(ax, states[k]["theta_star"], B1, center=(0, 0, 0), scale=2.3, rotation=(22, -24, 8), cmap_name="twilight_shifted")

            ax.set_xlim(-1.22, 1.22)
            ax.set_ylim(-1.22, 1.22)
            ax.set_zlim(-1.22, 1.22)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=18, azim=-62)
            ax.set_proj_type("ortho")
            ax.set_axis_off()

            k += 1

    fig.savefig(save_path, dpi=100, facecolor=background, edgecolor=background)
    plt.show()
    plt.close(fig)

    Image.open(save_path).convert("RGB").save(save_path)    
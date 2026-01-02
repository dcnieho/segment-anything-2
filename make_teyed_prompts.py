
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pathlib
import os

# -----------------------
# Path resolution helpers
# -----------------------

def _candidate_file_paths(video_path: pathlib.Path, gt_dir: pathlib.Path, suffix: str):
    """
    Generate candidate ground-truth file paths for a given (video, suffix).
    Tries your NVIDIAAR naming first, then the plain video-name based files
    used in your GW example (e.g., 'GW_1_1.mp4pupil_eli.txt').
    """
    if video_path.stem.startswith('GWNC'):
        video_path = video_path.with_stem(video_path.stem.replace('GWNC','GW'))
    stem_variant = f"NVIDIAAR_{video_path.stem.lstrip('0')}_1.mp4{suffix}"
    plain_variant = f"{video_path.name}{suffix}"
    return [
        gt_dir / stem_variant,
        gt_dir / plain_variant,
    ]

def _resolve_existing_file(video_path: pathlib.Path, gt_dir: pathlib.Path, suffix: str) -> pathlib.Path:
    for p in _candidate_file_paths(video_path, gt_dir, suffix):
        if p.exists():
            return p
    # Fall back to NVIDIAAR path to fail loudly with a predictable message
    return gt_dir / f"NVIDIAAR_{video_path.stem.lstrip('0')}_1.mp4{suffix}"

# -----------------------
# Frame picking
# -----------------------

def _pick_frame_index_from_valid(valid_pupil: pd.Series,
                                 valid_iris: pd.Series,
                                 frame_idx_override: int | None) -> int:
    """
    Frame selection requiring BOTH pupil and iris validity.

    - If frame_idx_override is provided, use it.
    - Else, if the first frame is valid for BOTH, use 0.
    - Else, find the first run of frames where BOTH are valid for > 10 frames,
      and pick index 4 in that run (or the first available).
    """
    if frame_idx_override is not None:
        return int(frame_idx_override)

    # Align lengths in case files differ slightly
    n = min(len(valid_pupil), len(valid_iris))
    valid = (valid_pupil.iloc[:n].reset_index(drop=True) &
             valid_iris.iloc[:n].reset_index(drop=True))

    if valid.iloc[0]:
        return 0

    # Find first run of valid values that is long enough (>10)
    a = (valid.diff(1) != 0).astype('int').cumsum()
    a[~valid] = -1
    b = a.groupby(a).size()
    candidates = np.where(np.logical_and(b > 10, b.index != -1))[0]
    if len(candidates) == 0:
        # Fallback: first valid anywhere
        idxs = np.where(valid.values)[0]
        if len(idxs) == 0:
            return 0
        return int(idxs[0])

    long_enough = candidates[0]
    fr_idxs = np.where(a == b.index[long_enough])[0]
    fr_idx = fr_idxs[4] if len(fr_idxs) > 4 else fr_idxs[0]
    return int(fr_idx)

# -----------------------
# Geometry helpers
# -----------------------

def _ellipse_horizontal_intersections_at_y(cx, cy, a, b, theta_deg, y0, eps=1e-9):
    """
    Intersect rotated ellipse with horizontal line y = y0. Returns [x_left, x_right].
    Ellipse: center (cx,cy), semi-axes (a,b), rotation theta_deg (degrees).
    """
    theta = np.deg2rad(theta_deg)
    A = a * np.sin(theta)
    B = b * np.cos(theta)
    r = np.hypot(A, B)
    delta = y0 - cy

    if r < eps:
        return [cx - a, cx + a]

    ratio = np.clip(delta / r, -1.0, 1.0)
    beta = np.arctan2(B, A)
    phi = np.arccos(ratio)
    t1 = beta + phi
    t2 = beta - phi

    cth = np.cos(theta)
    sth = np.sin(theta)
    x1 = cx + a * np.cos(t1) * cth - b * np.sin(t1) * sth
    x2 = cx + a * np.cos(t2) * cth - b * np.sin(t2) * sth

    xs = [float(x1), float(x2)]
    xs.sort()
    return xs

def _polygon_horizontal_intersections(poly_xy: np.ndarray, y0: float) -> list[float]:
    """
    All x where polygon intersects the horizontal line y=y0 (half-open rule).
    """
    xs = []
    n = len(poly_xy)
    if n < 2:
        return xs
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        if (y1 <= y0 < y2) or (y2 <= y0 < y1):
            if y2 != y1:
                t = (y0 - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                xs.append(float(x))
    xs.sort()
    return xs

def _point_in_polygon(x: float, y: float, poly_xy: np.ndarray) -> bool:
    inside = False
    n = len(poly_xy)
    if n < 3:
        return False
    for i in range(n):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[(i + 1) % n]
        intersects = ((yi > y) != (yj > y)) and \
                     (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersects:
            inside = not inside
    return inside

def _eyelid_polygon_from_row(row: pd.Series) -> np.ndarray:
    """
    Parse eyelid row -> (N,2) array. Assumes first two columns are FRAME, AVG INACCURACY.
    Remaining columns are LM X, LM Y pairs. Filters NaN and -1.
    """
    vals = row.values
    if len(vals) <= 2:
        return np.empty((0, 2), dtype=float)

    lm_vals = vals[2:]
    m = (len(lm_vals) // 2) * 2
    lm_vals = lm_vals[:m]
    xs = lm_vals[0::2].astype(float)
    ys = lm_vals[1::2].astype(float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs != -1) & (ys != -1)
    if np.any(mask):
        poly = np.stack([xs[mask], ys[mask]], axis=1)
    else:
        poly = np.empty((0, 2), dtype=float)
    return poly

# ---- Iris prompt placement along major axis ----

def _ellipse_support_radius_along_dir(a: float, b: float, theta_deg: float, d_unit: np.ndarray) -> float:
    """
    For an ellipse with semi-axes (a,b) rotated by theta_deg, returns the distance from the
    ellipse center to its boundary in direction d_unit (||d_unit||=1).
    r = 1 / sqrt( (dx'^2)/a^2 + (dy'^2)/b^2 ), where d' = R(-theta)*d_unit.
    """
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    dx =  c * d_unit[0] + s * d_unit[1]
    dy = -s * d_unit[0] + c * d_unit[1]
    denom = (dx*dx) / (a*a + 1e-12) + (dy*dy) / (b*b + 1e-12)
    denom = max(denom, 1e-18)
    return 1.0 / np.sqrt(denom)

def _major_axis_dir_and_len(a: float, b: float, theta_deg: float) -> tuple[np.ndarray, float]:
    """
    Returns (unit_direction, semi_len) for the ellipse's major axis.
    The ellipse local x-axis is orientation theta; if b > a, the major axis is rotated by +90°.
    """
    theta = np.deg2rad(theta_deg)
    u = np.array([np.cos(theta), np.sin(theta)])        # local x-axis
    v = np.array([-np.sin(theta), np.cos(theta)])       # local y-axis
    if a >= b:
        return u, a
    else:
        return v, b
    
    
def _min_distance_to_polygon_edges(pt: np.ndarray, poly_xy: np.ndarray) -> float:
    """
    Minimum Euclidean distance from point to any polygon edge (closed polyline).
    If polygon has <2 points, returns +inf.
    """
    if poly_xy is None or len(poly_xy) < 2:
        return float('inf')
    x0, y0 = float(pt[0]), float(pt[1])
    min_d2 = float('inf')
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        vx, vy = x2 - x1, y2 - y1
        wx, wy = x0 - x1, y0 - y1
        denom = vx*vx + vy*vy
        if denom <= 1e-18:
            # Degenerate segment: distance to vertex
            d2 = wx*wx + wy*wy
        else:
            t = max(0.0, min(1.0, (wx*vx + wy*vy) / denom))
            projx = x1 + t * vx
            projy = y1 + t * vy
            dx, dy = x0 - projx, y0 - projy
            d2 = dx*dx + dy*dy
        if d2 < min_d2:
            min_d2 = d2
    return float(np.sqrt(min_d2))


def _inside_with_margin(pt: np.ndarray, poly_xy: np.ndarray, margin_px: float) -> bool:
    """
    True iff the point is inside the polygon AND its distance to the polygon boundary
    is at least margin_px.
    """
    if not _point_in_polygon(float(pt[0]), float(pt[1]), poly_xy):
        return False
    d = _min_distance_to_polygon_edges(pt, poly_xy)
    return d >= float(margin_px)



def _ensure_inside_with_margin(pt: np.ndarray,
                               poly_xy: np.ndarray,
                               margin_px: float,
                               ellipse_center: tuple[float, float] | None = None,
                               ellipse_a: float | None = None,   # semi-axis a (WIDTH/2)
                               ellipse_b: float | None = None,   # semi-axis b (HEIGHT/2)
                               ellipse_angle_deg: float | None = None) -> np.ndarray:
    """
    Adjust `pt` so that:
      1) It lies INSIDE the eyelid polygon with ≥ `margin_px` clearance, and
      2) (if ellipse params provided) It is ≥ `margin_px` away from the ellipse edge.

    Strategy:
      - Move along the segment from the point toward the polygon centroid,
        preferring minimal movement and validating both clearances.
      - If no position achieves both clearances, choose the inside sample with
        maximal *combined* clearance, or fall back to the segment midpoint.

    Args:
        pt: (x, y) point to adjust.
        poly_xy: eyelid polygon (Nx2).
        margin_px: required clearance to polygon boundary and ellipse edge.
        ellipse_center, ellipse_a, ellipse_b, ellipse_angle_deg: optional ellipse parameters
            to enforce clearance from ellipse boundary (rotated).
            If omitted, only polygon margin is enforced.

    Returns:
        Adjusted point as np.ndarray([x, y], dtype=float).
    """
    pt = np.asarray(pt, dtype=float)

    # Helpers
    def _poly_clearance(p: np.ndarray) -> float:
        return _min_distance_to_polygon_edges(p, poly_xy)

    def _ellipse_clearance(p: np.ndarray) -> float:
        # If no ellipse constraint requested, treat as infinite clearance
        if ellipse_center is None or ellipse_a is None or ellipse_b is None or ellipse_angle_deg is None:
            return float('inf')
        cx, cy = float(ellipse_center[0]), float(ellipse_center[1])
        dv = np.array([p[0] - cx, p[1] - cy], dtype=float)
        L = float(np.linalg.norm(dv))
        if L < 1e-9:
            # At ellipse center: clearance equals the *minimum* support radius over all directions,
            # which is at least min(a, b)
            return float(min(ellipse_a, ellipse_b))
        d_unit = dv / L
        r = _ellipse_support_radius_along_dir(float(ellipse_a), float(ellipse_b),
                                              float(ellipse_angle_deg), d_unit)
        return float(max(0.0, r - L))  # interior radial clearance

    def _inside_with_both_margins(p: np.ndarray) -> bool:
        return _point_in_polygon(p[0], p[1], poly_xy) and \
               (_poly_clearance(p) >= margin_px) and \
               (_ellipse_clearance(p) >= margin_px)

    # If already meets both margins, keep it.
    if _inside_with_both_margins(pt):
        return pt

    # If polygon is not usable, fall back to enforcing ellipse margin only.
    if poly_xy is None or len(poly_xy) < 3:
        # Try small steps toward ellipse center if needed, else midpoint noop.
        return pt  # With no polygon, do not reposition aggressively.

    # Polygon centroid (simple mean; adequate for these eyelid shapes)
    cx = float(np.mean(poly_xy[:, 0]))
    cy = float(np.mean(poly_xy[:, 1]))
    centroid = np.array([cx, cy], dtype=float)
    v = pt - centroid

    # Parameterize candidate: p(alpha) = centroid + alpha * v, alpha in [0, 1]
    # alpha=1: original pt; alpha=0: centroid. Sample minimal movement first.
    alphas = np.linspace(1.0, 0.0, num=31)

    best_inside = None
    best_score = -float('inf')  # score = min(poly_clearance, ellipse_clearance)
    best_alpha_dist = float('inf')

    for alpha in alphas:
        p = centroid + alpha * v
        if not _point_in_polygon(p[0], p[1], poly_xy):
            continue
        poly_clr = _poly_clearance(p)
        ell_clr = _ellipse_clearance(p)
        meets_poly = (poly_clr >= margin_px)
        meets_ell  = (ell_clr  >= margin_px)

        if meets_poly and meets_ell:
            # Prefer minimal movement: alpha closest to 1.0
            return p

        # Track best inside sample by combined clearance, tie-breaking by closeness to original (alpha ~ 1)
        score = min(poly_clr, ell_clr)
        alpha_dist = abs(1.0 - alpha)
        if score > best_score or (score == best_score and alpha_dist < best_alpha_dist):
            best_score = score
            best_inside = p
            best_alpha_dist = alpha_dist

    # If none satisfies both margins:
    if best_inside is not None:
        # Fallback: return the inside sample with maximal combined clearance
        # (may be below the requested margin, but still inside).
        return best_inside

    # As a final fallback, return the midpoint along the path to the centroid (may violate margin).
    return centroid + 0.5 * v


def _ensure_inside_with_margin_on_segment(pt: np.ndarray,
                                          seg_start: np.ndarray,
                                          seg_end: np.ndarray,
                                          poly_xy: np.ndarray,
                                          margin_px: float) -> np.ndarray:
    """
    Adjust `pt` along the line segment [seg_start, seg_end] (pupil-edge ↔ iris-edge)
    so that:
      1) It lies INSIDE the eyelid polygon with at least `margin_px` clearance, and
      2) It stays at least `margin_px` away from BOTH segment endpoints
         (i.e., not on/too close to the pupil or iris ellipse edges).
    
    If the segment is too short to keep endpoint clearance (length < 2*margin_px),
    returns the segment midpoint. Otherwise, searches within the feasible
    sub-interval [t_min, t_max] = [margin_px/L, 1 - margin_px/L] for a point with
    polygon clearance ≥ margin_px. Among valid points, prefers the one closest to
    the segment midpoint. If none satisfy the clearance, returns the inside sample
    with maximal clearance; if no inside samples exist at all, returns the midpoint
    of the feasible interval (which still guarantees endpoint margin).
    """
    pt = np.asarray(pt, dtype=float)
    a = np.asarray(seg_start, dtype=float)
    b = np.asarray(seg_end, dtype=float)

    # Degenerate segment -> return midpoint
    ab = b - a
    L = float(np.linalg.norm(ab))
    if not np.isfinite(L) or L < 1e-9:
        return 0.5 * (a + b)

    # Enforce endpoint (ellipse-edge) margin by restricting t
    edge_margin = float(margin_px)
    t_min = edge_margin / L
    t_max = 1.0 - t_min

    # If the segment is too short to keep edge margin, return its midpoint
    if t_min >= t_max:
        return 0.5 * (a + b)

    # Helper: evaluate a candidate at parameter t
    def P(t: float) -> np.ndarray:
        return a + t * ab

    # Preferred parameter (midpoint of the feasible interval)
    t_mid = 0.5 * (t_min + t_max)

    # If original point is in the feasible interval and meets polygon margin, keep it
    # (project original pt onto segment, clamp to [t_min, t_max])
    t0 = float(np.dot(pt - a, ab) / (L * L))
    t0 = max(t_min, min(t_max, t0))
    p0 = P(t0)
    if _inside_with_margin(p0, poly_xy, margin_px):
        return p0

    # Sample across the feasible interval; prefer inside points with >= margin,
    # closest to the interval midpoint; else pick inside with maximal clearance.
    best_inside = None
    best_inside_clearance = -float('inf')
    best_inside_mid_dist = float('inf')

    # Coarse sampling first
    for t in np.linspace(t_min, t_max, num=41):
        p = P(t)
        if not _point_in_polygon(p[0], p[1], poly_xy):
            continue
        clr = _min_distance_to_polygon_edges(p, poly_xy)
        if clr >= margin_px:
            # Prefer nearest to midpoint of the feasible interval
            mid_dist = abs(t - t_mid)
            if mid_dist < best_inside_mid_dist:
                best_inside_mid_dist = mid_dist
                best_inside = p
                best_inside_clearance = clr
        else:
            # Track best-clearance inside point (in case none meet full margin)
            if clr > best_inside_clearance:
                best_inside_clearance = clr
                best_inside = p

    if best_inside is not None:
        return best_inside

    # If no inside samples exist in the feasible interval, return its midpoint.
    # This guarantees not being on/near the ellipse edges, even if polygon margin is unmet.
    return P(t_mid)


def compute_iris_prompt_points(
    pupil_cx: float, pupil_cy: float, ap: float, bp: float, pupil_ang_deg: float,
    iris_cx: float, iris_cy: float, ai: float, bi: float, iris_ang_deg: float,
):
    """
    Compute two iris prompt points along the iris major axis:
      - one on the side pointing TOWARD the pupil center,
      - one on the opposite side (AWAY from the pupil center).

    Each prompt is the midpoint between:
      * the iris outer edge on that side, and
      * the pupil edge along the same direction (support distance).

    Returns:
        pt_toward, pt_away,
        seg_toward=(pupil_edge_toward, iris_edge_toward),
        seg_away  =(pupil_edge_away,   iris_edge_away)
    where each point is np.ndarray(shape=(2,)), and each segment endpoint is np.ndarray(shape=(2,)).
    """
    # Iris major-axis unit direction and semi-length
    d_major, a_major = _major_axis_dir_and_len(ai, bi, iris_ang_deg)

    # Decide which direction points toward the pupil center
    vec_ip = np.array([pupil_cx - iris_cx, pupil_cy - iris_cy], dtype=float)
    s = 1.0 if np.dot(vec_ip, d_major) >= 0.0 else -1.0

    d_toward = s * d_major      # unit vector along major axis toward pupil
    d_away   = -s * d_major     # unit vector along major axis away from pupil

    # Outer iris edge points on each side
    iris_edge_toward = np.array([iris_cx, iris_cy], dtype=float) + a_major * d_toward
    iris_edge_away   = np.array([iris_cx, iris_cy], dtype=float) + a_major * d_away

    # Pupil edges along the same directions (support distances)
    r_pupil_toward = _ellipse_support_radius_along_dir(ap, bp, pupil_ang_deg, d_toward)
    r_pupil_away   = _ellipse_support_radius_along_dir(ap, bp, pupil_ang_deg, d_away)

    pupil_edge_toward = np.array([pupil_cx, pupil_cy], dtype=float) + r_pupil_toward * d_toward
    pupil_edge_away   = np.array([pupil_cx, pupil_cy], dtype=float) + r_pupil_away   * d_away

    # Midpoints (raw prompts)
    pt_toward = 0.5 * (iris_edge_toward + pupil_edge_toward)
    pt_away   = 0.5 * (iris_edge_away   + pupil_edge_away)

    # Return points and their segments (needed for margin adjustment along the segment)
    seg_toward = (pupil_edge_toward, iris_edge_toward)
    seg_away   = (pupil_edge_away,   iris_edge_away)

    return pt_toward, pt_away, seg_toward, seg_away


def adjust_iris_prompts_with_margin(
    pt_toward: np.ndarray, pt_away: np.ndarray,
    seg_toward: tuple[np.ndarray, np.ndarray],
    seg_away: tuple[np.ndarray, np.ndarray],
    eyelid_poly: np.ndarray,
    margin_px: float
):
    """
    Adjust both iris prompt points along their respective segments to satisfy:
      - inside the polygon with at least `margin_px` clearance, and
      - at least `margin_px` away from both segment endpoints (ellipse edges).
    """
    adj_toward = _ensure_inside_with_margin_on_segment(
        pt_toward, seg_toward[0], seg_toward[1], eyelid_poly, margin_px
    )
    adj_away = _ensure_inside_with_margin_on_segment(
        pt_away, seg_away[0], seg_away[1], eyelid_poly, margin_px
    )
    return adj_toward, adj_away


def _ellipse_horizontal_intersections_at_y(cx, cy, a, b, theta_deg, y0, eps=1e-9):
    """
    Intersect a rotated ellipse with the horizontal line y = y0. Returns [x_left, x_right].
    Ellipse: center (cx,cy), semi-axes (a,b), rotation theta_deg (degrees).
    """
    theta = np.deg2rad(theta_deg)
    A = a * np.sin(theta)
    B = b * np.cos(theta)
    r = np.hypot(A, B)
    delta = y0 - cy

    if r < eps:
        return [cx - a, cx + a]

    ratio = np.clip(delta / r, -1.0, 1.0)
    beta = np.arctan2(B, A)
    phi = np.arccos(ratio)
    t1 = beta + phi
    t2 = beta - phi

    cth = np.cos(theta)
    sth = np.sin(theta)
    x1 = cx + a * np.cos(t1) * cth - b * np.sin(t1) * sth
    x2 = cx + a * np.cos(t2) * cth - b * np.sin(t2) * sth

    xs = [float(x1), float(x2)]
    xs.sort()
    return xs

def _polygon_horizontal_intersections(poly_xy: np.ndarray, y0: float) -> list[float]:
    """
    All x where polygon intersects the horizontal line y=y0 (half-open rule).
    """
    xs = []
    n = len(poly_xy)
    if n < 2:
        return xs
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        if (y1 <= y0 < y2) or (y2 <= y0 < y1):
            if y2 != y1:
                t = (y0 - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                xs.append(float(x))
    xs.sort()
    return xs

def _point_in_polygon(x: float, y: float, poly_xy: np.ndarray) -> bool:
    """
    Ray casting point-in-polygon.
    """
    inside = False
    n = len(poly_xy)
    if n < 3:
        return False
    for i in range(n):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[(i + 1) % n]
        intersects = ((yi > y) != (yj > y)) and \
                     (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersects:
            inside = not inside
    return inside

# -----------------------
# Sclera point helpers
# -----------------------

def _sclera_point_on_scanline_side(
    y0: float,
    iris_cx: float, iris_cy: float, a: float, b: float, angle_deg: float,
    eyelid_poly: np.ndarray,
    side: str,
    min_clearance_px: float,
    min_allowed_px: float,
):
    """
    Compute a sclera point for ONE side ('left' or 'right') at scanline y=y0.
    Enforces clearance:
      - If gap >= 2*min_clearance_px: place inside the feasible interval so both distances >= min_clearance_px.
      - Else if gap >= 2*min_allowed_px: place MIDPOINT (distances = gap/2).
      - Else: skip (return None).

    Returns:
        (pt: np.ndarray[x, y],  gap: float, inside: bool, bounds: tuple) or (None, -inf, False, None)
        bounds are (iris_edge_x, lid_edge_x) for that side at y0 (used for nudging).
    """
    iris_xs = _ellipse_horizontal_intersections_at_y(iris_cx, iris_cy, a, b, angle_deg, y0)
    if len(iris_xs) < 2 or not np.all(np.isfinite(iris_xs)):
        return None, float("-inf"), False, None

    lid_xs = _polygon_horizontal_intersections(eyelid_poly, y0)
    if len(lid_xs) < 2:
        return None, float("-inf"), False, None

    lid_left_x, lid_right_x = float(np.min(lid_xs)), float(np.max(lid_xs))
    iris_left_x, iris_right_x = iris_xs[0], iris_xs[1]

    if side == 'left':
        gap = iris_left_x - lid_left_x
        if gap < 2 * min_allowed_px:
            return None, float(gap), False, None
        # choose clearance to enforce
        if gap >= 2 * min_clearance_px:
            c = float(min_clearance_px)
        else:
            c = gap / 2.0  # midpoint clearance
        # feasible interval for x
        x_lo = lid_left_x + c
        x_hi = iris_left_x - c
        x_mid = 0.5 * (x_lo + x_hi)  # also equals (lid_left_x + iris_left_x)/2
        pt = np.array([x_mid, y0], dtype=float)
        inside = _point_in_polygon(pt[0], pt[1], eyelid_poly)
        bounds = (iris_left_x, lid_left_x)
        return pt, float(gap), bool(inside), bounds

    elif side == 'right':
        gap = lid_right_x - iris_right_x
        if gap < 2 * min_allowed_px:
            return None, float(gap), False, None
        if gap >= 2 * min_clearance_px:
            c = float(min_clearance_px)
        else:
            c = gap / 2.0
        x_lo = iris_right_x + c
        x_hi = lid_right_x - c
        x_mid = 0.5 * (x_lo + x_hi)  # also equals (iris_right_x + lid_right_x)/2
        pt = np.array([x_mid, y0], dtype=float)
        inside = _point_in_polygon(pt[0], pt[1], eyelid_poly)
        bounds = (iris_right_x, lid_right_x)
        return pt, float(gap), bool(inside), bounds

    else:
        raise ValueError("side must be 'left' or 'right'")

def _scan_for_sclera_point(
    start_y: float,
    iris_cx: float, iris_cy: float, a: float, b: float, angle_deg: float,
    eyelid_poly: np.ndarray,
    side: str,
    min_clearance_px: float,
    min_allowed_px: float,
    max_scan_fraction: float,
    max_scan_px: int,
):
    """
    Try y=start_y, then scan up/down for a sclera point on given side.
    Preference:
      1) Nearest scanline with point inside polygon and gap >= 2*min_allowed_px.
      2) Else scanline with largest gap (quality), then try nudging inside within feasible interval.
      3) If still not possible, return None (skip).
    """
    # 1) first attempt at start_y
    pt, gap, inside, bounds = _sclera_point_on_scanline_side(
        start_y, iris_cx, iris_cy, a, b, angle_deg, eyelid_poly, side, min_clearance_px, min_allowed_px
    )
    if pt is not None and inside:
        return pt

    # 2) scan up/down
    max_scan = max(int(max_scan_fraction * (2 * b)), int(max_scan_px))  # scale by iris height (2b)
    candidates = []  # (abs_offset, pt, gap, inside, y, bounds)
    for k in range(1, max_scan + 1):
        for sign in (1, -1):
            y = start_y + sign * k
            c_pt, c_gap, c_inside, c_bounds = _sclera_point_on_scanline_side(
                y, iris_cx, iris_cy, a, b, angle_deg, eyelid_poly, side, min_clearance_px, min_allowed_px
            )
            if c_pt is not None:
                candidates.append((abs(sign * k), c_pt, c_gap, c_inside, y, c_bounds))

    # Prefer nearest inside
    for entry in sorted(candidates, key=lambda t: t[0]):
        _, c_pt, c_gap, c_inside, _, _ = entry
        if c_inside and c_gap >= 2 * min_allowed_px:
            return c_pt

    # Otherwise pick best gap and try nudging inside within feasible interval
    if len(candidates) > 0:
        # sort by gap descending (quality)
        candidates_sorted = sorted(candidates, key=lambda t: t[2], reverse=True)
        _, c_pt, c_gap, c_inside, y, c_bounds = candidates_sorted[0]
        if c_inside:
            return c_pt

        # Nudge horizontally within feasible interval to get inside while keeping clearances
        iris_edge_x, lid_edge_x = c_bounds
        gap = c_gap
        # choose clearance to enforce (min_clearance if possible else midpoint clearance)
        if gap >= 2 * min_clearance_px:
            c = float(min_clearance_px)
        else:
            c = gap / 2.0

        if side == 'left':
            x_lo = lid_edge_x + c
            x_hi = iris_edge_x - c
        else:  # right
            x_lo = iris_edge_x + c
            x_hi = lid_edge_x - c

        if x_hi > x_lo:
            # sample several positions across the feasible interval and pick the one closest to midpoint that is inside
            samples = np.linspace(x_lo, x_hi, num=9)
            x_mid = 0.5 * (x_lo + x_hi)
            best = None
            best_dist = float('inf')
            for x in samples:
                if _point_in_polygon(x, y, eyelid_poly):
                    d = abs(x - x_mid)
                    if d < best_dist:
                        best = x
                        best_dist = d
            if best is not None:
                return np.array([best, y], dtype=float)

    # 3) Not found: skip
    return None

# -----------------------
# Public: compute sclera points (left, right)
# -----------------------

def _compute_sclera_points(
    iris_cx: float, iris_cy: float,
    iris_w: float, iris_h: float,
    iris_angle_deg: float,
    eyelid_poly: np.ndarray,
    min_clearance_px: float = 30.0,
    min_allowed_px: float = 5.0,
    max_scan_fraction: float = 0.4,
    max_scan_px: int = 40,
):
    """
    Compute two sclera points (left/right) with clearance constraints and vertical scan fallback.

    Clearance policy per side:
      - If (eyelid-to-iris gap) >= 2*min_clearance_px: place in feasible interval so both distances >= min_clearance_px.
      - Else if gap >= 2*min_allowed_px: place MIDPOINT (distance = gap/2).
      - Else: skip that side (return [nan, nan]).

    Points are required to be **inside the eyelid polygon**; scanning up/down is used if needed.
    If no suitable scanline yields an inside point, the side is skipped.

    Returns:
        (scl_left: np.ndarray[2], scl_right: np.ndarray[2])
        If skipped, the corresponding array is [nan, nan].
    """
    a = iris_w / 2.0
    b = iris_h / 2.0

    # Left side
    left_pt = _scan_for_sclera_point(
        iris_cy, iris_cx, iris_cy, a, b, iris_angle_deg, eyelid_poly,
        side='left',
        min_clearance_px=min_clearance_px,
        min_allowed_px=min_allowed_px,
        max_scan_fraction=max_scan_fraction,
        max_scan_px=max_scan_px,
    )

    # Right side
    right_pt = _scan_for_sclera_point(
        iris_cy, iris_cx, iris_cy, a, b, iris_angle_deg, eyelid_poly,
        side='right',
        min_clearance_px=min_clearance_px,
        min_allowed_px=min_allowed_px,
        max_scan_fraction=max_scan_fraction,
        max_scan_px=max_scan_px,
    )

    if left_pt is None:
        left_pt = np.array([np.nan, np.nan], dtype=float)
    if right_pt is None:
        right_pt = np.array([np.nan, np.nan], dtype=float)

    return left_pt, right_pt


# -----------------------
# Visualization
# -----------------------

def _draw_marker(img, pt_xy, color_bgr, marker_type=cv2.MARKER_CROSS, size=6, thickness=2):
    x = int(round(pt_xy[0]))
    y = int(round(pt_xy[1]))
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.drawMarker(img, (x, y), color_bgr, marker_type, size, thickness)

def _draw_polygon(img, poly_xy: np.ndarray, color_bgr=(0, 255, 255), thickness=2):
    """
    Draw eyelid polygon as a closed polyline (yellow by default).
    """
    if poly_xy is None or len(poly_xy) < 2:
        return
    pts = np.round(poly_xy).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=thickness)

def _draw_ellipse(img, cx: float, cy: float, a: float, b: float, angle_deg: float,
                  color_bgr=(0, 255, 0), thickness=2):
    """
    Draw an ellipse outline using OpenCV's cv2.ellipse.
    a, b are semi-axes (not full width/height).
    """
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return
    center = (int(round(cx)), int(round(cy)))
    axes = (max(1, int(round(a))), max(1, int(round(b))))
    angle = float(angle_deg)
    cv2.ellipse(img, center, axes, angle, 0.0, 360.0, color_bgr, thickness)

def _save_frame_with_points(video_path: pathlib.Path,
                            frame_zero_based: int,
                            pupil_pt: np.ndarray,
                            iris_prompt_pt: np.ndarray,
                            sclera_left: np.ndarray,
                            sclera_right: np.ndarray,
                            out_png: pathlib.Path,
                            eyelid_poly: np.ndarray | None = None,
                            pupil_ellipse: tuple[float,float,float,float,float] | None = None,
                            iris_ellipse: tuple[float,float,float,float,float] | None = None):
    """
    Save an annotated frame:
      - eyelid polygon
      - pupil & iris ellipses
      - markers: pupil (green), iris prompt (red), sclera L (blue), sclera R (magenta)
    pupil_ellipse / iris_ellipse: (cx, cy, a, b, angle_deg)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_zero_based)
        ok, img = cap.read()
        if not ok or img is None:
            raise RuntimeError(f"Failed to read frame {frame_zero_based} from {video_path}")

        # --- Overlays ---
        # 1) Eyelid polygon
        _draw_polygon(img, eyelid_poly, color_bgr=(0, 255, 255), thickness=2)  # yellow

        # 2) Ellipses
        if pupil_ellipse is not None:
            pcx, pcy, pa, pb, pang = pupil_ellipse
            _draw_ellipse(img, pcx, pcy, pa, pb, pang, color_bgr=(0, 200, 0), thickness=2)  # greenish
        if iris_ellipse is not None:
            icx, icy, ia, ib, iang = iris_ellipse
            _draw_ellipse(img, icx, icy, ia, ib, iang, color_bgr=(0, 0, 200), thickness=2)  # reddish

        # 3) Markers
        clr_pupil = (0, 255, 0)     # green
        clr_irisP = (0, 0, 255)     # red (iris prompt)
        clr_scl_l = (255, 0, 0)     # blue
        clr_scl_r = (255, 0, 255)   # magenta

        def _finite(pt): return np.all(np.isfinite(pt))

        if _finite(pupil_pt):      _draw_marker(img, pupil_pt,    clr_pupil)
        if _finite(iris_prompt_pt):_draw_marker(img, iris_prompt_pt, clr_irisP)
        if _finite(sclera_left):   _draw_marker(img, sclera_left, clr_scl_l)
        if _finite(sclera_right):  _draw_marker(img, sclera_right, clr_scl_r)

        # Save with PIL (convert BGR->RGB for correct colors)
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(out_png)
    finally:
        cap.release()


def _as_int_xy(pt: np.ndarray) -> tuple[int, int]:
    """Round to nearest integer pixel coordinates."""
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def write_prompt_to_file(prompt_obj: dict, out_path: str | pathlib.Path) -> None:
    """
    Write prompts to a tab-separated file with the required negatives and sorted order.

    Sorting:
      - by class: pupil, iris, sclera
      - within each class: positives (label=1) first, then negatives (label=0)

    Negative rules:
      - Positive pupil → add iris 0 + sclera 0 at pupil coords
      - Positive iris  → add sclera 0 at iris coords
      - Positive sclera(s) → add iris 0 at each sclera coord

    Args:
        prompt_obj: dict returned from retrieve_prompt_from_subject(...)
        out_path: output file path
    """
    out_path = pathlib.Path(out_path)
    prom = prompt_obj['prompt']

    pupil_pts = prom['pupil']['points']   # shape (1, 2)
    iris_pts  = prom['iris']['points']    # shape (2, 2)
    sclera_pts = prom['sclera']['points'] # shape (N, 2), may contain NaNs

    lines: list[tuple[str, int, int, int]] = []

    # ---- POSITIVES ----
    # pupil
    if pupil_pts is not None and len(pupil_pts) > 0:
        px, py = _as_int_xy(pupil_pts[0])
        lines.append(('pupil', px, py, 1))

    # iris (prompt)
    if iris_pts is not None and len(iris_pts) > 0:
        ix, iy = _as_int_xy(iris_pts[0])
        lines.append(('iris', ix, iy, 1))
        ix, iy = _as_int_xy(iris_pts[1])
        lines.append(('sclera', ix, iy, 0))

    # sclera (left/right; skip NaNs)
    sclera_int_pts: list[tuple[int, int]] = []
    if sclera_pts is not None and len(sclera_pts) > 0:
        for p in sclera_pts:
            if np.all(np.isfinite(p)):
                sx, sy = _as_int_xy(p)
                sclera_int_pts.append((sx, sy))
                lines.append(('sclera', sx, sy, 1))

    # ---- NEGATIVES per rule ----
    # Positive pupil → iris 0 + sclera 0 at pupil coords
    if pupil_pts is not None and len(pupil_pts) > 0:
        px, py = _as_int_xy(pupil_pts[0])
        lines.append(('iris',   px, py, 0))
        lines.append(('sclera', px, py, 0))

    # Positive iris → sclera 0 at iris coords
    if iris_pts is not None and len(iris_pts) > 0:
        ix, iy = _as_int_xy(iris_pts[0])
        lines.append(('sclera', ix, iy, 0))

    # Positive sclera(s) → iris 0 at each sclera coord
    for (sx, sy) in sclera_int_pts:
        lines.append(('iris', sx, sy, 0))

    # ---- Deduplicate (keep first occurrence to preserve relative order) ----
    first_index: dict[tuple[str,int,int,int], int] = {}
    for idx, rec in enumerate(lines):
        if rec not in first_index:
            first_index[rec] = idx

    # Prepare sortable records with original index
    records_with_idx = [(rec, idx) for rec, idx in first_index.items()]

    # Sorting: class order + positives first, then by original index to keep stable intent
    class_rank = {'pupil': 0, 'iris': 1, 'sclera': 2}
    def sort_key(item):
        (cls, x, y, lab), idx = item
        return (class_rank.get(cls, 999), 0 if lab == 1 else 1, idx)

    sorted_records = sorted(records_with_idx, key=sort_key)

    # ---- Write to file ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for (cls, x, y, lab), _ in sorted_records:
            f.write(f"{cls}\t{x}\t{y}\t{lab}\n")


# -----------------------
# Main API
# -----------------------

def retrieve_prompt_from_subject(video_path,
                                 gt_dir,
                                 frame_idx_override: dict[str,int],
                                 save_overlay: bool = False,
                                 overlay_out_dir: str | pathlib.Path | None = None,
                                 inside_margin_px: float = 5.0):
    """
    Extended version:
      - Chooses frame based on BOTH pupil and iris validity (or uses override).
      - Returns pupil center, *iris prompt* point (not on pupil), and two sclera points (left, right).
      - Optionally saves a PNG overlay with eyelid polygon and ellipses.

    Args:
        video_path: Path or str to the video (.mp4).
        gt_dir: Path or str to the ground-truth directory.
        frame_idx_override: dict mapping video filename -> override GT row index.
        save_overlay: If True, save an annotated PNG of the selected frame.
        overlay_out_dir: Optional directory to save the PNG (default = video dir).

    Returns:
        dict with 'prompt' (pupil/iris/sclera points) and 'frame' (0-based video frame index).
    """
    video_path = pathlib.Path(video_path)
    gt_dir = pathlib.Path(gt_dir)

    # Read GT tables
    pupil_file = _resolve_existing_file(video_path, gt_dir, 'pupil_eli.txt')
    iris_file  = _resolve_existing_file(video_path, gt_dir, 'iris_eli.txt')
    lid_file   = _resolve_existing_file(video_path, gt_dir, 'lid_lm_2D.txt')

    try:
        pupil_gt = pd.read_csv(pupil_file, sep=';')
    except:
        print(f'error reading {pupil_file}, skipping')
        return
    try:
        iris_gt  = pd.read_csv(iris_file,  sep=';')
    except:
        print(f'error reading {iris_file}, skipping')
        return
    try:
        lid_gt   = pd.read_csv(lid_file,   sep=';')
    except:
        print(f'error reading {lid_file}, skipping')
        return

    # Validity (CENTER X != -1) for both
    valid_pupil = (pupil_gt['CENTER X'] != -1)
    valid_iris  = (iris_gt['CENTER X']  != -1)

    # Pick frame index using combined validity (or override)
    override_idx = frame_idx_override.get(video_path.name, None)
    fr_idx = _pick_frame_index_from_valid(valid_pupil, valid_iris, override_idx)

    # Pupil center + ellipse
    pupil_cx = float(pupil_gt['CENTER X'].iloc[fr_idx])
    pupil_cy = float(pupil_gt['CENTER Y'].iloc[fr_idx])
    pupil_w  = float(pupil_gt['WIDTH'].iloc[fr_idx])
    pupil_h  = float(pupil_gt['HEIGHT'].iloc[fr_idx])
    pupil_ang= float(pupil_gt['ANGLE'].iloc[fr_idx])
    ap = pupil_w / 2.0
    bp = pupil_h / 2.0

    # Iris center + ellipse
    iris_cx  = float(iris_gt['CENTER X'].iloc[fr_idx])
    iris_cy  = float(iris_gt['CENTER Y'].iloc[fr_idx])
    iris_w   = float(iris_gt['WIDTH'].iloc[fr_idx])    # full width
    iris_h   = float(iris_gt['HEIGHT'].iloc[fr_idx])   # full height
    iris_ang = float(iris_gt['ANGLE'].iloc[fr_idx])
    ai = iris_w / 2.0
    bi = iris_h / 2.0

    # Eyelid polygon (same row index)
    eyelid_poly = _eyelid_polygon_from_row(lid_gt.iloc[fr_idx])

    # Robust sclera points (left, right) with vertical scan strategy
    scl_left, scl_right = _compute_sclera_points(iris_cx, iris_cy, iris_w, iris_h, iris_ang, eyelid_poly)

    # Iris prompt point (midpoint along major axis between pupil edge and iris outer edge)
    iris_prompt_toward, iris_prompt_away, seg_toward, seg_away = compute_iris_prompt_points(
        pupil_cx, pupil_cy, ap, bp, pupil_ang,
        iris_cx, iris_cy, ai, bi, iris_ang,
    )

    # Ensure pupil point is inside with margin (including ellipse margin)
    pupil_pt = np.array([pupil_cx, pupil_cy], dtype=float)
    pupil_prompt_pt = _ensure_inside_with_margin(
        pupil_pt, eyelid_poly, inside_margin_px,
        ellipse_center=(pupil_cx, pupil_cy),
        ellipse_a=ap, ellipse_b=bp,
        ellipse_angle_deg=pupil_ang
    )

    # Adjust iris prompts along their segments to satisfy margin constraints
    iris_prompt_toward, iris_prompt_away = adjust_iris_prompts_with_margin(
        iris_prompt_toward, iris_prompt_away,
        seg_toward, seg_away,
        eyelid_poly, inside_margin_px
    )

    # If you want a deterministic left/right order by screen x:
    iris_prompts_sorted = sorted([iris_prompt_toward, iris_prompt_away], key=lambda p: p[0])
    iris_prompt_left  = iris_prompts_sorted[0]
    iris_prompt_right = iris_prompts_sorted[1]

    # Convert GT frame (1-based) to video frame index (0-based)
    frame_zero_based = int(pupil_gt['FRAME'].iloc[fr_idx]) - 1

    # Optional overlay
    if save_overlay:
        out_dir = pathlib.Path(overlay_out_dir) if overlay_out_dir is not None else video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{video_path.name}_frame{frame_zero_based:04d}.png"
        _save_frame_with_points(
            video_path,
            frame_zero_based,
            pupil_prompt_pt,
            iris_prompt_left,
            scl_left,
            scl_right,
            out_png,
            eyelid_poly=eyelid_poly,
            pupil_ellipse=(pupil_cx, pupil_cy, ap, bp, pupil_ang),
            iris_ellipse=(iris_cx,  iris_cy,  ai, bi, iris_ang)
        )

    prompt = {
        'prompt': {
            'pupil': {
                'points': pupil_prompt_pt.reshape(1, 2).astype(float),
                'labels': np.array([1], dtype=int),
                'box': None
            },
            'iris': {
                # Iris prompt point (not on the pupil), placed per your rule
                'points': np.vstack((iris_prompt_left.reshape(1, 2).astype(float), iris_prompt_right.reshape(1, 2).astype(float))),
                'labels': np.array([1, 0], dtype=int),
                'box': None
            },
            'sclera': {
                # order: left, right
                'points': np.stack([scl_left, scl_right], axis=0),
                'labels': np.array([1, 1], dtype=int),
                'box': None
            }
        },
        'frame': frame_zero_based
    }

    # Choose an output filename; e.g., one per (video, frame)
    out_file = overlay_out_dir / f"{pathlib.Path(video_path).stem}.points.txt"

    write_prompt_to_file(prompt, out_file)




if __name__ == '__main__':
    vid_dir = 'VIDEOS'
    gt_dir = 'ANNOTATIONS'
    root_dir = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\datasets real\TEyeD")
    prompt_out_dir = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\datasets real\TEyeD\prompts")

    datasets = [fp for f in os.scandir(root_dir) if (fp:=pathlib.Path(f.path)).is_dir() and all((fp/s).is_dir() for s in [vid_dir,gt_dir])]

    for d in datasets:
        video_files = list((d/vid_dir).glob("*.mp4"))
        for v in video_files:
            retrieve_prompt_from_subject(v, d/gt_dir, {}, True, prompt_out_dir/d.name)


import os

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pathlib

# -----------------------
# Path resolution helpers
# -----------------------

def _candidate_file_paths(video_path: pathlib.Path, gt_dir: pathlib.Path, suffix: str):
    """
    Generate candidate ground-truth file paths for a given (video, suffix).
    Tries NVIDIAAR naming first, then the plain video-name style (e.g., 'GW_1_1.mp4pupil_eli.txt').
    """
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

    - If frame_idx_override is provided, use it (GT row index).
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

    if n == 0:
        return 0

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


def _line_polygon_intersections(origin: np.ndarray,
                                direction: np.ndarray,
                                poly_xy: np.ndarray) -> list[float]:
    """
    Intersect an *infinite* line L(t) = origin + t * direction with a polygon.
    Returns the list of all parameter values t at which the line crosses each polygon edge.
    The list may have more than two entries if the polygon is non-convex.
    """
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    dn = np.linalg.norm(d)
    if dn < 1e-12 or poly_xy is None or len(poly_xy) < 2:
        return []
    d = d / dn

    def cross2(a, b): return a[0]*b[1] - a[1]*b[0]

    ts = []
    n = len(poly_xy)
    for i in range(n):
        p1 = poly_xy[i].astype(float)
        p2 = poly_xy[(i + 1) % n].astype(float)
        v = p2 - p1
        denom = cross2(d, v)
        if abs(denom) < 1e-12:
            continue  # parallel
        # Solve for t (line parameter) and u (segment parameter)
        t = cross2((p1 - o), v) / denom
        u = cross2((p1 - o), d) / denom
        if 0.0 <= u <= 1.0:
            ts.append(float(t))
    ts.sort()
    return ts

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
    x2 = cx + a * np.cos(t2) * cth * 1.0 - b * np.sin(t2) * sth

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

def _polygon_vertical_intersections(poly_xy: np.ndarray, x0: float) -> list[float]:
    """
    All y where polygon intersects the vertical line x=x0 (half-open rule).
    """
    ys = []
    n = len(poly_xy)
    if n < 2:
        return ys
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        if (x1 <= x0 < x2) or (x2 <= x0 < x1):
            if x2 != x1:
                t = (x0 - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                ys.append(float(y))
    ys.sort()
    return ys

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

def _min_distance_to_polygon_edges(pt: np.ndarray, poly_xy: np.ndarray) -> float:
    """
    Minimum Euclidean distance from point to any polygon edge.
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

# -----------------------
# Eyelid polygon parsing (with tail flip)
# -----------------------

def _eyelid_polygon_from_row(row: pd.Series) -> np.ndarray:
    """
    Parse eyelid row -> (N,2) array.
    Assumes first two columns are FRAME, AVG INACCURACY; remaining are LM X, LM Y pairs.
    Landmarks 18..end are in reverse order in the file; fix by reversing the tail (idx 17..end).
    Filters NaN and -1 after reordering.
    """
    vals = row.values
    if len(vals) <= 2:
        return np.empty((0, 2), dtype=float)

    lm_vals = vals[2:]
    m = (len(lm_vals) // 2) * 2
    lm_vals = lm_vals[:m]
    xs = lm_vals[0::2].astype(float)
    ys = lm_vals[1::2].astype(float)
    pts = np.stack([xs, ys], axis=1)  # (N,2)

    if len(pts) >= 18:
        head = pts[:17]
        tail = pts[17:][::-1]
        pts = np.vstack([head, tail])

    mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & (pts[:, 0] != -1) & (pts[:, 1] != -1)
    if np.any(mask):
        pts = pts[mask]
    else:
        pts = np.empty((0, 2), dtype=float)

    return pts

# -----------------------
# Iris prompt placement (vertical eyelid midpoint)
# -----------------------

def _ellipse_support_radius_along_dir(a: float, b: float, theta_deg: float, d_unit: np.ndarray) -> float:
    """
    Distance from ellipse center to boundary in direction d_unit (||d_unit||=1).
    Uses support function for rotated ellipse.
    """
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    dx =  c * d_unit[0] + s * d_unit[1]
    dy = -s * d_unit[0] + c * d_unit[1]
    denom = (dx*dx) / (a*a + 1e-12) + (dy*dy) / (b*b + 1e-12)
    denom = max(denom, 1e-18)
    return 1.0 / np.sqrt(denom)

def _major_axis_dir_and_len(a: float, b: float, theta_deg: float):
    """
    Returns (unit_direction, semi_len) for the ellipse's major axis.
    """
    theta = np.deg2rad(theta_deg)
    u = np.array([np.cos(theta), np.sin(theta)])        # local x-axis
    v = np.array([-np.sin(theta), np.cos(theta)])       # local y-axis
    if a >= b:
        return u, a
    else:
        return v, b

def _find_y_mid_of_eyelid(eyelid_poly: np.ndarray,
                          x_ref: float,
                          search_dx: int = 30) -> float | None:
    """
    Find vertical midpoint y_mid between the two eyelid edges at some x.
    Start at x_ref; if no vertical intersections, search left/right up to search_dx pixels.
    Returns y_mid or None if not found.
    """
    ys = _polygon_vertical_intersections(eyelid_poly, x_ref)
    if len(ys) >= 2:
        return 0.5 * (ys[0] + ys[-1])

    # search horizontally
    for k in range(1, search_dx + 1):
        for sign in (+1, -1):
            xs = x_ref + sign * k
            ys = _polygon_vertical_intersections(eyelid_poly, xs)
            if len(ys) >= 2:
                return 0.5 * (ys[0] + ys[-1])
    return None


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
                               ellipse_a: float | None = None,
                               ellipse_b: float | None = None,
                               ellipse_angle_deg: float | None = None) -> np.ndarray:
    """
    Adjust `pt` so that:
      1) It lies INSIDE the eyelid polygon with ≥ `margin_px` clearance, and
      2) (if ellipse params provided) It is ≥ `margin_px` away from the ellipse edge.
    Minimal movement strategy toward polygon centroid; returns best inside candidate if margin is impossible.
    """
    pt = np.asarray(pt, dtype=float)

    def _ellipse_clearance(p: np.ndarray) -> float:
        if ellipse_center is None or ellipse_a is None or ellipse_b is None or ellipse_angle_deg is None:
            return float('inf')
        cx, cy = float(ellipse_center[0]), float(ellipse_center[1])
        dv = np.array([p[0] - cx, p[1] - cy], dtype=float)
        L = float(np.linalg.norm(dv))
        if L < 1e-9:
            return float(min(ellipse_a, ellipse_b))
        d_unit = dv / L
        r = _ellipse_support_radius_along_dir(float(ellipse_a), float(ellipse_b),
                                              float(ellipse_angle_deg), d_unit)
        return float(max(0.0, r - L))

    def _inside_both(p: np.ndarray) -> bool:
        return _point_in_polygon(p[0], p[1], poly_xy) and \
               (_min_distance_to_polygon_edges(p, poly_xy) >= margin_px) and \
               (_ellipse_clearance(p) >= margin_px)

    if _inside_both(pt):
        return pt

    if poly_xy is None or len(poly_xy) < 3:
        return pt

    cx = float(np.mean(poly_xy[:, 0]))
    cy = float(np.mean(poly_xy[:, 1]))
    centroid = np.array([cx, cy], dtype=float)
    v = pt - centroid

    alphas = np.linspace(1.0, 0.0, num=31)
    best_inside = None
    best_score = -float('inf')
    best_alpha_dist = float('inf')

    for alpha in alphas:
        p = centroid + alpha * v
        if not _point_in_polygon(p[0], p[1], poly_xy):
            continue
        poly_clr = _min_distance_to_polygon_edges(p, poly_xy)
        cx_ell = _ellipse_clearance(p)
        meets_poly = (poly_clr >= margin_px)
        meets_ell  = (cx_ell   >= margin_px)

        if meets_poly and meets_ell:
            return p

        score = min(poly_clr, cx_ell)
        alpha_dist = abs(1.0 - alpha)
        if score > best_score or (score == best_score and alpha_dist < best_alpha_dist):
            best_score = score
            best_inside = p
            best_alpha_dist = alpha_dist

    if best_inside is not None:
        return best_inside

    return centroid + 0.5 * v

def _ensure_inside_with_margin_on_segment(pt: np.ndarray,
                                          seg_start: np.ndarray,
                                          seg_end: np.ndarray,
                                          poly_xy: np.ndarray,
                                          margin_px: float) -> np.ndarray:
    """
    Adjust `pt` along [seg_start, seg_end] so it is:
      - inside polygon with ≥ margin_px clearance, and
      - ≥ margin_px away from BOTH endpoints (ellipse edges).
    If segment length < 2*margin_px, return midpoint. Otherwise sample inside, prefer closest to feasible midpoint.
    """
    pt = np.asarray(pt, dtype=float)
    a = np.asarray(seg_start, dtype=float)
    b = np.asarray(seg_end, dtype=float)

    ab = b - a
    L = float(np.linalg.norm(ab))
    if not np.isfinite(L) or L < 1e-9:
        return 0.5 * (a + b)

    t_min = float(margin_px) / L
    t_max = 1.0 - t_min
    if t_min >= t_max:
        return 0.5 * (a + b)

    def P(t: float) -> np.ndarray:
        return a + t * ab

    t_mid = 0.5 * (t_min + t_max)

    # Project original to segment and clamp to feasible interval
    t0 = float(np.dot(pt - a, ab) / (L * L))
    t0 = max(t_min, min(t_max, t0))
    p0 = P(t0)
    if _inside_with_margin(p0, poly_xy, margin_px):
        return p0

    best_inside = None
    best_inside_clearance = -float('inf')
    best_inside_mid_dist = float('inf')

    for t in np.linspace(t_min, t_max, num=41):
        p = P(t)
        if not _point_in_polygon(p[0], p[1], poly_xy):
            continue
        clr = _min_distance_to_polygon_edges(p, poly_xy)
        if clr >= margin_px:
            mid_dist = abs(t - t_mid)
            if mid_dist < best_inside_mid_dist:
                best_inside_mid_dist = mid_dist
                best_inside = p
                best_inside_clearance = clr
        else:
            if clr > best_inside_clearance:
                best_inside_clearance = clr
                best_inside = p

    if best_inside is not None:
        return best_inside

    return P(t_mid)

# -----------------------
# Iris prompts at eyelid vertical midpoint (left/right)
# -----------------------

def compute_iris_prompt_points_mid_eyelid(
    pupil_cx: float, pupil_cy: float, ap: float, bp: float, pupil_ang_deg: float,
    iris_cx: float, iris_cy: float, ai: float, bi: float, iris_ang_deg: float,
    eyelid_poly: np.ndarray,
    margin_px: float = 5.0,
    x_for_mid: float | None = None,
    scan_y_updown: int = 40
):
    """
    Compute TWO iris prompt points at the vertical midpoint of the eyelid opening.

    Steps:
      1) Find y_mid = midpoint between upper & lower eyelid along a vertical line x = x_for_mid
         (default: x_for_mid = iris_cx; searches horizontally if needed).
      2) On the horizontal scanline y = y_mid, intersect both the pupil and the iris ellipses:
         - pupil: x_pL, x_pR
         - iris:  x_iL, x_iR
      3) Left iris prompt = midpoint between x_pL and x_iL at y=y_mid.
         Right iris prompt = midpoint between x_pR and x_iR at y=y_mid.
      4) Adjust each point ALONG ITS HORIZONTAL SEGMENT between the two edges to ensure:
         - inside eyelid polygon with ≥ margin_px clearance,
         - ≥ margin_px away from both ellipse edges (segment endpoints).
      5) If y_mid doesn’t intersect an ellipse, scan ±1..scan_y_updown px vertically to find a usable y.

    Returns:
      (pt_left, pt_right), and also the segments: (seg_left, seg_right), where each seg is (start=endpoints)
      with start = pupil edge point, end = iris edge point, both on y=y_mid.
    """
    # 1) vertical midpoint y of eyelid opening
    x_ref = iris_cx if x_for_mid is None else float(x_for_mid)
    y_mid = _find_y_mid_of_eyelid(eyelid_poly, x_ref, search_dx=30)
    if y_mid is None:
        # fallback to iris center y
        y_mid = float(iris_cy)

    def _edges_at_y(y0: float):
        ps = _ellipse_horizontal_intersections_at_y(pupil_cx, pupil_cy, ap, bp, pupil_ang_deg, y0)
        is_ = _ellipse_horizontal_intersections_at_y(iris_cx, iris_cy, ai, bi, iris_ang_deg, y0)
        if len(ps) >= 2 and len(is_) >= 2:
            pL, pR = ps[0], ps[1]
            iL, iR = is_[0], is_[1]
            return (pL, pR, iL, iR)
        return None

    edges = _edges_at_y(y_mid)

    # If no valid intersections at y_mid, scan up/down a bit
    if edges is None:
        found = None
        for k in range(1, scan_y_updown + 1):
            for sign in (+1, -1):
                y = y_mid + sign * k
                e = _edges_at_y(y)
                if e is not None:
                    found = (y, e)
                    break
            if found is not None:
                break
        if found is None:
            # give up: place NaNs
            na = np.array([np.nan, np.nan], float)
            return na, na, (na, na), (na, na)
        y_mid, edges = found

    pL, pR, iL, iR = edges

    # 3) midpoints (raw prompts)
    pt_left_raw  = np.array([(pL + iL) * 0.5, y_mid], dtype=float)
    pt_right_raw = np.array([(pR + iR) * 0.5, y_mid], dtype=float)

    # Build segments (pupil->iris) for margin adjustment
    seg_left  = (np.array([pL, y_mid], float), np.array([iL, y_mid], float))
    seg_right = (np.array([pR, y_mid], float), np.array([iR, y_mid], float))

    # 4) Adjust along each horizontal segment to meet polygon margin and stay away from both edges
    pt_left  = _ensure_inside_with_margin_on_segment(pt_left_raw,  seg_left[0],  seg_left[1],  eyelid_poly, margin_px)
    pt_right = _ensure_inside_with_margin_on_segment(pt_right_raw, seg_right[0], seg_right[1], eyelid_poly, margin_px)

    return pt_left, pt_right, seg_left, seg_right

# -----------------------
# Sclera points: NEW strategy (corner ↔ closest iris point inside polygon)
# -----------------------

def _find_eyelid_corners(poly_xy: np.ndarray):
    """
    Estimate two eyelid 'corners' via projection to principal axis (PCA). Fallback to x-extrema.
    Returns (corner_a, corner_b), each (2,).
    """
    if poly_xy is None or len(poly_xy) < 2:
        return (np.array([np.nan, np.nan], float), np.array([np.nan, np.nan], float))

    pts = np.asarray(poly_xy, float)
    mu = np.mean(pts, axis=0)
    X = pts - mu
    if len(pts) > 2:
        C = np.cov(X.T)
        w, V = np.linalg.eig(C)
        u = V[:, np.argmax(w.real)].real
        u /= (np.linalg.norm(u) + 1e-12)
        t = X @ u
        i_min = int(np.argmin(t))
        i_max = int(np.argmax(t))
        corner_a = pts[i_min]
        corner_b = pts[i_max]
        if np.linalg.norm(corner_b - corner_a) < 1e-6:
            i_min = int(np.argmin(pts[:, 0]))
            i_max = int(np.argmax(pts[:, 0]))
            corner_a = pts[i_min]
            corner_b = pts[i_max]
    else:
        corner_a = pts[0]
        corner_b = pts[-1]
    return corner_a.astype(float), corner_b.astype(float)

def _sample_ellipse_points(cx: float, cy: float, a: float, b: float, angle_deg: float,
                           num: int = 360) -> np.ndarray:
    """
    Sample points on a rotated ellipse perimeter.
    Returns array of shape (num, 2).
    """
    t = np.linspace(0.0, 2.0 * np.pi, num=num, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    th = np.deg2rad(angle_deg)
    cth, sth = np.cos(th), np.sin(th)
    x = cx + a * ct * cth - b * st * sth
    y = cy + a * ct * sth + b * st * cth
    return np.stack([x, y], axis=1)

def _closest_iris_point_inside_polygon(corner: np.ndarray,
                                       iris_pts: np.ndarray,
                                       eyelid_poly: np.ndarray) -> np.ndarray | None:
    """
    Among sampled iris points, return the one inside the eyelid polygon closest to the corner.
    If none are inside, return None.
    """
    inside_mask = np.array([_point_in_polygon(px, py, eyelid_poly) for px, py in iris_pts], dtype=bool)
    if not np.any(inside_mask):
        return None
    pts_inside = iris_pts[inside_mask]
    d2 = np.sum((pts_inside - corner[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return pts_inside[idx]



def _compute_sclera_points(
    pupil_cx: float, pupil_cy: float,
    ap: float, bp: float, pupil_ang: float,
    iris_cx: float, iris_cy: float,
    iris_w: float, iris_h: float,
    iris_angle_deg: float,
    eyelid_poly: np.ndarray,
    *,
    # Polygon margin: desired clearance along the perpendicular segment (optional trim)
    min_clearance_px: float = 30.0,
    # Minimal opening fallback (if perpendicular inside segment too short)
    min_allowed_px: float = 5.0,
    # Horizontal search for vertical intersections fallback (if needed)
    search_dx: int = 30,
    # NEW: fraction along the line from iris point to corner (0 -> at iris, 1 -> at corner)
    sclera_frac: float = 0.5,
    # Iris sampling resolution (for closest-point search inside polygon)
    iris_samples_num: int = 720,
):
    """
    New placement strategy:
      For each eyelid corner:
        1) Find the closest iris-ellipse point that lies inside the eyelid polygon.
        2) Place a candidate point p0 at fraction 'sclera_frac' along the segment (iris_point -> corner).
        3) At p0, take the line perpendicular to (iris_point -> corner) and intersect with the eyelid polygon.
           Choose the inside segment that *brackets p0* (nearest negative/positive line parameters).
        4) Final sclera point is the midpoint of that inside segment, optionally trimmed by min_clearance_px.

    Returns:
        (scl_left, scl_right) ordered by x (left then right). If a side is infeasible, returns [nan, nan].
    """
    # Iris semi-axes
    ai = iris_w / 2.0
    bi = iris_h / 2.0

    # --- corners and iris sampling ---
    corner_a, corner_b = _find_eyelid_corners(eyelid_poly)
    iris_samples = _sample_ellipse_points(iris_cx, iris_cy, ai, bi, iris_angle_deg, num=iris_samples_num)

    def _sclera_for_corner(corner: np.ndarray) -> np.ndarray:
        if corner is None or not np.all(np.isfinite(corner)):
            return np.array([np.nan, np.nan], float)

        # Closest iris perimeter point that is *inside* the eyelid polygon
        iris_pt = _closest_iris_point_inside_polygon(corner, iris_samples, eyelid_poly)
        if iris_pt is None:
            return np.array([np.nan, np.nan], float)

        # 1) Fractional placement along corner–iris line
        v = corner.astype(float) - iris_pt.astype(float)
        p0 = iris_pt.astype(float) + float(sclera_frac) * v

        # 2) Perpendicular line through p0
        vn = np.linalg.norm(v)
        if vn < 1e-9:
            return np.array([np.nan, np.nan], float)
        d = v / vn
        n_perp = np.array([-d[1], d[0]], float)  # 90° rotation

        # 3) Intersect infinite perpendicular line with polygon and choose the segment bracketing p0
        ts = _line_polygon_intersections(p0, n_perp, eyelid_poly)
        if len(ts) < 2:
            # Fallback: try nearby shifts along d to find bracketing segment
            found = None
            for k in range(1, search_dx + 1):
                for sign in (+1, -1):
                    p_try = p0 + sign * k * d
                    ts_try = _line_polygon_intersections(p_try, n_perp, eyelid_poly)
                    if len(ts_try) >= 2:
                        found = (p_try, ts_try)
                        break
                if found is not None:
                    break
            if found is None:
                return np.array([np.nan, np.nan], float)
            p0, ts = found

        # Choose nearest negative and nearest positive t to bracket p0
        t_minus_candidates = [t for t in ts if t < 0]
        t_plus_candidates  = [t for t in ts if t > 0]
        if len(t_minus_candidates) == 0 or len(t_plus_candidates) == 0:
            # If polygon is highly concave and doesn't bracket p0, fallback to widest segment
            t_lo, t_hi = ts[0], ts[-1]
        else:
            t_lo = max(t_minus_candidates)
            t_hi = min(t_plus_candidates)

        # Optional margin trim along perpendicular segment
        feasible_lo = t_lo + float(min_clearance_px)
        feasible_hi = t_hi - float(min_clearance_px)
        if feasible_hi >= feasible_lo:
            t_pick = 0.5 * (feasible_lo + feasible_hi)
        else:
            # If the opening is tight but >= 2*min_allowed_px, use segment midpoint
            if (t_hi - t_lo) >= 2.0 * float(min_allowed_px):
                t_pick = 0.5 * (t_lo + t_hi)
            else:
                return np.array([np.nan, np.nan], float)

        p = p0 + t_pick * n_perp

        # Numerical guard: ensure point is inside polygon (should be by construction)
        if not _point_in_polygon(p[0], p[1], eyelid_poly):
            # nudge toward perpendicular midpoint
            t_mid = 0.5 * (t_lo + t_hi)
            p = p0 + t_mid * n_perp

        return p

    pt1 = _sclera_for_corner(corner_a)
    pt2 = _sclera_for_corner(corner_b)

    # Order as (left, right) by x
    pts_fin = [pt1, pt2]
    finite_pts = [p for p in pts_fin if np.all(np.isfinite(p))]
    if len(finite_pts) == 0:
        return (np.array([np.nan, np.nan], float),
                np.array([np.nan, np.nan], float))
    if len(finite_pts) == 2:
        finite_sorted = sorted(finite_pts, key=lambda p: p[0])
        return finite_sorted[0], finite_sorted[1]

    p_only = finite_pts[0]
    x_center = float(np.mean(eyelid_poly[:, 0])) if eyelid_poly is not None and len(eyelid_poly) else p_only[0]
    if p_only[0] < x_center:
        return p_only, np.array([np.nan, np.nan], float)
    else:
        return np.array([np.nan, np.nan], float), p_only

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
    if poly_xy is None or len(poly_xy) < 2:
        return
    pts = np.round(poly_xy).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=thickness)

def _draw_ellipse(img, cx: float, cy: float, a: float, b: float, angle_deg: float,
                  color_bgr=(0, 255, 0), thickness=2):
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return
    center = (int(round(cx)), int(round(cy)))
    axes = (max(1, int(round(a))), max(1, int(round(b))))
    angle = float(angle_deg)
    cv2.ellipse(img, center, axes, angle, 0.0, 360.0, color_bgr, thickness)



def _save_frame_with_points(video_path: pathlib.Path,
                            frame_zero_based: int,
                            pupil_pt: np.ndarray,
                            iris_prompt_pts: list[np.ndarray],
                            sclera_left: np.ndarray,
                            sclera_right: np.ndarray,
                            out_png: pathlib.Path,
                            eyelid_poly: np.ndarray | None = None,
                            pupil_ellipse: tuple[float,float,float,float,float] | None = None,
                            iris_ellipse: tuple[float,float,float,float,float] | None = None,
                            eyelid_corners: tuple[np.ndarray, np.ndarray] | None = None,
                            iris_corner_pts: tuple[np.ndarray | None, np.ndarray | None] | None = None):
    """
    Save an annotated frame:
      - eyelid polygon
      - pupil & iris ellipses
      - markers: pupil (green), iris prompts (red), sclera L (blue), sclera R (magenta)
      - eyelid corners (cyan circles + small labels)
      - closest iris points to corners (orange circles + small labels)

    pupil_ellipse / iris_ellipse: (cx, cy, a, b, angle_deg)
    eyelid_corners: (corner_a, corner_b) each shape (2,)
    iris_corner_pts: (iris_at_corner_a, iris_at_corner_b) each shape (2,) or None if not found
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_zero_based)
        ok, img = cap.read()
        if not ok or img is None:
            raise RuntimeError(f"Failed to read frame {frame_zero_based} from {video_path}")

        # Draw eyelid polygon
        _draw_polygon(img, eyelid_poly, color_bgr=(0, 255, 255), thickness=2)  # yellow

        # Draw ellipses
        if pupil_ellipse is not None:
            pcx, pcy, pa, pb, pang = pupil_ellipse
            _draw_ellipse(img, pcx, pcy, pa, pb, pang, color_bgr=(0, 200, 0), thickness=2)
        if iris_ellipse is not None:
            icx, icy, ia, ib, iang = iris_ellipse
            _draw_ellipse(img, icx, icy, ia, ib, iang, color_bgr=(0, 0, 200), thickness=2)

        # Colors
        clr_pupil = (0, 255, 0)
        clr_irisP = (0, 0, 255)
        clr_scl_l = (255, 0, 0)
        clr_scl_r = (255, 0, 255)
        clr_corner = (255, 255, 0)   # cyan-ish
        clr_iris_corner = (0, 165, 255)  # orange

        def _finite(pt): return np.all(np.isfinite(pt))

        # Draw points
        if _finite(pupil_pt): _draw_marker(img, pupil_pt, clr_pupil)
        for ip in iris_prompt_pts:
            if _finite(ip): _draw_marker(img, ip, clr_irisP)
        if _finite(sclera_left):  _draw_marker(img, sclera_left,  clr_scl_l)
        if _finite(sclera_right): _draw_marker(img, sclera_right, clr_scl_r)

        # Draw eyelid corners
        if eyelid_corners is not None:
            corner_a, corner_b = eyelid_corners
            corners = []
            if corner_a is not None and _finite(corner_a): corners.append(('L', corner_a))
            if corner_b is not None and _finite(corner_b): corners.append(('R', corner_b))
            # Ensure consistent L/R by x-order if both present
            if len(corners) == 2 and corners[0][1][0] > corners[1][1][0]:
                corners = [corners[1], corners[0]]
            for tag, c in corners:
                cx_i = int(round(float(c[0])))
                cy_i = int(round(float(c[1])))
                cv2.circle(img, (cx_i, cy_i), radius=5, color=clr_corner, thickness=-1)
                cv2.putText(img, f"corner {tag}", (cx_i + 6, cy_i - 6),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=clr_corner, thickness=1, lineType=cv2.LINE_AA)

        # --- NEW: draw closest iris points to each corner ---
        if iris_corner_pts is not None:
            iris_a, iris_b = iris_corner_pts
            pts = []
            if iris_a is not None and _finite(iris_a): pts.append(('L', iris_a))
            if iris_b is not None and _finite(iris_b): pts.append(('R', iris_b))
            # If both present but swapped in x, reorder to L/R by x
            if len(pts) == 2 and pts[0][1][0] > pts[1][1][0]:
                pts = [pts[1], pts[0]]
            for tag, p in pts:
                px_i = int(round(float(p[0])))
                py_i = int(round(float(p[1])))
                cv2.circle(img, (px_i, py_i), radius=5, color=clr_iris_corner, thickness=-1)
                cv2.putText(img, f"iris@corner {tag}", (px_i + 6, py_i - 6),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=clr_iris_corner, thickness=1, lineType=cv2.LINE_AA)

        # Save (BGR->RGB)
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(out_png)
    finally:
        cap.release()

# -----------------------
# Prompt writer (sorted)
# -----------------------

def _as_int_xy(pt: np.ndarray) -> tuple[int, int]:
    return int(round(float(pt[0]))), int(round(float(pt[1])))

def write_prompt_to_file(prompt_obj: dict, out_path: str | pathlib.Path) -> None:
    """
    Write prompts to a tab-separated file with negatives per rule and sorted order:
      class order: pupil, iris, sclera; within each: positives first, then negatives.
    Negative rules:
      - Positive pupil → add iris 0 + sclera 0 at pupil coords
      - Positive iris (each) → add sclera 0 at iris coords
      - Positive sclera(s) → add iris 0 at each sclera coord
    """
    out_path = pathlib.Path(out_path)
    prom = prompt_obj['prompt']

    pupil_pts = prom['pupil']['points']          # (1,2)
    iris_pts  = prom['iris']['points']           # (M,2), M=1 or 2
    sclera_pts = prom['sclera']['points']        # (N,2), N=2; may contain NaNs

    lines: list[tuple[str, int, int, int]] = []

    # Positives
    sclera_int_pts: list[tuple[int, int]] = []
    if pupil_pts is not None and len(pupil_pts) > 0:
        px, py = _as_int_xy(pupil_pts[0])
        lines.append(('pupil', px, py, 1))
    if iris_pts is not None and len(iris_pts) > 0:
        for p in iris_pts:
            ix, iy = _as_int_xy(p)
            lines.append(('iris', ix, iy, 1))
    if sclera_pts is not None and len(sclera_pts) > 0:
        for p in sclera_pts:
            if np.all(np.isfinite(p)):
                sx, sy = _as_int_xy(p)
                sclera_int_pts.append((sx, sy))
                lines.append(('sclera', sx, sy, 1))

    # Negatives per rule
    if pupil_pts is not None and len(pupil_pts) > 0:
        px, py = _as_int_xy(pupil_pts[0])
        lines.append(('iris',   px, py, 0))
        lines.append(('sclera', px, py, 0))
    if iris_pts is not None and len(iris_pts) > 0:
        for p in iris_pts:
            ix, iy = _as_int_xy(p)
            lines.append(('sclera', ix, iy, 0))
    for (sx, sy) in sclera_int_pts:
        lines.append(('iris', sx, sy, 0))

    # Deduplicate preserving first occurrence
    first_index: dict[tuple[str,int,int,int], int] = {}
    for idx, rec in enumerate(lines):
        if rec not in first_index:
            first_index[rec] = idx
    records_with_idx = [(rec, idx) for rec, idx in first_index.items()]

    class_rank = {'pupil': 0, 'iris': 1, 'sclera': 2}
    def sort_key(item):
        (cls, x, y, lab), idx = item
        return (class_rank.get(cls, 999), 0 if lab == 1 else 1, idx)
    sorted_records = sorted(records_with_idx, key=sort_key)

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
                                 inside_margin_px: float = 10.0):
    """
    Extended:
      - Chooses frame based on BOTH pupil and iris validity (or uses override).
      - Returns pupil point (inside-with-margin), TWO iris prompt points (left/right) placed
        at vertical eyelid midpoint with horizontal midpoint between pupil & iris edges,
        adjusted along segment to be inside-with-margin and away from ellipse edges,
        and two sclera points (left/right) per your NEW strategy (corner ↔ closest iris point inside polygon).
      - Optionally saves a PNG overlay with eyelid polygon and ellipses.
    """
    video_path = pathlib.Path(video_path)
    gt_dir = pathlib.Path(gt_dir)

    pupil_file = _resolve_existing_file(video_path, gt_dir, 'pupil_eli.txt')
    iris_file  = _resolve_existing_file(video_path, gt_dir, 'iris_eli.txt')
    lid_file   = _resolve_existing_file(video_path, gt_dir, 'lid_lm_2D.txt')

    error = False
    try:
        pupil_gt = pd.read_csv(pupil_file, sep=';')
    except:
        print(f'error reading {pupil_file}, skipping')
        error = True
    try:
        iris_gt  = pd.read_csv(iris_file,  sep=';')
    except:
        print(f'error reading {iris_file}, skipping')
        error = True
    try:
        lid_gt   = pd.read_csv(lid_file,   sep=';')
    except:
        print(f'error reading {lid_file}, skipping')
        error = True
    if error:
        return

    valid_pupil = (pupil_gt['CENTER X'] != -1)
    valid_iris  = (iris_gt['CENTER X']  != -1)

    override_idx = frame_idx_override.get(video_path.name, None)
    fr_idx = _pick_frame_index_from_valid(valid_pupil, valid_iris, override_idx)

    # Pupil
    pupil_cx = float(pupil_gt['CENTER X'].iloc[fr_idx])
    pupil_cy = float(pupil_gt['CENTER Y'].iloc[fr_idx])
    pupil_w  = float(pupil_gt['WIDTH'].iloc[fr_idx])
    pupil_h  = float(pupil_gt['HEIGHT'].iloc[fr_idx])
    pupil_ang= float(pupil_gt['ANGLE'].iloc[fr_idx])
    ap = pupil_w / 2.0
    bp = pupil_h / 2.0

    # Iris
    iris_cx  = float(iris_gt['CENTER X'].iloc[fr_idx])
    iris_cy  = float(iris_gt['CENTER Y'].iloc[fr_idx])
    iris_w   = float(iris_gt['WIDTH'].iloc[fr_idx])
    iris_h   = float(iris_gt['HEIGHT'].iloc[fr_idx])
    iris_ang = float(iris_gt['ANGLE'].iloc[fr_idx])
    ai = iris_w / 2.0
    bi = iris_h / 2.0

    # Eyelid polygon
    eyelid_poly = _eyelid_polygon_from_row(lid_gt.iloc[fr_idx])

    # Sclera points (left, right) — NEW strategy
    scl_left, scl_right = _compute_sclera_points(
        pupil_cx, pupil_cy, ap, bp, pupil_ang,
        iris_cx, iris_cy, iris_w, iris_h, iris_ang,
        eyelid_poly,
        min_clearance_px=30.0,
        min_allowed_px=8.0,
        search_dx=30,
        sclera_frac=0.4  # <-- 33% from iris toward eye corner
    )

    # Iris prompt points at vertical eyelid mid (left/right)
    iris_prompt_left, iris_prompt_right, seg_left, seg_right = compute_iris_prompt_points_mid_eyelid(
        pupil_cx, pupil_cy, ap, bp, pupil_ang,
        iris_cx, iris_cy, ai, bi, iris_ang,
        eyelid_poly,
        margin_px=inside_margin_px,
        x_for_mid=iris_cx,
        scan_y_updown=40
    )

    # Ensure pupil point is inside-with-margin and away from pupil ellipse edge
    pupil_pt = _ensure_inside_with_margin(
        np.array([pupil_cx, pupil_cy], dtype=float),
        eyelid_poly,
        inside_margin_px,
        ellipse_center=(pupil_cx, pupil_cy),
        ellipse_a=ap, ellipse_b=bp,
        ellipse_angle_deg=pupil_ang
    )

    # GT frame (1-based) -> video frame index (0-based)
    frame_zero_based = int(pupil_gt['FRAME'].iloc[fr_idx]) - 1

    # Optional overlay
    if save_overlay:
        out_dir = pathlib.Path(overlay_out_dir) if overlay_out_dir is not None else video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{video_path.name}_frame{frame_zero_based:04d}.png"

        # Eyelid polygon
        eyelid_poly = _eyelid_polygon_from_row(lid_gt.iloc[fr_idx])

        # Compute eyelid corners
        corner_a, corner_b = _find_eyelid_corners(eyelid_poly)

        # --- NEW: closest iris points to each corner (inside polygon) ---
        # iris semi-axes already computed as ai, bi
        iris_samples = _sample_ellipse_points(iris_cx, iris_cy, ai, bi, iris_ang, num=720)
        iris_corner_a = _closest_iris_point_inside_polygon(corner_a, iris_samples, eyelid_poly)
        iris_corner_b = _closest_iris_point_inside_polygon(corner_b, iris_samples, eyelid_poly)

        _save_frame_with_points(
            video_path,
            frame_zero_based,
            pupil_pt,
            [iris_prompt_left, iris_prompt_right],
            scl_left,
            scl_right,
            out_png,
            eyelid_poly=eyelid_poly,
            pupil_ellipse=(pupil_cx, pupil_cy, ap, bp, pupil_ang),
            iris_ellipse=(iris_cx,  iris_cy,  ai, bi, iris_ang),
            eyelid_corners=(corner_a, corner_b),
            iris_corner_pts=(iris_corner_a, iris_corner_b)
        )

    prompt = {
        'prompt': {
            'pupil': {
                'points': pupil_pt.reshape(1, 2).astype(float),
                'labels': np.array([1], dtype=int),
                'box': None
            },
            'iris': {
                # TWO iris prompt points: left, right
                'points': np.stack([iris_prompt_left, iris_prompt_right], axis=0).astype(float),
                'labels': np.array([1, 1], dtype=int),
                'box': None
            },
            'sclera': {
                # order: left, right (may contain NaNs if skipped)
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

# -----------------------
# Example usage (script)
# -----------------------

if __name__ == '__main__':
    vid_dir = 'VIDEOS'
    gt_dir = 'ANNOTATIONS'
    root_dir = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\datasets real\TEyeD")
    prompt_out_dir = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\datasets real\TEyeD\prompts2")

    prompt_frame_hardcode = {'GW_5_7.mp4':10, 'GW_6_6.mp4':12}

    datasets = [fp for f in os.scandir(root_dir) if (fp:=pathlib.Path(f.path)).is_dir() and all((fp/s).is_dir() for s in [vid_dir,gt_dir])]

    for d in datasets:
        video_files = list((d/vid_dir).glob("*.mp4"))
        for v in video_files:
            retrieve_prompt_from_subject(v, d/gt_dir, prompt_frame_hardcode, True, prompt_out_dir/d.name)

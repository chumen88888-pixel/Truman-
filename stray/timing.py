"""Timing, road and staging for STRAY. Pure Python + numpy (no Blender)."""

from __future__ import annotations

import numpy as np

FPS = 60
DUR = 64.0
BPM = 105.0
BEAT = 60.0 / BPM
BAR = BEAT * 4.0
N = int(round(DUR * FPS))  # 3840


def bar(i: int) -> float:
    """Start time of a 1-based bar."""
    return (i - 1) * BAR


# Musical / picture events. Picture lands on the nearest frame (~one frame).
T_REV1 = bar(2)
T_REV2 = bar(3)
T_COPS = bar(4)
T_FLOOR = bar(5)
T_LAUNCH1 = bar(6)
T_DROP = bar(7)
T_SILENCE0 = T_DROP - 0.30
T_BOOM = bar(12)
T_LAUNCH2 = T_BOOM - 0.22
T_LAND2 = T_LAUNCH2 + 0.86
T_CHOPPER = bar(13)
T_BARRICADE = bar(15)
T_GLASS_IN = bar(17)
T_CURL = bar(18) + 0.20
T_COPS_FALL = bar(20)
T_GLASS_OUT = bar(21)
T_RING = bar(22)
T_SLO_IN = bar(24) - 0.85
T_SCREEN = bar(24)
T_SLO_OUT = bar(25) + 0.05
T_EDGE = bar(26)
T_IGNITE = bar(27)
T_FADE = DUR - 1.7


def frame_time(frame: int) -> float:
    """Blender frame (1-based) to seconds."""
    return (frame - 1) / FPS


def nearest_frame(t: float) -> int:
    return int(round(t * FPS)) + 1


def _smooth(a: float, b: float, t: float) -> float:
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    u = (t - a) / (b - a)
    return u * u * (3.0 - 2.0 * u)


def _lerp(keys, t: float) -> float:
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = _smooth(t0, t1, t) if (t1 - t0) < 1.2 else (t - t0) / (t1 - t0)
            # short gaps ease; long gaps stay linear so speed holds
            if t1 - t0 >= 1.2:
                u = (t - t0) / (t1 - t0)
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def sim_rate(t: float) -> float:
    """World clock rate. 1 = realtime. Picture, rain, lights and SFX share it."""
    r = 1.0
    # Bullet time over the alley blockade. Ease in, hold, snap back on the drop.
    if T_LAUNCH1 - 0.05 <= t <= T_DROP:
        u = _smooth(T_LAUNCH1 - 0.05, T_LAUNCH1 + 0.18, t)
        out = 1.0 - _smooth(T_DROP - 0.10, T_DROP + 0.02, t)
        r *= 1.0 - 0.86 * u * max(out, 0.0)
    # Screen smash.
    if T_SLO_IN <= t <= T_SLO_OUT + 0.15:
        u = _smooth(T_SLO_IN, T_SLO_IN + 0.20, t)
        out = 1.0 - _smooth(T_SLO_OUT, T_SLO_OUT + 0.18, t)
        r *= 1.0 - 0.72 * min(u, max(out, 0.0))
    return float(np.clip(r, 0.08, 1.0))


def desired_speed(t: float) -> float:
    """Metres per second along the road, in screen time (already includes slo-mo)."""
    keys = [
        (0.0, 0.0),
        (T_FLOOR, 0.0),
        # Short lunge so the blockade stays a few car-lengths ahead.
        (T_LAUNCH1 - 0.75, 0.0),
        (T_LAUNCH1 - 0.18, 11.0),
        (T_LAUNCH1, 15.0),
        (T_LAUNCH1 + 0.10, 8.4),
        (T_DROP - 0.06, 7.4),
        (T_DROP + 0.12, 64.0),  # ~230 km/h
        (T_LAUNCH2 - 0.4, 58.0),
        (T_LAUNCH2, 52.0),
        (T_LAND2, 56.0),
        (T_BARRICADE - 1.5, 54.0),
        (T_BARRICADE, 48.0),
        (T_GLASS_IN, 46.0),
        (T_CURL, 42.0),
        (T_GLASS_OUT, 40.0),
        (T_RING, 38.0),
        (T_SLO_IN, 34.0),
        (T_SLO_IN + 0.25, 13.0),
        (T_SCREEN, 12.0),
        (T_SLO_OUT, 14.0),
        (T_SLO_OUT + 0.35, 36.0),
        (T_EDGE - 0.3, 40.0),
        (T_EDGE, 42.0),
        (DUR, 42.0),
    ]
    return _lerp(keys, t)


def _catmull(p0, p1, p2, p3, u):
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
    )


def _build_centerline():
    pts = [np.array(p, dtype=np.float64) for p in [
        (0.0, -20.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 160.0, 0.0),
        (0.0, 420.0, 0.0),
        (0.0, 760.0, 0.0),
        (0.0, 1100.0, 0.0),
        (6.0, 1280.0, 2.0),
        (36.0, 1460.0, 10.0),
        (100.0, 1600.0, 24.0),
        (190.0, 1680.0, 36.0),
    ]]
    # Partial ring, then a straight elevated spur that simply stops.
    center = np.array([190.0, 1860.0, 40.0])
    radius = 180.0
    # South of center, sweeping counterclockwise toward the east.
    for a in np.linspace(-np.pi / 2, 0.55, 16):
        pts.append(
            center
            + np.array([np.sin(a) * radius, -np.cos(a) * radius, 0.0])
        )
    last = pts[-1]
    prev = pts[-2]
    direction = last - prev
    direction[2] = 0.0
    direction /= np.linalg.norm(direction)
    for dist in (40.0, 90.0, 150.0, 230.0, 320.0):
        p = last + direction * dist
        p[2] = 40.0 - dist * 0.012
        pts.append(p)

    # Dense Catmull-Rom, then arc-length resample every metre.
    dense = []
    m = len(pts)
    for i in range(m - 1):
        p0 = pts[max(0, i - 1)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(m - 1, i + 2)]
        steps = 40
        for s in range(steps):
            dense.append(_catmull(p0, p1, p2, p3, s / steps))
    dense.append(pts[-1])
    dense = np.stack(dense, axis=0)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    samples = np.arange(0.0, total, 1.0)
    pos = np.stack([np.interp(samples, arc, dense[:, k]) for k in range(3)], axis=1)
    return pos, total


def _frames(pos: np.ndarray):
    tang = np.gradient(pos, axis=0)
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(norm, 1e-8)
    n = len(pos)
    fwd = np.zeros_like(pos)
    right = np.zeros_like(pos)
    up = np.zeros_like(pos)
    for i in range(n):
        t = tang[i]
        u = np.array([0.0, 0.0, 1.0])
        u = u - t * np.dot(u, t)
        if np.linalg.norm(u) < 1e-5:
            u = np.array([0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        r = np.cross(t, u)
        r /= np.linalg.norm(r)
        u = np.cross(r, t)
        u /= np.linalg.norm(u)
        # Gentle banking from curvature. Camera itself stays level.
        if 0 < i < n - 1:
            turn = tang[i] - tang[i - 1]
            bank = float(np.clip(np.dot(turn, r) * -8.0, -0.11, 0.11))
        else:
            bank = 0.0
        c, s = np.cos(bank), np.sin(bank)
        r2 = r * c + u * s
        u2 = -r * s + u * c
        fwd[i], right[i], up[i] = t, r2, u2
    return fwd, right, up


def road_width(s: float) -> float:
    if s < 70:
        return 6.6
    if s < 120:
        return 6.6 + (18.5 - 6.6) * _smooth(70, 120, s)
    if s < 1280:
        return 18.5
    if s < 1500:
        return 18.5 + (12.0 - 18.5) * _smooth(1280, 1500, s)
    return 11.5


def _sample_road(pos, fwd, right, up, s):
    s = float(np.clip(s, 0.0, len(pos) - 1.001))
    i = int(np.floor(s))
    u = s - i
    j = min(i + 1, len(pos) - 1)

    def mix(a, b):
        return a * (1 - u) + b * u

    p = mix(pos[i], pos[j])
    f = mix(fwd[i], fwd[j])
    r = mix(right[i], right[j])
    uu = mix(up[i], up[j])
    f /= np.linalg.norm(f)
    r = r - f * np.dot(r, f)
    r /= np.linalg.norm(r)
    uu = np.cross(r, f)
    uu /= np.linalg.norm(uu)
    return p, f, r, uu


def _pitch(fwd, right, up, ang):
    c, s = np.cos(ang), np.sin(ang)
    f2 = fwd * c + up * s
    u2 = -fwd * s + up * c
    f2 /= np.linalg.norm(f2)
    u2 = np.cross(right, f2)
    u2 /= np.linalg.norm(u2)
    return f2, u2


def _jump(t: float) -> float:
    def hump(t0, t1, h):
        if t < t0 or t > t1:
            return 0.0
        u = (t - t0) / (t1 - t0)
        return h * np.sin(np.pi * u)

    return hump(T_LAUNCH1, T_DROP, 4.6) + hump(T_LAUNCH2, T_LAND2, 5.6)


def _body_pitch(t: float) -> float:
    ang = 0.0
    for tr, amp in ((T_REV1, 0.045), (T_REV2, 0.06)):
        dt = t - tr
        if 0.0 < dt < 0.55:
            ang += amp * np.sin(dt * 26.0) * np.exp(-dt * 5.0)
    ang += 0.10 * np.sin(np.pi * _smooth(T_LAUNCH1, T_DROP, t)) * (1 if T_LAUNCH1 <= t <= T_DROP else 0)
    if T_LAUNCH1 <= t <= T_DROP:
        u = (t - T_LAUNCH1) / (T_DROP - T_LAUNCH1)
        ang += 0.16 * np.sin(np.pi * u)
    if T_LAUNCH2 <= t <= T_LAND2:
        u = (t - T_LAUNCH2) / (T_LAND2 - T_LAUNCH2)
        ang += 0.20 * np.sin(np.pi * u)
    # Landing squat recovery is vertical, pitch kicks slightly down then up.
    for tr in (T_DROP, T_LAND2):
        dt = t - tr
        if 0.0 < dt < 0.35:
            ang += -0.05 * np.sin(np.pi * dt / 0.35)
    return float(ang)


def _cam_pose(t: float):
    keys = [
        # t, x, y, z, lookx, looky, lookz, fov
        # Whole car in frame, alley and the blockade readable down the bore.
        # High enough to read the whole car, then the two cruisers across the alley.
        (0.00, -2.4, -13.5, 3.6, 0.0, 6.5, 0.85, 42.0),
        (T_REV1, -1.6, -12.2, 3.3, 0.0, 8.0, 0.9, 38.0),
        (T_REV2, -0.9, -11.2, 3.05, 0.0, 9.0, 0.9, 34.0),
        (T_COPS - 0.15, -0.8, -14.0, 5.1, 0.0, 5.5, 1.35, 46.0),
        (T_COPS + 0.55, -0.4, -12.6, 4.6, 0.0, 6.5, 1.25, 40.0),
        (T_FLOOR, -1.4, -11.2, 3.4, 0.0, 6.0, 0.95, 40.0),
        (T_LAUNCH1, -1.8, -9.2, 3.3, 0.0, 3.5, 1.8, 44.0),
        (T_DROP, -2.4, -11.0, 4.6, 0.1, 4.5, 0.7, 48.0),
        # Wide enough to hold the tanker, the blast, and the car in the air.
        # Wide and a little behind the nose so the tanker and the blast sit in frame.
        (T_LAUNCH2 - 0.7, -5.2, -14.5, 4.6, 2.2, -1.5, 1.1, 58.0),
        (T_BOOM, -5.6, -13.5, 4.0, 2.8, -5.5, 1.3, 62.0),
        (T_BOOM + 0.4, -4.4, -12.5, 4.4, 1.6, -1.0, 1.5, 56.0),
        (T_CHOPPER, -1.6, -12.0, 3.6, 0.2, 8.0, 2.6, 50.0),
        (T_BARRICADE - 0.8, -1.6, -9.2, 2.5, 0.0, 7.0, 0.85, 44.0),
        (T_BARRICADE + 0.25, -1.5, -8.4, 2.3, 0.0, 6.0, 0.8, 42.0),
        # Ride the hood, just outside the glass. Inside the shell the camera
        # only sees bodywork, so the push stops on the windshield.
        (T_GLASS_IN - 1.6, -2.1, -4.2, 1.9, 0.0, 6.0, 0.7, 42.0),
        (T_GLASS_IN - 0.35, 0.05, 1.7, 1.32, 0.0, 8.0, 0.55, 50.0),
        (T_GLASS_IN + 0.25, -0.05, 1.25, 1.22, 0.0, 10.0, 0.5, 54.0),
        (T_CURL + 0.2, -0.2, 1.15, 1.28, -0.4, 12.0, 1.4, 56.0),
        (T_COPS_FALL, -0.25, 1.05, 1.32, -0.6, 14.0, 2.2, 54.0),
        (T_GLASS_OUT - 0.15, 0.0, 1.35, 1.35, 0.0, 10.0, 0.6, 50.0),
        (T_GLASS_OUT + 0.45, -1.8, -3.5, 2.2, 0.0, 8.0, 0.85, 44.0),
        (T_GLASS_OUT + 1.25, -2.0, -9.0, 3.05, 0.0, 12.0, 1.0, 42.0),
        (T_RING, -1.4, -8.4, 2.7, 0.0, 14.0, 1.1, 42.0),
        (T_SLO_IN, -0.6, -8.6, 2.7, 0.0, 16.0, 1.4, 44.0),
        (T_SCREEN, -0.4, -9.2, 2.5, 0.0, 11.0, 1.3, 46.0),
        (T_SLO_OUT + 0.25, -1.4, -9.0, 3.2, 0.0, 20.0, -0.4, 46.0),
        (T_EDGE - 0.2, -2.2, -9.4, 3.8, 0.0, 18.0, -4.0, 48.0),
        (T_EDGE + 0.7, -2.6, -8.8, 2.8, 0.1, 8.0, -1.2, 48.0),
        (T_IGNITE, -3.4, -12.5, 4.4, 0.0, 5.0, 0.35, 50.0),
        (T_IGNITE + 1.5, -2.6, -11.5, 3.6, 0.0, 14.0, 1.0, 48.0),
        (DUR, -1.8, -10.5, 2.7, 0.0, 20.0, 1.3, 44.0),
    ]
    if t <= keys[0][0]:
        return keys[0][1:]
    if t >= keys[-1][0]:
        return keys[-1][1:]
    for a, b in zip(keys, keys[1:]):
        if a[0] <= t <= b[0]:
            u = _smooth(a[0], b[0], t)
            return tuple(a[i] * (1 - u) + b[i] * u for i in range(1, 8))
    return keys[-1][1:]


def throttle_at(t: float) -> float:
    th = 0.22
    if t >= T_FLOOR:
        th = 0.22 + 0.78 * _smooth(T_FLOOR, T_LAUNCH1, t)
    for tr, amp in ((T_REV1, 0.55), (T_REV2, 0.72)):
        dt = t - tr
        if -0.02 < dt < 0.42:
            th = max(th, 0.25 + amp * np.sin(np.pi * dt / 0.42))
    if T_LAUNCH2 - 0.05 <= t <= T_LAND2:
        th = 1.0
    if T_IGNITE <= t:
        th = 1.0
    if T_SILENCE0 <= t < T_DROP:
        th *= 0.35
    return float(np.clip(th, 0.0, 1.0))


def build():
    """Return a dict of per-frame arrays and the road samples."""
    road_pos, road_len = _build_centerline()
    fwd, right, up = _frames(road_pos)
    t = np.arange(N) / FPS
    s = np.zeros(N)
    speed = np.zeros(N)
    v = 0.0
    for i in range(N):
        target = desired_speed(t[i])
        tau = 0.18
        if abs(t[i] - T_DROP) < 0.2:
            tau = 0.05
        v += (target - v) * min(1.0, (1.0 / FPS) / tau)
        v = max(0.0, v)
        speed[i] = v
        if i:
            s[i] = s[i - 1] + v / FPS
    s += 22.0  # hero starts a little way down the alley

    hero_pos = np.zeros((N, 3))
    hero_fwd = np.zeros((N, 3))
    hero_right = np.zeros((N, 3))
    hero_up = np.zeros((N, 3))
    wheel = np.zeros(N)
    steer = np.zeros(N)
    cam_pos = np.zeros((N, 3))
    cam_tgt = np.zeros((N, 3))
    fov = np.zeros(N)
    simt = np.zeros(N)
    rates = np.zeros(N)
    thrott = np.zeros(N)
    dist = 0.0

    # Integrate flight once we know the edge state. First pass fills the road part.
    edge_i = int(np.clip(round(T_EDGE * FPS), 1, N - 2))

    for i in range(N):
        rates[i] = sim_rate(t[i])
        simt[i] = (simt[i - 1] + rates[i] / FPS) if i else 0.0
        thrott[i] = throttle_at(t[i])
        p, f, r, u = _sample_road(road_pos, fwd, right, up, s[i])
        # Lane weave on the avenue only.
        if 140 < s[i] < 1280 and t[i] < T_GLASS_IN - 0.5:
            weave = 1.7 * np.sin((t[i] - T_DROP) * 0.95)
            weave *= _smooth(T_DROP, T_DROP + 0.6, t[i])
        else:
            weave = 0.0
        p = p + r * weave
        p = p + np.array([0.0, 0.0, _jump(t[i])])
        pit = _body_pitch(t[i])
        f, u = _pitch(f, r, u, pit)
        r = np.cross(f, u)
        r /= np.linalg.norm(r)
        hero_pos[i], hero_fwd[i], hero_right[i], hero_up[i] = p, f, r, u
        if i:
            dist += np.linalg.norm(hero_pos[i] - hero_pos[i - 1])
        wheel[i] = dist / 0.355
        if i:
            yaw = np.arctan2(hero_fwd[i, 0], hero_fwd[i, 1])
            yaw0 = np.arctan2(hero_fwd[i - 1, 0], hero_fwd[i - 1, 1])
            dy = (yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi
            steer[i] = float(np.clip(-dy * FPS * 0.55, -0.22, 0.22))

        cx, cy, cz, lx, ly, lz, fv = _cam_pose(t[i])
        cam_pos[i] = p + r * cx + f * cy + u * cz
        cam_tgt[i] = p + r * lx + f * ly + u * lz
        fov[i] = fv

    # Replace post-edge motion with a ballistic fall and a powered recovery.
    p0 = hero_pos[edge_i].copy()
    v0 = hero_fwd[edge_i] * max(speed[edge_i], 30.0)
    g = np.array([0.0, 0.0, -10.5])
    # Integrate from the edge so the car leaves the road on that exact frame.
    pos = p0.copy()
    vel = v0.copy()
    ign_i = int(round(T_IGNITE * FPS))
    for i in range(edge_i, N):
        dt = 1.0 / FPS
        if i < ign_i:
            vel = vel + g * dt
            pos = pos + vel * dt
            pit = -0.22 * _smooth(T_EDGE, T_EDGE + 0.8, t[i])
        else:
            # Thrust builds over a few frames: tail fire and belly jets together.
            burn = _smooth(T_IGNITE, T_IGNITE + 0.35, t[i])
            horiz = vel.copy()
            horiz[2] = 0.0
            hn = np.linalg.norm(horiz)
            forward = horiz / hn if hn > 1e-4 else hero_fwd[edge_i]
            accel = forward * (26.0 * burn) + np.array([0.0, 0.0, 34.0 * burn])
            # Kill the downward speed quickly, then climb.
            vel = vel + accel * dt
            vel[2] += (-vel[2]) * min(1.0, 4.0 * burn * dt) * 0.0
            if vel[2] < 0.0:
                vel[2] = vel[2] * (1.0 - 0.55 * burn) + 6.0 * burn * dt * 40.0
            pos = pos + vel * dt
            pit = -0.22 + 0.30 * burn
        f = vel.copy()
        if np.linalg.norm(f) < 1e-4:
            f = hero_fwd[edge_i].copy()
        f = f / np.linalg.norm(f)
        rr = np.cross(f, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(rr) < 1e-4:
            rr = np.array([1.0, 0.0, 0.0])
        rr /= np.linalg.norm(rr)
        uu = np.cross(rr, f)
        f, uu = _pitch(f, rr, uu, pit)
        rr = np.cross(f, uu)
        rr /= np.linalg.norm(rr)
        hero_pos[i], hero_fwd[i], hero_right[i], hero_up[i] = pos, f, rr, uu
        if i > edge_i:
            dist += np.linalg.norm(hero_pos[i] - hero_pos[i - 1])
        wheel[i] = dist / 0.355
        cx, cy, cz, lx, ly, lz, fv = _cam_pose(t[i])
        cam_pos[i] = pos + rr * cx + f * cy + uu * cz
        cam_tgt[i] = pos + rr * lx + f * ly + uu * lz
        fov[i] = fv
        speed[i] = float(np.linalg.norm(vel))

    # Cops.
    cop_a, cop_b, cop_c = _cops(t, s, road_pos, fwd, right, up, edge_i, hero_fwd)

    heli_pos, heli_fwd = _heli(t, hero_pos, hero_fwd, hero_right)

    return {
        "t": t,
        "s": s,
        "speed": speed,
        "hero_pos": hero_pos,
        "hero_fwd": hero_fwd,
        "hero_right": hero_right,
        "hero_up": hero_up,
        "wheel": wheel,
        "steer": steer,
        "cam_pos": cam_pos,
        "cam_tgt": cam_tgt,
        "fov": fov,
        "simt": simt,
        "rate": rates,
        "throttle": thrott,
        "road_pos": road_pos,
        "road_fwd": fwd,
        "road_right": right,
        "road_up": up,
        "road_len": road_len,
        "edge_i": edge_i,
        "cop_a": cop_a,
        "cop_b": cop_b,
        "cop_c": cop_c,
        "heli_pos": heli_pos,
        "heli_fwd": heli_fwd,
        "s_tank": float(s[int(round(T_LAUNCH2 * FPS))]),
        "s_bar": float(s[int(round(T_BARRICADE * FPS))]),
        "s_curl": float(s[int(round(T_CURL * FPS))]),
        "s_screen": float(s[min(int(round(T_SCREEN * FPS)), edge_i)]),
        "s_edge": float(s[edge_i]),
    }


def _road_at(road_pos, fwd, right, up, s):
    return _sample_road(road_pos, fwd, right, up, s)


def _cops(t, s, road_pos, fwd, right, up, edge_i, hero_fwd):
    """Each cop is an array (N, 3) position plus a 'live' mask and a yaw vector."""
    n = len(t)
    launch_i = int(round(T_LAUNCH1 * FPS))
    boom_i = int(round(T_BOOM * FPS))
    fall_i = int(round(T_COPS_FALL * FPS))
    s_block = float(s[launch_i] + 4.2)
    s_tank = float(s[int(round(T_LAUNCH2 * FPS))])

    def pack(offsets_s, offsets_x, dead_after, drop_from=None):
        pos = np.zeros((n, 3))
        fwd_a = np.zeros((n, 3))
        live = np.ones(n)
        for i in range(n):
            ss = offsets_s[i]
            p, f, r, u = _road_at(road_pos, fwd, right, up, ss)
            p = p + r * offsets_x[i]
            if drop_from is not None and i >= drop_from:
                dt = (i - drop_from) / FPS
                # Thrown upward then down. Impact near the planned beat.
                vz = 9.0
                p = p + np.array([0.0, 0.0, vz * dt - 0.5 * 16.0 * dt * dt])
                if p[2] < road_pos[int(np.clip(ss, 0, len(road_pos) - 1)), 2] + 0.4:
                    p[2] = road_pos[int(np.clip(ss, 0, len(road_pos) - 1)), 2] + 0.4
                    live[i] = 0.0
            if i >= dead_after:
                live[i] = 0.0
            pos[i] = p
            fwd_a[i] = f
        return {"pos": pos, "fwd": fwd_a, "live": live}

    # Cop A: waits across the alley, then trails ~9 m back and hits the tanker.
    off_s = np.zeros(n)
    off_x = np.zeros(n)
    for i in range(n):
        if t[i] < T_DROP + 0.15:
            off_s[i] = s_block
            # Slide in from the right, stopped by T_COPS.
            slide = _smooth(T_COPS - 1.05, T_COPS, t[i])
            off_x[i] = 14.0 * (1.0 - slide) + 2.2 * slide
        elif t[i] < T_BOOM:
            off_s[i] = s[i] - 9.0
            off_x[i] = 3.4
        else:
            off_s[i] = s_tank
            off_x[i] = 2.2
    cop_a = pack(off_s, off_x, boom_i)

    # Cop B: the partner blocker, then the survivor all the way to the cliff.
    off_s = np.zeros(n)
    off_x = np.zeros(n)
    for i in range(n):
        if t[i] < T_DROP + 0.25:
            off_s[i] = s_block
            slide = _smooth(T_COPS - 1.05, T_COPS + 0.1, t[i])
            off_x[i] = -16.0 * (1.0 - slide) - 2.2 * slide
        else:
            off_s[i] = min(s[i] - 16.0, s[min(i, edge_i)] - 8.0)
            # After the tanker, move to a clear lane.
            off_x[i] = -3.6 if t[i] > T_BOOM + 0.3 else -2.2
            if t[i] > T_EDGE:
                # Follows off the edge and falls, no jets.
                dt = t[i] - T_EDGE
                off_s[i] = s[edge_i] - 10.0
        if t[i] > T_EDGE:
            off_x[i] = -2.0
    cop_b = pack(off_s, off_x, n + 1)
    # Pull B off the cliff with the hero's horizontal direction.
    for i in range(edge_i, n):
        dt = (i - edge_i) / FPS
        base = cop_b["pos"][edge_i]
        cop_b["pos"][i] = base + hero_fwd[edge_i] * (20.0 * dt) + np.array([0.0, 0.0, -0.5 * 10.5 * dt * dt])
        cop_b["live"][i] = 1.0 if dt < 2.4 else 0.2

    # Cop C: joins from a side street, dies in the curl.
    join_i = int(round((T_CHOPPER + 0.2) * FPS))
    off_s = np.zeros(n)
    off_x = np.full(n, -4.2)
    for i in range(n):
        if i < join_i:
            off_s[i] = s[join_i] + 18.0
            off_x[i] = 22.0 * (1.0 - _smooth(T_CHOPPER - 1.2, T_CHOPPER + 0.4, t[i])) - 4.0
        elif t[i] < T_CURL:
            off_s[i] = s[i] - 12.0
            off_x[i] = -4.3
        else:
            off_s[i] = s[int(round(T_CURL * FPS))] - 6.0
            off_x[i] = -4.3
    cop_c = pack(off_s, off_x, fall_i, drop_from=int(round((T_COPS_FALL - 0.85) * FPS)))
    return cop_a, cop_b, cop_c


def _heli(t, hero_pos, hero_fwd, hero_right):
    n = len(t)
    pos = np.zeros((n, 3))
    fwd = np.zeros((n, 3))
    for i in range(n):
        # Behind and high until the helicopter beat, then it drops into frame,
        # then climbs away so the ending belongs to the car.
        if t[i] < T_CHOPPER - 1.6:
            lead = -28.0
            height = 60.0
        elif t[i] < T_BARRICADE + 0.8:
            u = _smooth(T_CHOPPER - 1.6, T_CHOPPER + 0.35, t[i])
            lead = -28.0 + 50.0 * u
            height = 60.0 - 46.0 * u
        else:
            u = _smooth(T_BARRICADE + 0.8, T_GLASS_IN, t[i])
            lead = 22.0 + 50.0 * u
            height = 14.0 + 55.0 * u
        side = 1.2
        pos[i] = hero_pos[i] + hero_fwd[i] * lead + hero_right[i] * side
        pos[i, 2] = max(hero_pos[i, 2] + height, height)
        fwd[i] = hero_fwd[i]
    return pos, fwd


def report(data=None):
    data = data or build()
    t = data["t"]
    print(f"frames {N}  road {data['road_len']:.0f} m  max hero s {data['s'].max():.0f} m")
    print(f"s_tank {data['s_tank']:.1f}  s_bar {data['s_bar']:.1f}  s_curl {data['s_curl']:.1f}")
    print(f"s_screen {data['s_screen']:.1f}  s_edge {data['s_edge']:.1f}")
    names = [
        ("rev1", T_REV1), ("rev2", T_REV2), ("cops", T_COPS), ("floor", T_FLOOR),
        ("launch1", T_LAUNCH1), ("drop", T_DROP), ("launch2", T_LAUNCH2),
        ("boom", T_BOOM), ("chopper", T_CHOPPER), ("barricade", T_BARRICADE),
        ("glass_in", T_GLASS_IN), ("curl", T_CURL), ("cops_fall", T_COPS_FALL),
        ("glass_out", T_GLASS_OUT), ("screen", T_SCREEN), ("edge", T_EDGE),
        ("ignite", T_IGNITE),
    ]
    for name, tt in names:
        i = int(np.clip(round(tt * FPS), 0, N - 1))
        print(
            f"{name:10} t={tt:7.3f} f={i+1:4d} s={data['s'][i]:7.1f} "
            f"v={data['speed'][i]:5.1f} rate={data['rate'][i]:.2f} "
            f"z={data['hero_pos'][i,2]:6.1f}"
        )
    # Camera / car separation
    d = np.linalg.norm(data["cam_pos"] - data["hero_pos"], axis=1)
    print(f"cam dist min {d.min():.2f} max {d.max():.2f}")
    if data["s"].max() > data["road_len"] - 5:
        print("WARNING: hero runs off the sampled road before the edge")
    return data


if __name__ == "__main__":
    report()

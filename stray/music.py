"""Original 105 BPM score for STRAY. One continuous pass, not sliced.

28 bars of 4/4. A sparse intro, a riser, a short silence, then the drop
on bar 7 — the same instant the car lands. Later downbeats carry the
impacts, but the grid never breaks and slow motion does not stretch it.
"""

from __future__ import annotations

import os

import numpy as np

import timing as T

SR = 48000
DUR = T.DUR
N = int(DUR * SR)


def _env(n, attack, release):
    t = np.arange(n) / SR
    a = np.minimum(t / max(attack, 1e-4), 1.0)
    rel = np.exp(-np.maximum(t - attack, 0.0) / max(release, 1e-4))
    return a * rel


def _kick():
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    f = 46 + 130 * np.exp(-t * 26)
    phase = np.cumsum(f) / SR * 2 * np.pi
    body = np.sin(phase) * np.exp(-t * 9.5)
    click = np.random.default_rng(1).standard_normal(n) * np.exp(-t * 90) * 0.35
    return (body + click).astype(np.float32)


def _hat():
    n = int(0.045 * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(2).standard_normal(n)
    return (noise * np.exp(-t * 55)).astype(np.float32)


def _snare():
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(3).standard_normal(n)
    tone = np.sin(2 * np.pi * 196 * t) * np.exp(-t * 22)
    return (noise * np.exp(-t * 16) * 0.7 + tone * 0.35).astype(np.float32)


def _braam():
    n = int(0.7 * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for f in (55, 82.5, 110, 164):
        sig += np.sin(2 * np.pi * f * t) * (0.4 if f < 100 else 0.18)
    sig = np.tanh(sig * 2.2)
    return (sig * np.exp(-t * 3.2)).astype(np.float32)


def _place(buf, sig, t0, gain, pan=0.0):
    if t0 >= DUR or gain == 0:
        return
    i0 = int(t0 * SR)
    i1 = min(N, i0 + len(sig))
    if i1 <= i0:
        return
    chunk = sig[: i1 - i0] * gain
    left = np.sqrt(0.5 * (1 - pan))
    right = np.sqrt(0.5 * (1 + pan))
    buf[i0:i1, 0] += chunk * left
    buf[i0:i1, 1] += chunk * right


def render():
    rng = np.random.default_rng(7)
    buf = np.zeros((N, 2), np.float32)
    kick = _kick()
    hat = _hat()
    snare = _snare()
    braam = _braam()
    bar = T.BAR
    beat = T.BEAT

    # Drums. Intro is kicks on the downbeat only. After the drop, four on the floor.
    drop = T.T_DROP
    for i in range(28 * 4):
        t = i * beat
        if t < drop - 0.05 and (i % 4) != 0:
            continue
        if abs(t - drop) < 0.02:
            _place(buf, kick, t, 1.15, 0.0)
        elif t < drop:
            _place(buf, kick, t, 0.55, 0.0)
        else:
            _place(buf, kick, t, 0.92, 0.0)
        if t >= drop and (i % 2) == 1:
            _place(buf, snare, t, 0.38, 0.05)
        if t >= drop - bar:
            _place(buf, hat, t, 0.16, 0.35 if i % 2 == 0 else -0.35)
            _place(buf, hat, t + beat * 0.5, 0.09, -0.2)

    # The two engine revs get a quiet tick so the picture has something to sit on.
    for t, g in ((T.T_REV1, 0.22), (T.T_REV2, 0.28)):
        _place(buf, kick, t, g)

    # Bass: D1, following the kick grid after the drop, a pulse before it.
    bass_n = int(0.34 * SR)
    tt = np.arange(bass_n) / SR
    for i, t in enumerate(np.arange(0, DUR, beat)):
        if T.T_SILENCE0 <= t < drop:
            continue
        f = 36.7 if (int(t / bar) % 2 == 0) else 43.65
        if t < drop:
            f = 36.7
            if int(t / beat) % 4 not in (0,):
                continue
        sig = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 6.5)
        sig = np.tanh(sig * 1.6)
        gain = 0.28 if t < drop else 0.55
        _place(buf, sig.astype(np.float32), t, gain)

    # Minor pad, wide, quiet in the intro, open after the drop.
    pad_t = np.arange(N) / SR
    pad = np.zeros((N, 2), np.float32)
    for f, pan in ((146.8, -0.6), (174.6, 0.15), (220.0, 0.65), (293.7, -0.25)):
        tone = np.sin(2 * np.pi * f * pad_t + np.sin(pad_t * 0.3) * 0.4)
        # Open only outside the silence.
        tone *= 0.5 + 0.5 * np.sin(2 * np.pi * pad_t / bar)
        left = np.sqrt(0.5 * (1 - pan))
        right = np.sqrt(0.5 * (1 + pan))
        pad[:, 0] += tone.astype(np.float32) * left
        pad[:, 1] += tone.astype(np.float32) * right
    pad_gain = np.full(N, 0.045, np.float32)
    # Rise through bars 5-6, gone in the silence, present after the drop.
    for i, t in enumerate(pad_t):
        if t < T.T_FLOOR:
            pad_gain[i] = 0.03
        elif t < T.T_SILENCE0:
            u = (t - T.T_FLOOR) / (T.T_SILENCE0 - T.T_FLOOR)
            pad_gain[i] = 0.03 + 0.08 * u
        elif t < drop:
            pad_gain[i] = 0.0
        else:
            pad_gain[i] = 0.07
    buf += pad * pad_gain[:, None]

    # Riser: noise swept by an exponential, bars 5-6, into the silence.
    rise_n = int((T.T_SILENCE0 - T.T_FLOOR) * SR)
    rt = np.arange(rise_n) / SR
    noise = rng.standard_normal(rise_n).astype(np.float32)
    # One-pole sweep.
    cut = 0.02 + 0.55 * (rt / max(rt[-1], 1e-6)) ** 2
    y = 0.0
    swept = np.empty_like(noise)
    for i in range(rise_n):
        y += cut[i] * (noise[i] - y)
        swept[i] = y
    swept *= (rt / max(rt[-1], 1e-6)) ** 1.4
    swept *= np.linspace(0.0, 1.0, rise_n).astype(np.float32)
    i0 = int(T.T_FLOOR * SR)
    buf[i0:i0 + rise_n, 0] += swept * 0.22
    buf[i0:i0 + rise_n, 1] += swept * 0.22

    # Snare roll into the silence.
    roll_start = T.T_FLOOR + 0.4
    step = 0.24
    t = roll_start
    while t < T.T_SILENCE0:
        _place(buf, snare, t, 0.12 + 0.25 * ((t - roll_start) / (T.T_SILENCE0 - roll_start)))
        step = max(0.055, step * 0.86)
        t += step

    # Silence, with a 12 ms edge so it isn't a click.
    s0 = int(T.T_SILENCE0 * SR)
    s1 = int(drop * SR)
    fade = int(0.012 * SR)
    if s1 > s0:
        buf[s0:s1] *= 0
        if s0 >= fade:
            ramp = np.linspace(1, 0, fade, dtype=np.float32)[:, None]
            buf[s0 - fade:s0] *= ramp

    # Picture hits that want a braam. They already sit on downbeats.
    for t, g, pan in (
        (T.T_BOOM, 0.7, 0.15),
        (T.T_BARRICADE, 0.55, -0.1),
        (T.T_COPS_FALL, 0.6, 0.0),
        (T.T_SCREEN, 0.75, 0.05),
        (T.T_IGNITE, 0.85, 0.0),
    ):
        _place(buf, braam, t, g, pan)

    # Fade with the picture. The score itself is not time-stretched in slow motion.
    fade_n = int(1.7 * SR)
    ramp = np.linspace(1, 0, fade_n, dtype=np.float32)[:, None]
    buf[-fade_n:] *= ramp

    peak = float(np.max(np.abs(buf))) + 1e-9
    buf *= min(1.0, 0.9 / peak)
    return np.tanh(buf * 1.1).astype(np.float32)


def main():
    audio = render()
    out = os.path.join(os.path.dirname(__file__), "..", "build", "music.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import wave

    pcm = np.clip(audio, -1, 1)
    ints = (pcm * 32767).astype(np.int16)
    with wave.open(out, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(ints.tobytes())
    # Report the silence and the drop so the grid can be checked.
    s0 = int(T.T_SILENCE0 * SR)
    s1 = int(T.T_DROP * SR)
    silence = np.sqrt(np.mean(audio[s0:s1] ** 2))
    drop = np.max(np.abs(audio[s1:s1 + int(0.05 * SR)]))
    print(f"music {out} silence_rms {silence:.5f} drop_peak {drop:.3f}")


if __name__ == "__main__":
    main()

"""Mix the score with real recordings. Diegetic sounds follow sim_rate.

Engine, siren, rotor, glass, rain and the cannon are recordings.
The score is the only synthesized part, and it stays on the 105 BPM grid.
"""

from __future__ import annotations

import os
import subprocess
import wave

import numpy as np

import music
import timing as T

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SFX = os.path.join(ROOT, "assets", "sfx")
SR = music.SR
N = music.N


def load_mono(path):
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    )
    return np.frombuffer(raw, np.float32).copy()


def looped(src, fade_s=0.05):
    src = np.asarray(src, np.float32).copy()
    n = min(int(fade_s * SR), len(src) // 4)
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    src[:n] = src[:n] * ramp + src[-n:] * (1 - ramp)
    return src


def resample(src, rate, n, gain):
    """Play src at a per-sample rate (source samples per output sample), looping."""
    if len(src) < 4:
        return np.zeros(n, np.float32)
    pos = np.cumsum(rate).astype(np.float64)
    span = len(src) - 2
    pos = np.mod(pos, span)
    i0 = np.floor(pos).astype(np.int32)
    frac = (pos - i0).astype(np.float32)
    out = src[i0] * (1 - frac) + src[i0 + 1] * frac
    return out * gain


def place(buf, sig, t0, gain, pan=0.0):
    i0 = int(t0 * SR)
    if i0 >= N or gain == 0:
        return
    i1 = min(N, i0 + len(sig))
    chunk = sig[: i1 - i0] * gain
    buf[i0:i1, 0] += chunk * np.sqrt(0.5 * (1 - pan))
    buf[i0:i1, 1] += chunk * np.sqrt(0.5 * (1 + pan))


def to_samples(frame_values):
    hop = SR // T.FPS
    return np.repeat(np.asarray(frame_values, np.float32), hop)[:N]


def write_wav(path, audio):
    pcm = np.clip(audio, -1, 1)
    ints = (pcm * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(ints.tobytes())


def main():
    data = T.build()
    t = to_samples(data["t"])
    rate = to_samples(data["rate"])
    throttle = to_samples(data["throttle"])
    hero = data["hero_pos"]
    # Distance from the hero to the nearest living pursuer, held per frame.
    dist = np.full(T.N, 80.0, np.float32)
    for info in (data["cop_a"], data["cop_b"], data["cop_c"]):
        d = np.linalg.norm(info["pos"] - hero, axis=1)
        live = info["live"] > 0.5
        dist = np.where(live, np.minimum(dist, d), dist)
    dist_s = to_samples(dist)

    music_path = os.path.join(ROOT, "build", "music.wav")
    if not os.path.exists(music_path):
        music.main()
    score = load_mono(music_path)
    # music.wav is stereo; reload properly
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", music_path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    )
    score = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    if len(score) < N:
        score = np.pad(score, ((0, N - len(score)), (0, 0)))
    score = score[:N]

    engine = looped(load_mono(os.path.join(SFX, "regera.ogg"))[int(1.6 * SR):])
    # Idle is slower; the two revs and the run open the throttle.
    eng_rate = rate * (0.42 + 0.95 * throttle)
    eng_gain = (0.18 + 0.72 * throttle) * (0.55 + 0.45 * rate)
    eng = resample(engine, eng_rate, N, eng_gain.astype(np.float32))

    siren_src = looped(load_mono(os.path.join(SFX, "siren_pd.ogg")), 0.08)
    siren_gain = np.clip(18.0 / (dist_s + 6.0), 0.0, 0.62).astype(np.float32)
    siren_gain *= (t >= (T.T_COPS - 0.5)).astype(np.float32)
    # The recording is distant; a little extra level, still pitched by the world clock.
    siren = resample(siren_src, np.maximum(rate, 0.2), N, siren_gain * 1.4)

    rotor_src = looped(load_mono(os.path.join(SFX, "chinook.ogg")), 0.1)
    heli = data["heli_pos"]
    height = np.maximum(heli[:, 2] - hero[:, 2], 0.0)
    lead = np.linalg.norm(heli[:, :2] - hero[:, :2], axis=1)
    near = np.clip(1.0 - (height - 10.0) / 40.0, 0.0, 1.0) * np.clip(1.0 - lead / 70.0, 0.0, 1.0)
    rotor_gain = to_samples(near) * ((t >= T.T_CHOPPER - 1.8) & (t <= T.T_BARRICADE + 1.2)).astype(np.float32)
    rotor = resample(rotor_src, np.maximum(rate, 0.25) * 0.85, N, rotor_gain.astype(np.float32) * 0.55)

    rain_src = looped(load_mono(os.path.join(SFX, "rain.ogg")), 0.1)
    rain_gain = (0.05 + 0.06 * rate).astype(np.float32)
    rain = resample(rain_src, np.maximum(rate, 0.35) * 0.8, N, rain_gain)

    cannon = load_mono(os.path.join(SFX, "cannon.wav"))
    bang = cannon[int(11.82 * SR): int(14.4 * SR)]
    glass = load_mono(os.path.join(SFX, "glass1.wav"))

    bed = np.zeros((N, 2), np.float32)
    bed[:, 0] += eng * 0.9
    bed[:, 1] += eng * 0.9
    bed[:, 0] += siren * 0.85
    bed[:, 1] += siren * 1.05
    bed[:, 0] += rotor * 1.05
    bed[:, 1] += rotor * 0.9
    bed[:, 0] += rain
    bed[:, 1] += rain

    # World clock also slows the one-shots.
    def slowed(src, t0, gain, pan, min_rate=0.35):
        i = int(np.clip(round(t0 * T.FPS), 0, T.N - 1))
        play = max(float(data["rate"][i]), min_rate)
        # Stretch by stepping through the source slower.
        n = int(len(src) / play)
        idx = np.minimum((np.arange(n) * play).astype(np.int32), len(src) - 1)
        sig = src[idx]
        place(bed, sig, t0, gain, pan)

    slowed(bang, T.T_BOOM, 0.95, 0.1, 0.7)
    slowed(bang, T.T_COPS_FALL, 0.8, -0.15, 0.7)
    slowed(bang, T.T_BARRICADE, 0.45, 0.0, 0.75)
    slowed(glass, T.T_SCREEN, 0.7, 0.05, 0.4)

    # The musical silence stays clean: diegetic beds duck, the one-shots above
    # the drop are already past it.
    s0 = int(T.T_SILENCE0 * SR)
    s1 = int(T.T_DROP * SR)
    bed[s0:s1] *= 0.08

    mix = score * 0.92 + bed
    fade_n = int(1.7 * SR)
    ramp = np.linspace(1, 0, fade_n, dtype=np.float32)[:, None]
    mix[-fade_n:] *= ramp
    peak = float(np.max(np.abs(mix))) + 1e-9
    mix *= min(1.0, 0.89 / peak)
    mix = np.tanh(mix * 1.05).astype(np.float32)

    out = os.path.join(ROOT, "build", "mix.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_wav(out, mix)
    silence = float(np.sqrt(np.mean(mix[s0:s1] ** 2)))
    drop = float(np.max(np.abs(mix[s1:s1 + int(0.04 * SR)])))
    print(f"mix {out} silence_rms {silence:.5f} drop_peak {drop:.3f} peak {float(np.max(np.abs(mix))):.3f}")


if __name__ == "__main__":
    main()

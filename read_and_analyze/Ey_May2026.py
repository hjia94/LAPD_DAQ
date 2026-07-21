#!/usr/bin/env python3
"""
Run-10 plane parse: Langmuir Isat (slowscope) + Ey dipole (fastscope), both
acquired at the same Hades p22 positions on a shared LAPD trigger. Each scope
has its own LeCroy WAVEDESC giving dt, t0, gain, offset — read at runtime.

The two scopes share the same master-trigger zero. WAVEDESC HORIZ_OFFSET
(= t0_ms here) is the time of the first sample relative to that trigger:
  slowscope  t0 ≈ -10 ms, total 50 ms  →  captures the full discharge
  fastscope  t0 ≈ -0.28 ms, total 5 ms →  captures the first ~4.7 ms

Writes postprocessing/run{N}_C{K}_{label}_tbeg{a}_tend{b}.npz and a (x=0,y=0)
diagnostic trace per section.
"""

from __future__ import annotations
from pathlib import Path
import struct
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def parse_lecroy_header(hdr_bytes: bytes) -> dict:
    """Pull dt/t0/gain/offset out of a LeCroy WAVEDESC block.
    Physical voltage = raw * gain - offset."""
    b = bytes(hdr_bytes)
    idx = b.find(b"WAVEDESC")
    if idx < 0:
        raise ValueError("WAVEDESC magic not found in header")
    end = "<" if struct.unpack_from("<h", b, idx + 34)[0] == 1 else ">"
    return dict(
        dt_ms  = struct.unpack_from(end + "f", b, idx + 176)[0] * 1e3,
        t0_ms  = struct.unpack_from(end + "d", b, idx + 180)[0] * 1e3,
        gain   = struct.unpack_from(end + "f", b, idx + 156)[0],
        offset = struct.unpack_from(end + "f", b, idx + 160)[0],
    )


# --- zoe-plotting style (usetex off — works on midas without latex) --------
plt.rcParams.update({
    "text.usetex":           False,
    "font.family":           "serif",
    "font.size":             13,
    "axes.linewidth":        1.5,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.top":             True,
    "ytick.right":           True,
    "xtick.minor.visible":   True,
    "ytick.minor.visible":   True,
    "legend.frameon":        False,
})

# --- parameters ------------------------------------------------------------
HERE     = Path(__file__).resolve().parent
DATA_DIR = r"M:\BAPSF_Data\Low_Density_Topo\May2026"

RUN              = 19
OUT_DIR  = Path(r"G:\processed_data") / f"run{RUN}"
FIG_DIR  = Path(r"E:\Shadow data\Pat\plots") / f"run{RUN}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
MG_PATH          = "Control/Positions/<Hades>    p22_EP"
NUM_SHOT         = 5
OUTLIER_K        = 3.0    # shots with |x - median| > k * 1.4826 * MAD → NaN
FORCE_REPROCESS  = False  # set True to ignore existing .npz and re-parse
FFT_FREQ_HZ      = 2.5e9  # nominal target frequency for "fft" reducer
FFT_SEARCH_HALFBW_HZ    = 100e6  # per-shot peak search: FFT_FREQ_HZ ± this
FFT_INTEGRATE_HALFBW_HZ = 50e3   # coherent integration band around peak
TEST_ONLY        = False  # True: only do the (0,0) diagnostic, skip plane parse

# Each section is (scope, channel, tbeg_ms, tend_ms, label, ylabel, reducer)
# reducer = "mean" → ⟨V⟩ (good for DC like Isat)
# reducer = "rms"  → sqrt(⟨V²⟩) (AC magnitude, broadband)
# reducer = "fft"  → 2|FFT(V)|/N at FFT_FREQ_HZ (single-tone amplitude)
SECTIONS = [
    # ("slowscope", 1, 0.0, 20.0, "isat", r"$I_\mathrm{sat}$ (V)",   "mean"),
    ("fastscope", 2, 0.02, 0.03, "ey_1p15GHz", r"$|E_y(2.5\,\mathrm{GHz})|$ (V)", "fft"),
]


def mad_filter(values: np.ndarray, k: float = OUTLIER_K) -> np.ndarray:
    """Robust outlier mask along the last axis: any sample > k * 1.4826 * MAD
    from the median is set to NaN. NaN-aware; no-op if MAD == 0."""
    v   = values.astype(np.float64, copy=True)
    med = np.nanmedian(v, axis=-1, keepdims=True)
    mad = np.nanmedian(np.abs(v - med), axis=-1, keepdims=True)
    bad = (mad > 0) & (np.abs(v - med) > k * 1.4826 * mad)
    v[bad] = np.nan
    return v


def search_filename(directory: Path, run: int) -> Path:
    prefix = f"{run:02d}"
    for f in directory.glob(f"{prefix}*.hdf5"):
        if f.is_file():
            print(f"Matching file: {f.name}")
            return f
    raise FileNotFoundError(f"No file in {directory} starts with {prefix}")


def process(h5, scope, channel, tbeg_ms, tend_ms, label, ylabel, reducer,
            xx_grid, yy_grid, shot_lookup, fpath):
    signal = f"C{channel}"
    print(f"\n=== {scope} / {signal}  ({label}, t={tbeg_ms}–{tend_ms} ms) ===")

    # --- header + window (always) ------------------------------------------
    cal = parse_lecroy_header(
        h5[f"/{scope}/shot_1/{signal}_header"][()].tobytes())
    DT_MS, T0_MS = cal["dt_ms"], cal["t0_ms"]
    GAIN, OFFSET = cal["gain"], cal["offset"]
    print(f"  dt={DT_MS:.3e} ms/sample, t0={T0_MS:.3f} ms, "
          f"gain={GAIN:.3e} V/count, offset={OFFSET:.3e} V")

    n_total = h5[f"/{scope}/shot_1/{signal}_data"].shape[0]
    i0 = max(0, int(round((tbeg_ms - T0_MS) / DT_MS)))
    i1 = min(n_total, int(round((tend_ms - T0_MS) / DT_MS)))
    if i1 <= i0:
        print(f"  WARNING: window [{tbeg_ms}, {tend_ms}] ms is empty for "
              f"scope range [{T0_MS:.2f}, {T0_MS + n_total * DT_MS:.2f}] ms")
        return
    print(f"  window: samples {i0}:{i1} "
          f"({i1 - i0} samples, {(i1 - i0) * DT_MS:.3f} ms)")

    Nx, Ny = xx_grid.size, yy_grid.size

    # --- diagnostic single shot at (x, y) nearest (0, 0) (always) ----------
    xi0 = int(np.argmin(np.abs(xx_grid)))
    yj0 = int(np.argmin(np.abs(yy_grid)))
    shot0 = int(shot_lookup[(xi0, yj0)][0])
    raw = h5[f"/{scope}/shot_{shot0}/{signal}_data"][i0:i1]
    v_diag = raw.astype(np.float64) * GAIN - OFFSET
    t_diag = T0_MS + (i0 + np.arange(v_diag.size)) * DT_MS

    v_mean    = v_diag.mean()
    v_peak    = 0.5 * (v_diag.max() - v_diag.min())
    cspec     = np.fft.rfft(v_diag - v_mean)
    freqs_g   = np.fft.rfftfreq(v_diag.size, d=DT_MS * 1e-3) / 1e9
    amp_spec  = 2.0 * np.abs(cspec) / v_diag.size
    phase_spec = np.angle(cspec)
    ipk = int(np.argmax(amp_spec))
    ibin_target = int(np.argmin(np.abs(freqs_g - FFT_FREQ_HZ / 1e9)))
    print(f"  sanity @(0,0) shot {shot0}: "
          f"|mean|={abs(v_mean):.3e} V, peak={v_peak:.3e} V")
    print(f"    FFT peak: {amp_spec[ipk]:.3e} V at {freqs_g[ipk]:.4f} GHz, "
          f"phase={phase_spec[ipk]:+.3f} rad")
    print(f"    FFT @ {FFT_FREQ_HZ/1e9:g} GHz "
          f"(bin {ibin_target}): amp={amp_spec[ibin_target]:.3e} V, "
          f"phase={phase_spec[ibin_target]:+.3f} rad "
          f"({np.degrees(phase_spec[ibin_target]):+.1f} deg)")

    # FFT spectrum figure (amplitude + phase)
    fig, (ax_a, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(7.6, 5.2),
                                     constrained_layout=True)
    ax_a.semilogy(freqs_g, amp_spec, color="k", linewidth=0.8)
    ax_a.axvline(FFT_FREQ_HZ / 1e9, color="C3", linestyle="--",
                 linewidth=1.0, label=f"{FFT_FREQ_HZ/1e9:g} GHz")
    ax_a.set_ylabel("|FFT| (V)")
    ax_a.legend(loc="best")

    ax_p.plot(freqs_g, phase_spec, color="k", linewidth=0.5)
    ax_p.axvline(FFT_FREQ_HZ / 1e9, color="C3", linestyle="--",
                 linewidth=1.0)
    ax_p.axhline(0.0, color=[0.7, 0.7, 0.7], linewidth=0.5)
    ax_p.set_xlim(0, freqs_g[-1])
    ax_p.set_ylim(-np.pi, np.pi)
    ax_p.set_xlabel("frequency (GHz)")
    ax_p.set_ylabel("arg(FFT) (rad)")

    fig.suptitle(
        rf"FFT spectrum, shot {shot0}, $(x,y)=({xx_grid[xi0]:g},"
        rf"{yy_grid[yj0]:g})$ cm, $t\in[{tbeg_ms:g},{tend_ms:g}]$ ms")
    base_fft = f"run{RUN}_C{channel}_{label}_xy00_shot{shot0}_fft"
    fig.savefig(FIG_DIR / f"{base_fft}.pdf")
    fig.savefig(FIG_DIR / f"{base_fft}.png", dpi=300)
    plt.close(fig)
    print(f"  saved figures/{base_fft}.pdf / .png")

    # time-domain trace
    fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
    ax.plot(t_diag, v_diag, color="k", linewidth=0.6)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        rf"shot {shot0}, $(x,y)=({xx_grid[xi0]:g},{yy_grid[yj0]:g})$ cm, "
        rf"$t\in[{tbeg_ms:g},{tend_ms:g}]$ ms")
    base_tr = f"run{RUN}_C{channel}_{label}_xy00_shot{shot0}_trace"
    fig.savefig(FIG_DIR / f"{base_tr}.pdf")
    fig.savefig(FIG_DIR / f"{base_tr}.png", dpi=300)
    plt.close(fig)
    print(f"  saved figures/{base_tr}.pdf / .png")

    if TEST_ONLY:
        print(f"  TEST_ONLY=True → skipping plane parse")
        return

    # --- plane parse (or load cached) --------------------------------------
    out = OUT_DIR / f"run{RUN}_C{channel}_{label}_tbeg{tbeg_ms:g}_tend{tend_ms:g}.npz"
    plane_phase = None
    if out.exists() and not FORCE_REPROCESS:
        print(f"  loading cached {out.name}")
        npz = np.load(out, allow_pickle=True)
        plane_avg = npz["plane_avg"]
        plane_err = npz["plane_err"]
        if "plane_phase" in npz.files:
            plane_phase = npz["plane_phase"]
    else:
        if reducer == "mean":
            reduce = lambda v: v.mean()
            plane = np.full((Nx, Ny, NUM_SHOT), np.nan)
        elif reducer == "fft":
            n_win  = i1 - i0
            freqs  = np.fft.rfftfreq(n_win, d=DT_MS * 1e-3)
            df     = freqs[1]
            search_idx = np.where(
                np.abs(freqs - FFT_FREQ_HZ) <= FFT_SEARCH_HALFBW_HZ)[0]
            n_band = int(round(FFT_INTEGRATE_HALFBW_HZ / df))
            print(f"  peak search: {len(search_idx)} bins in "
                  f"[{freqs[search_idx[0]]/1e9:.4f}, "
                  f"{freqs[search_idx[-1]]/1e9:.4f}] GHz")
            print(f"  band integration: ±{n_band} bins "
                  f"(±{n_band * df / 1e3:.1f} kHz)")

            def reduce(v):
                F   = np.fft.rfft(v)
                ipk = search_idx[int(np.argmax(np.abs(F[search_idx])))]
                lo  = max(0, ipk - n_band)
                hi  = min(F.size, ipk + n_band + 1)
                return 2.0 * F[lo:hi].sum() / n_win, freqs[ipk]

            plane = np.full((Nx, Ny, NUM_SHOT), np.nan + 1j * np.nan,
                            dtype=np.complex128)
            plane_fpeak = np.full((Nx, Ny, NUM_SHOT), np.nan)
        else:
            raise ValueError(f"unknown reducer {reducer!r}")

        for xi in range(Nx):
            for yj in range(Ny):
                shots = shot_lookup[(xi, yj)]
                for si in range(min(NUM_SHOT, shots.size)):
                    try:
                        ds  = h5[f"/{scope}/shot_{int(shots[si])}/{signal}_data"]
                    except:
                        print(f"  shot {shots[si]} not found in {scope}")
                        ds = np.nan
                        break
                    v   = ds[i0:i1].astype(np.float64) * GAIN - OFFSET
                    if reducer == "fft":
                        val, fpeak = reduce(v)
                        plane[xi, yj, si]       = val
                        plane_fpeak[xi, yj, si] = fpeak
                    else:
                        plane[xi, yj, si] = reduce(v)
            print(f"  row {xi + 1} / {Nx} done")

        if reducer == "fft":
            f_med = np.nanmedian(plane_fpeak) / 1e9
            f_std = np.nanstd(plane_fpeak)     / 1e6
            print(f"  per-shot peak frequency: median={f_med:.5f} GHz, "
                  f"std={f_std:.2f} MHz across all (x,y,shot)")

        # outlier mask on amplitude; NaN both real & imag for complex case
        amp_for_filter = np.abs(plane) if np.iscomplexobj(plane) else plane
        n_pre = np.sum(~np.isnan(amp_for_filter))
        amp_filtered = mad_filter(amp_for_filter)
        bad_mask = np.isnan(amp_filtered) & ~np.isnan(amp_for_filter)
        if np.iscomplexobj(plane):
            plane[bad_mask] = np.nan + 1j * np.nan
        else:
            plane[bad_mask] = np.nan
        n_post = np.sum(~np.isnan(amp_filtered))
        if n_pre > n_post:
            print(f"  outlier mask: {n_pre - n_post}/{n_pre} shots → NaN "
                  f"(k={OUTLIER_K})")

        if np.iscomplexobj(plane):
            coherent_mean = np.nanmean(plane, axis=2)
            plane_avg   = np.abs(coherent_mean)
            plane_phase = np.angle(coherent_mean)
            plane_err   = np.nanstd(np.abs(plane), axis=2)
        else:
            plane_avg = np.nanmean(plane, axis=2)
            plane_err = np.nanstd(plane, axis=2)

        save_kwargs = dict(
            xx_grid=xx_grid, yy_grid=yy_grid,
            plane_avg=plane_avg, plane_err=plane_err,
            trace_t=t_diag, trace_y=v_diag.astype(np.float32),
            shot_diag=shot0,
            meta=dict(RUN=RUN, CHANNEL=channel, SCOPE=scope, MG_PATH=MG_PATH,
                      DT_MS=DT_MS, T0_MS=T0_MS, GAIN=GAIN, OFFSET=OFFSET,
                      NUM_SHOT=NUM_SHOT, TBEG_MS=tbeg_ms, TEND_MS=tend_ms,
                      label=label, reducer=reducer, OUTLIER_K=OUTLIER_K,
                      FFT_FREQ_HZ=FFT_FREQ_HZ, source=str(fpath)),
        )
        if plane_phase is not None:
            save_kwargs["plane_phase"] = plane_phase
        if reducer == "fft":
            save_kwargs["plane_fpeak"] = plane_fpeak
        np.savez_compressed(out, **save_kwargs)
        print(f"  saved {out.name}")

    # --- plane heatmap -----------------------------------------------------
    base2d = (f"run{RUN}_C{channel}_{label}_{reducer}_"
              f"tbeg{tbeg_ms:g}_tend{tend_ms:g}_plane")
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    extent = (xx_grid.min(), xx_grid.max(), yy_grid.min(), yy_grid.max())
    if reducer == "mean":
        vmax = np.nanmax(np.abs(plane_avg))
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    else:
        kw = dict(cmap="YlGnBu", vmin=0, vmax=np.nanmax(plane_avg))
    im = ax.imshow(plane_avg.T, origin="lower", extent=extent,
                   aspect="equal", interpolation="nearest", **kw)
    ax.set_xlabel(r"$x$ (cm)")
    ax.set_ylabel(r"$y$ (cm)")
    title_tag = (rf"FFT @ {FFT_FREQ_HZ / 1e9:g} GHz"
                 if reducer == "fft" else reducer)
    ax.set_title(rf"$t \in [{tbeg_ms:g}, {tend_ms:g}]$ ms, {title_tag}")
    cb = fig.colorbar(im, ax=ax, shrink=0.95)
    cb.set_label(ylabel)
    cb.outline.set_linewidth(1.2)
    fig.savefig(FIG_DIR / f"{base2d}.pdf")
    fig.savefig(FIG_DIR / f"{base2d}.png", dpi=300)
    plt.close(fig)
    print(f"  saved figures/{base2d}.pdf / .png")

    if reducer == "fft" and plane_phase is not None:
        base_ph = (f"run{RUN}_C{channel}_{label}_fft_"
                   f"tbeg{tbeg_ms:g}_tend{tend_ms:g}_phase")
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        im = ax.imshow(plane_phase.T, origin="lower", extent=extent,
                       aspect="equal", interpolation="nearest",
                       cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_xlabel(r"$x$ (cm)")
        ax.set_ylabel(r"$y$ (cm)")
        ax.set_title(
            rf"$t \in [{tbeg_ms:g}, {tend_ms:g}]$ ms, "
            rf"phase @ {FFT_FREQ_HZ/1e9:g} GHz")
        cb = fig.colorbar(im, ax=ax, shrink=0.95,
                          ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$",
                               r"$\pi/2$", r"$\pi$"])
        cb.set_label("arg(coherent mean) (rad)")
        cb.outline.set_linewidth(1.2)
        fig.savefig(FIG_DIR / f"{base_ph}.pdf")
        fig.savefig(FIG_DIR / f"{base_ph}.png", dpi=300)
        plt.close(fig)
        print(f"  saved figures/{base_ph}.pdf / .png")


def main():
    fpath = search_filename(Path(DATA_DIR), RUN)

    with h5py.File(fpath, "r") as h5:
        pa    = h5[f"/{MG_PATH}/positions_array"][:]
        setup = h5[f"/{MG_PATH}/positions_setup_array"][:]

        xx_grid = np.unique(np.round(setup["x"].astype(float), 1))
        yy_grid = np.unique(np.round(setup["y"].astype(float), 1))
        Nx, Ny  = xx_grid.size, yy_grid.size

        x_snap = np.round(pa["x"].astype(float), 1)
        y_snap = np.round(pa["y"].astype(float), 1)
        shot_lookup = {}
        for i, x in enumerate(xx_grid):
            for j, y in enumerate(yy_grid):
                mask = (x_snap == x) & (y_snap == y)
                shot_lookup[(i, j)] = pa["shot_num"][mask].astype(int)
        print(f"Plane: {Nx}x{Ny}, {shot_lookup[(0, 0)].size} shots/position "
              f"(using first {NUM_SHOT})")

        for scope, channel, tb, te, label, ylabel, reducer in SECTIONS:
            process(h5, scope, channel, tb, te, label, ylabel, reducer,
                    xx_grid, yy_grid, shot_lookup, fpath)


if __name__ == "__main__":
    main()

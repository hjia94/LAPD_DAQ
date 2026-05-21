# -*- coding: utf-8 -*-
"""
Inspect, validate, and plot HDF5 files written by ``Data_Run_bmotion.py``.

Reading/decoding is delegated to ``lab_scopes.io.hdf5``; this module adds a
format validator, console summary, and trace plotting on top.

Setup (once):  pip install -e "C:/.../lab_scopes[hdf5,plot]"
Run:           python -m read_and_analyze.read_bmotion_data <file.hdf5>
See doc/README.md for full usage, the SHOW_PLOT/SAVE_PLOT toggles, and the API.

Created May.2026
@author: Jia Han
"""

import argparse
import json
import os
import sys

import numpy as np

from lab_scopes.io.hdf5 import (
    read_hdf5_scope_data,
    read_hdf5_scope_tarr,
    read_scope_channel_descriptions,
)

# User-changeable knobs live in analysis_config.py (single source of truth);
# imported here under the historical names so the rest of the module is unchanged.
try:  # works as a package (python -m read_and_analyze.read_bmotion_data)
    from read_and_analyze.analysis_config import (
        DATA_FILE as DEFAULT_FILE, SHOW_PLOT, SAVE_PLOT,
    )
except ImportError:  # fallback when run directly from inside the folder
    from analysis_config import (
        DATA_FILE as DEFAULT_FILE, SHOW_PLOT, SAVE_PLOT,
    )

NON_SCOPE_GROUPS = {"Configuration", "Control"}  # root groups that aren't scopes
WAVEDESC_BYTES = 346                              # LeCroy header size
_EXPECTED_POSITION_FIELDS = ("shot_num", "x", "y")


# ======================================================================================
# Small helpers
# ======================================================================================

def _scope_groups(f):
    """Return the list of scope group names in an open HDF5 file."""
    return [name for name, g in f.items()
            if name not in NON_SCOPE_GROUPS and hasattr(g, "keys")
            and any(k.startswith("shot_") for k in g.keys())]


def _shot_numbers(scope_group):
    """Return sorted shot numbers present in a scope group."""
    nums = []
    for k in scope_group.keys():
        if k.startswith("shot_"):
            try:
                nums.append(int(k.split("_", 1)[1]))
            except ValueError:
                pass
    return sorted(nums)


def _channel_names(scope_group, shot_num):
    """Channel names (e.g. ['C1','C2']) recorded for a given shot."""
    shot = scope_group.get(f"shot_{shot_num}")
    if shot is None:
        return []
    return sorted(k[:-5] for k in shot.keys() if k.endswith("_data"))


def _sample_shots(shot_nums, n=3):
    """Pick first, middle, last (deduped) from a sorted shot-number list."""
    if not shot_nums:
        return []
    picks = [shot_nums[0], shot_nums[len(shot_nums) // 2], shot_nums[-1]]
    seen, out = set(), []
    for s in picks[:n]:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ======================================================================================
# Positions  (the one piece lab_scopes does not provide)
# ======================================================================================

def read_positions(f, mg_name=None):
    """Read probe positions from ``/Control/Positions`` of an open HDF5 file.

    With ``mg_name=None`` returns ``{group_name: info}``; otherwise the single
    group's ``info`` dict (keys: name, key, xpos, ypos, setup_array,
    positions_array). Returns ``{}``/``None`` if no position data is present.
    """
    if "Control" not in f or "Positions" not in f["Control"]:
        return {} if mg_name is None else None

    pos_grp = f["Control"]["Positions"]

    def _one(grp):
        setup = grp["positions_setup_array"][()] if "positions_setup_array" in grp else None
        recorded = grp["positions_array"][()] if "positions_array" in grp else None
        xpos = ypos = None
        if setup is not None and "positions_setup_array" in grp:
            attrs = grp["positions_setup_array"].attrs
            xpos = np.asarray(attrs["xpos"]) if "xpos" in attrs else None
            ypos = np.asarray(attrs["ypos"]) if "ypos" in attrs else None
        return {
            "name": grp.attrs.get("name"),
            "key": grp.attrs.get("key"),
            "xpos": xpos,
            "ypos": ypos,
            "setup_array": setup,
            "positions_array": recorded,
        }

    if mg_name is not None:
        if mg_name not in pos_grp:
            return None
        return _one(pos_grp[mg_name])

    return {name: _one(pos_grp[name]) for name in pos_grp.keys()}


def _position_for_shot(positions, shot_num):
    """Return (x, y) recorded for a shot, searching all motion groups. None if absent."""
    if not positions:
        return None
    for info in positions.values():
        arr = info.get("positions_array")
        if arr is None:
            continue
        idx = shot_num - 1  # array is 0-based: index = shot_num - 1
        if 0 <= idx < len(arr) and int(arr["shot_num"][idx]) == shot_num:
            return float(arr["x"][idx]), float(arr["y"][idx])
    return None


# ======================================================================================
# Validation
# ======================================================================================

def validate_file(path):
    """Validate a bmotion HDF5 file's structure.

    Returns ``(ok, report)``: ``ok`` is False if any check FAILed; ``report`` is
    a list of lines each tagged ``[PASS]`` / ``[WARN]`` / ``[FAIL]``.
    """
    import h5py

    report = []
    state = {"fail": False}

    def add(level, msg):
        report.append(f"[{level}] {msg}")
        if level == "FAIL":
            state["fail"] = True

    if not os.path.exists(path):
        return False, [f"[FAIL] File not found: {path}"]

    with h5py.File(path, "r") as f:
        # --- root attributes ---
        for a in ("description", "creation_time", "source_code"):
            if a in f.attrs:
                add("PASS", f"root attr '{a}' present")
            else:
                add("WARN", f"root attr '{a}' missing")

        # --- Configuration ---
        if "Configuration" in f:
            cfg = f["Configuration"]
            for ds in ("experiment_config", "bmotion_config", "bmotion_selection"):
                if ds in cfg:
                    add("PASS", f"/Configuration/{ds} present")
                else:
                    add("WARN", f"/Configuration/{ds} missing")
            if "bmotion_selection" in cfg:
                try:
                    json.loads(np.asarray(cfg["bmotion_selection"][()]).tobytes().decode())
                    add("PASS", "bmotion_selection parses as JSON")
                except Exception as e:
                    add("FAIL", f"bmotion_selection is not valid JSON: {e}")
        else:
            add("WARN", "/Configuration group missing")

        # --- Control/Positions ---
        positions = read_positions(f)
        if positions:
            for name, info in positions.items():
                setup = info["setup_array"]
                recorded = info["positions_array"]
                if setup is None:
                    add("FAIL", f"motion group '{name}' missing positions_setup_array")
                else:
                    fields = tuple(setup.dtype.names or ())
                    if fields == _EXPECTED_POSITION_FIELDS:
                        add("PASS", f"motion group '{name}' setup dtype OK {fields}")
                    else:
                        add("FAIL", f"motion group '{name}' unexpected setup dtype fields {fields}")
                if info["xpos"] is None or info["ypos"] is None:
                    add("WARN", f"motion group '{name}' missing xpos/ypos attrs")
                else:
                    add("PASS", f"motion group '{name}' grid {len(info['xpos'])} x-pos, "
                                f"{len(info['ypos'])} y-pos")
                if recorded is None:
                    add("FAIL", f"motion group '{name}' missing positions_array")
                else:
                    _validate_positions_array(name, recorded, info, add)
        else:
            add("WARN", "no /Control/Positions data (not a bmotion file?)")

        # --- scopes ---
        scopes = _scope_groups(f)
        if not scopes:
            add("FAIL", "no scope groups found")
        for scope in scopes:
            _validate_scope(f, scope, positions, add)

    return (not state["fail"]), report


def _validate_positions_array(name, recorded, info, add):
    """Check positions_array: population, monotonic shot_num, in-grid."""
    shot_nums = recorded["shot_num"]
    n_zero = int(np.count_nonzero(shot_nums == 0))
    if n_zero == 0:
        add("PASS", f"'{name}' positions_array fully populated ({len(recorded)} shots)")
    else:
        add("WARN", f"'{name}' positions_array has {n_zero} unset entries "
                    f"(skipped shots?) of {len(recorded)}")

    nonzero = shot_nums[shot_nums != 0]
    if nonzero.size and np.all(np.diff(nonzero.astype(np.int64)) >= 0):
        add("PASS", f"'{name}' shot_num monotonic non-decreasing")
    else:
        add("WARN", f"'{name}' shot_num not monotonic")

    xpos, ypos = info.get("xpos"), info.get("ypos")
    if xpos is not None and ypos is not None and nonzero.size:
        mask = shot_nums != 0
        x, y = recorded["x"][mask], recorded["y"][mask]
        tol = 1.0  # mm; encoder readings drift slightly off the nominal grid
        x_ok = np.all((x >= xpos.min() - tol) & (x <= xpos.max() + tol))
        y_ok = np.all((y >= ypos.min() - tol) & (y <= ypos.max() + tol))
        if x_ok and y_ok:
            add("PASS", f"'{name}' recorded positions within grid bounds")
        else:
            add("WARN", f"'{name}' some recorded positions outside grid bounds "
                        f"(x {x.min():.2f}..{x.max():.2f}, y {y.min():.2f}..{y.max():.2f})")


def _validate_scope(f, scope, positions, add):
    """Validate one scope: time_array, shots, and a sample of decoded traces."""
    scope_group = f[scope]

    if "time_array" in scope_group:
        add("PASS", f"/{scope}/time_array present ({scope_group['time_array'].shape[0]} samples)")
    else:
        add("WARN", f"/{scope}/time_array missing (will reconstruct from header)")

    shot_nums = _shot_numbers(scope_group)
    if not shot_nums:
        add("FAIL", f"/{scope} has no shot_* groups")
        return
    add("PASS", f"/{scope} has {len(shot_nums)} shots (shot_{shot_nums[0]}..shot_{shot_nums[-1]})")

    try:
        tarr = read_hdf5_scope_tarr(f, scope)
    except Exception:
        tarr = None

    # Sample first/middle/last + the first skipped shot.
    sample = list(_sample_shots(shot_nums))
    skipped = next((s for s in shot_nums
                    if scope_group[f"shot_{s}"].attrs.get("skipped", False)), None)
    if skipped is not None and skipped not in sample:
        sample.append(skipped)

    for s in sample:
        shot = scope_group[f"shot_{s}"]
        if shot.attrs.get("skipped", False):
            add("PASS", f"/{scope}/shot_{s} marked skipped: "
                        f"{shot.attrs.get('skip_reason', 'no reason')}")
            continue
        for ch in _channel_names(scope_group, s):
            _validate_trace(f, scope, ch, s, shot, tarr, add)


def _validate_trace(f, scope, ch, s, shot, tarr, add):
    """Validate one channel/shot: header size, dtype, decode, length, finite."""
    data_key, hdr_key = f"{ch}_data", f"{ch}_header"

    if hdr_key not in shot:
        add("FAIL", f"/{scope}/shot_{s}/{hdr_key} missing")
        return
    hdr = shot[hdr_key]
    if hdr.dtype.itemsize != WAVEDESC_BYTES:
        add("WARN", f"/{scope}/shot_{s}/{hdr_key} is {hdr.dtype.itemsize} bytes "
                    f"(expected {WAVEDESC_BYTES})")

    if shot[data_key].dtype != np.int16:
        add("WARN", f"/{scope}/shot_{s}/{data_key} dtype {shot[data_key].dtype} (expected int16)")

    try:
        volts, dt, t0 = read_hdf5_scope_data(f, scope, ch, s)
    except Exception as e:
        add("FAIL", f"/{scope}/shot_{s}/{ch} failed to decode: {e}")
        return

    if not np.all(np.isfinite(volts)):
        add("WARN", f"/{scope}/shot_{s}/{ch} contains non-finite voltages")

    if tarr is not None and len(tarr) != len(volts):
        add("WARN", f"/{scope}/shot_{s}/{ch} time_array len {len(tarr)} "
                    f"!= trace len {len(volts)} (reader will reconstruct)")
    else:
        add("PASS", f"/{scope}/shot_{s}/{ch} decoded OK "
                    f"(n={len(volts)}, dt={dt:.3g}s, V {volts.min():.4g}..{volts.max():.4g})")


# ======================================================================================
# Summary
# ======================================================================================

def print_summary(path):
    """Print a quick console overview of the file's contents."""
    import h5py

    print("=" * 70)
    print(f"FILE: {path}")
    size_mb = os.path.getsize(path) / 1e6
    print(f"Size: {size_mb:.1f} MB")

    with h5py.File(path, "r") as f:
        desc = f.attrs.get("description")
        if desc:
            first_line = str(desc).strip().splitlines()[0]
            print(f"Description: {first_line}")
        print(f"Created: {f.attrs.get('creation_time')}")

        for scope in _scope_groups(f):
            sg = f[scope]
            shots = _shot_numbers(sg)
            print(f"\nScope '{scope}'  ({sg.attrs.get('scope_type', '').strip()})")
            print(f"  ip={sg.attrs.get('ip_address')}  description={sg.attrs.get('description')}")
            print(f"  shots: {len(shots)}  (shot_{shots[0]}..shot_{shots[-1]})")
            ta = sg.get("time_array")
            if ta is not None and ta.shape[0] > 1:
                t = ta[()]
                print(f"  time_array: {ta.shape[0]} samples, "
                      f"{t[0] * 1e3:.3f}..{t[-1] * 1e3:.3f} ms, dt={ (t[1]-t[0])*1e6:.4g} us")
            for ch, d in read_scope_channel_descriptions(f, scope).items():
                print(f"    {ch}: {d}")

        positions = read_positions(f)
        for name, info in positions.items():
            xpos, ypos = info["xpos"], info["ypos"]
            rec = info["positions_array"]
            nshots = 0 if rec is None else int(np.count_nonzero(rec["shot_num"] != 0))
            npos = (len(xpos) if xpos is not None else 0) * (len(ypos) if ypos is not None else 0)
            print(f"\nMotion group '{name}' (key={info['key']})")
            if xpos is not None:
                print(f"  x: {len(xpos)} positions  {xpos.min():.1f}..{xpos.max():.1f}")
            if ypos is not None:
                print(f"  y: {len(ypos)} positions  {ypos.min():.1f}..{ypos.max():.1f}")
            if npos:
                print(f"  grid positions: {npos}   recorded shots: {nshots}   "
                      f"(~{nshots / npos:.1f} shots/position)")
    print("=" * 70)


# ======================================================================================
# Plotting
# ======================================================================================

def plot_traces(path, scope=None, channels=None, shots=None, show=None, save=None):
    """Overlay a few traces per scope for visual comparison against the scope.

    scope/channels/shots default to all-scopes / all-channels / first-middle-last.
    show/save default to module-level SHOW_PLOT/SAVE_PLOT; saved PNGs go in a
    ``plots/`` subdir next to the data file (one per scope). Returns saved paths.
    """
    import h5py
    import matplotlib.pyplot as plt

    if show is None:
        show = SHOW_PLOT
    if save is None:
        save = SAVE_PLOT

    saved = []
    if save:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "plots")
        os.makedirs(plots_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    with h5py.File(path, "r") as f:
        scopes = [scope] if scope else _scope_groups(f)
        positions = read_positions(f)

        for sc in scopes:
            sg = f[sc]
            shot_nums = _shot_numbers(sg)
            use_shots = shots if shots else _sample_shots(shot_nums)
            use_shots = [s for s in use_shots
                         if not sg[f"shot_{s}"].attrs.get("skipped", False)]
            if not use_shots:
                print(f"Scope '{sc}': no plottable (non-skipped) shots")
                continue
            use_channels = channels if channels else _channel_names(sg, use_shots[0])
            descs = read_scope_channel_descriptions(f, sc)

            fig, axes = plt.subplots(len(use_channels), 1, sharex=True,
                                     figsize=(10, 2.4 * len(use_channels)), squeeze=False)
            axes = axes[:, 0]
            for ax, ch in zip(axes, use_channels):
                for s in use_shots:
                    try:
                        volts, dt, t0 = read_hdf5_scope_data(f, sc, ch, s)
                    except Exception as e:
                        print(f"  skip {sc}/shot_{s}/{ch}: {e}")
                        continue
                    try:
                        tarr = read_hdf5_scope_tarr(f, sc)
                        if len(tarr) != len(volts):
                            tarr = np.arange(len(volts)) * dt + t0
                    except Exception:
                        tarr = np.arange(len(volts)) * dt + t0
                    pos = _position_for_shot(positions, s)
                    label = f"shot {s}"
                    if pos is not None:
                        label += f" @ x={pos[0]:.1f}, y={pos[1]:.1f}"
                    ax.plot(tarr * 1e3, volts, lw=0.8, label=label)
                ax.set_ylabel("V")
                ax.set_title(f"{ch}: {descs.get(ch, '')}", fontsize=9, loc="left")
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("time (ms)")
            fig.suptitle(f"{os.path.basename(path)}  —  scope '{sc}'", fontsize=10)
            fig.tight_layout()

            if save:
                out_png = os.path.join(plots_dir, f"{base}_{sc}.png")
                fig.savefig(out_png, dpi=150)
                saved.append(out_png)
                print(f"Saved plot: {out_png}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved


# ======================================================================================
# CLI
# ======================================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Inspect/validate a Data_Run_bmotion.py HDF5 file and plot traces.")
    p.add_argument("path", nargs="?", default=DEFAULT_FILE,
                   help=f"HDF5 file (default: {DEFAULT_FILE})")
    # --no-show/--no-save override SHOW_PLOT/SAVE_PLOT for a single run.
    p.add_argument("--no-show", action="store_true",
                   help=f"do not display plots (overrides SHOW_PLOT={SHOW_PLOT})")
    p.add_argument("--no-save", action="store_true",
                   help=f"do not save plots (overrides SAVE_PLOT={SAVE_PLOT})")
    p.add_argument("--scope", default=None, help="scope group to plot (default: all)")
    p.add_argument("--channels", nargs="+", default=None, help="channels to plot (default: all)")
    p.add_argument("--shots", nargs="+", type=int, default=None,
                   help="shot numbers to overlay (default: first/middle/last)")
    args = p.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        return 2

    print_summary(args.path)

    print("\nVALIDATION")
    print("-" * 70)
    ok, report = validate_file(args.path)
    for line in report:
        print(line)
    print("-" * 70)
    print("RESULT:", "OK (no failures)" if ok else "FAILURES PRESENT")

    show = False if args.no_show else SHOW_PLOT
    save = False if args.no_save else SAVE_PLOT
    if show or save:
        plot_traces(args.path, scope=args.scope, channels=args.channels,
                    shots=args.shots, show=show, save=save)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

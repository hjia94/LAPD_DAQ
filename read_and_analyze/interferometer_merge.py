# coding utf-8
'''
Automatic interferometer merge for LAPD_DAQ-format datarun hdf5 files.

Runs at the END of offload (Offload_Run.py), once all spool data has been
written into the datarun file, for any run acquired through the LAPD_DAQ repo --
Data_Run.py (grid / stationary) and Data_Run_bmotion.py alike: both write shots
through the same acquisition/hdf5_writer code, so the per-scope layout read here
is identical.

This module is fully SELF-CONTAINED: it needs only the Python stdlib, h5py,
numpy, and the vendored ``scope_io.wavedesc.LeCroyWavedesc``. It does NOT import
``lab_scopes`` or ``bapsf_interferometer`` -- LAPD_DAQ only needs a path to the
folder holding ``interferometer_data_<YYYY-MM-DD>.hdf5`` (see
``INTERFEROMETER_DATA_DIR`` below). It was ported from
``bapsf_interferometer/interf_merge_lapd_daq.py`` with its shared helpers (from
``interf_merge_datarun.py``) inlined here.

Author: Jia Han
Created: 2026-07-06 (ported into LAPD_DAQ 2026-07-21)

What is merged: TWO traces per interferometer channel. Among the interferometer
traces collected strictly WITHIN the run's duration (first shot trigger .. last
shot trigger), the one closest to the first shot is saved as shot_1 and the one
closest to the last shot as the last shot number. This guarantees the merged
traces were acquired while the run was in progress without requiring the two
clocks to agree tightly, and is enough to sanity-check plasma density at the
start and end of a run.

If no trace falls inside the run window, an unattended merge (interactive=False,
as called by the offload) merges nothing and reports it; a manual call may pass
pad_seconds to widen the window, or (interactive=True) get a terminal prompt.
The interferometer time actually saved for each shot is recorded as attributes
on every merged dataset and printed to the terminal.

TODO (future, if needed): shot-to-shot matching for every shot. The building
block get_shot_timestamps() already returns every shot's trigger time; the
robust clock-offset estimator (estimate_clock_offset / _nearest_delta) that
would drive a tight per-shot tolerance was left in
bapsf_interferometer/interf_merge_lapd_daq.py and can be restored from there.
It is omitted here because the offset is only identifiable modulo the shot
period (both systems trigger off the same periodic LAPD edge), so per-shot
pairing is ambiguous when the clocks are badly out of sync.

The new acquisition (LAPD_DAQ repo, spooled) writes:

	/<scope_name>/                 attrs: description, ip_address, scope_type
		time_array                 float64 seconds
		shot_<n>/                  attrs: acquisition_time [, skipped, skip_reason]
			C<k>_data              int16, (N,) or (n_segments, N) in sequence mode
			C<k>_header            346-byte LeCroy WAVEDESC (np.void)

Per-shot timestamps for matching:
- Primary: the trigger time embedded in each stored C*_header WAVEDESC
  (tt_year..tt_second fields). Stamped by the scope at the actual trigger, with
  sub-second resolution, immune to how far behind the offload process was.
- Fallback: the shot group's 'acquisition_time' attr (ctime string, 1 s
  resolution). In runs written before the spool acquire-time fix this can be the
  offload *write* time, so it is used only when the WAVEDESC cannot be decoded
  (e.g. skipped shots have no header).

The scope RTC fields are wall-clock local time, so they are converted with
time.mktime (local), matching the interferometer dataset names which are seconds
since epoch from the interferometer machine's clock.

How to access the merged data (identical to interf_merge_datarun.py output):
merged traces land under diagnostics/interferometer/<group>/<shot_number>. Which
interferometer times were saved is noted on every merged dataset
('interferometer timestamp (s since epoch)', 'interferometer time (local)',
'time difference from shot trigger (s)') and on diagnostics/interferometer
('timestamp source', 'reference scope', 'run window (s since epoch)',
'window pad applied (s)', 'merged shot numbers',
'merged interferometer timestamps (s since epoch)').
'''

import os
import re
import sys
import bisect
import contextlib
import datetime
import time

import h5py
import numpy as np

# Allow running directly (IDE "Run" button, ``python interferometer_merge.py``
# from inside this folder) as well as ``python -m
# read_and_analyze.interferometer_merge`` from the repo root. The root-level
# ``scope_io`` package is only importable when the repo root is on sys.path;
# ``-m`` adds it but a direct script run does not, so put it there ourselves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Vendored (byte-identical) WAVEDESC decoder -- LAPD_DAQ ships this so readers
# work without lab_scopes installed. (bapsf's interf_merge_lapd_daq.py imports
# the same class from lab_scopes.lecroy.) scope_shot_numbers is the shared
# shot_<n> parser so we don't re-derive it; scope_io depends only on numpy+h5py.
from scope_io import scope_shot_numbers as _shot_numbers
from scope_io.wavedesc import LeCroyWavedesc


#===============================================================================================================================================
# Configuration
#===============================================================================================================================================

# Folder on the DAQ machine holding interferometer_data_YYYY-MM-DD.hdf5 files
# (network share written by the interferometer PC). Edit when the folder moves.
# The automatic end-of-offload merge (run_interferometer_merge) reads from here;
# it SKIPs cleanly if this folder does not exist.
INTERFEROMETER_DATA_DIR = r"N:\interferometer"   # <-- set to the real path

# The interferometer channel groups. phase_p20 is the canonical reader index in
# newer files (written last, so partial shots are skipped); the others fall back
# for older formats.
DATA_GROUPS = ("phase_p20", "phase_p29", "phase_p40", "time_array", "time_array_p40")


#===============================================================================================================================================
# New-format datarun reading
#===============================================================================================================================================

# Root groups that are never scope groups. 'diagnostics' is where this script
# writes its own output, so it must be excluded when re-running on a file. Kept
# local here rather than imported from read_and_analyze.read_bmotion_data: that
# module's NON_SCOPE_GROUPS omits 'diagnostics', and importing it would pull
# plotting dependencies into the offload path.
NON_SCOPE_GROUPS = {"Configuration", "Control", "diagnostics"}

_CTIME_FMT = "%a %b %d %H:%M:%S %Y"  # what time.ctime() produces


def _scope_groups(f):
	'''Return the datarun's scope group names (root groups with shot_* children).'''
	names = []
	for name, g in f.items():
		if name in NON_SCOPE_GROUPS or not hasattr(g, 'keys'):
			continue
		if _shot_numbers(g):
			names.append(name)
	return names


def _wavedesc_epoch_local(wd):
	'''
	Convert a WAVEDESC's trigger-time fields to seconds since epoch.

	The scope RTC runs on wall-clock local time, so the fields are interpreted
	with time.mktime (local timezone, DST resolved automatically). This keeps
	the value directly comparable to the interferometer dataset names (epoch
	seconds), leaving only genuine clock mis-sync as the difference.
	(lab_scopes' wavedesc_trigger_timestamp uses UTC on purpose -- it is meant
	only for same-shot differences between scopes -- so it is not used here.)

	Returns float epoch seconds, or None if the timestamp fields are unset.
	'''
	try:
		if int(wd.tt_year) <= 0:
			return None
		sec = float(wd.tt_second)
		whole = int(sec)
		frac = sec - whole
		t = time.mktime((int(wd.tt_year), int(wd.tt_months), int(wd.tt_days),
		                 int(wd.tt_hours), int(wd.tt_minute), whole, 0, 0, -1))
		return float(t) + frac
	except Exception:
		return None


def _shot_trigger_timestamp(shot_group):
	'''
	Best-effort timestamp for one shot_<n> group.

	Returns (timestamp, source) where source is 'wavedesc' or 'acquisition_time',
	or (None, None) if neither is available. Skipped shots carry no header, so
	they land on the acquisition_time fallback (and are flagged as such).
	'''
	for key in sorted(shot_group.keys()):
		if not key.endswith('_header'):
			continue
		try:
			raw = bytes(shot_group[key][()])
			ts = _wavedesc_epoch_local(LeCroyWavedesc(raw).wd)
		except Exception:
			ts = None
		if ts is not None:
			return ts, 'wavedesc'

	ctime_str = shot_group.attrs.get('acquisition_time')
	if ctime_str is not None:
		if isinstance(ctime_str, bytes):
			ctime_str = ctime_str.decode('utf-8', 'replace')
		try:
			return time.mktime(time.strptime(str(ctime_str), _CTIME_FMT)), 'acquisition_time'
		except ValueError:
			pass
	return None, None


def get_shot_timestamps(datarun_path, verbose=True):
	'''
	Get shot numbers and trigger timestamps from a LAPD_DAQ-format datarun file.

	One scope is used as the timestamp reference: the first scope (file order)
	for which at least one WAVEDESC decodes. All scopes trigger off the same
	edge, so any scope's trigger times identify the plasma shots.

	Skipped shots (attrs['skipped']) are excluded -- they have no scope data and
	their acquisition_time may be an offload-lagged stamp.

	Parameters:
	datarun_path (str): Path to the datarun hdf5 file.
	verbose (bool): Print the reference scope and shot count.

	Returns:
	(numpy.ndarray, numpy.ndarray, list, str): parallel arrays of shot numbers
		(int) and timestamps (epoch seconds, float), the per-shot timestamp
		source ('wavedesc' or 'acquisition_time'), and the reference scope name.
	'''
	with h5py.File(datarun_path, 'r') as f:
		scopes = _scope_groups(f)
		if not scopes:
			raise ValueError(f"No scope groups with shot_* found in {datarun_path} "
			                 "(is this an old-format datarun? use interf_merge_datarun.py)")

		best = None  # (shots, timestamps, sources, scope_name)
		for scope_name in scopes:
			sg = f[scope_name]
			shots, stamps, sources = [], [], []
			for n in _shot_numbers(sg):
				shot = sg[f'shot_{n}']
				if shot.attrs.get('skipped', False):
					continue
				ts, source = _shot_trigger_timestamp(shot)
				if ts is None:
					continue
				shots.append(n)
				stamps.append(ts)
				sources.append(source)
			if not shots:
				continue
			if 'wavedesc' in sources:
				best = (shots, stamps, sources, scope_name)
				break
			if best is None:
				best = (shots, stamps, sources, scope_name)

	if best is None:
		raise ValueError(f"No usable shot timestamps in {datarun_path}")

	shots, stamps, sources, scope_name = best
	if verbose:
		n_fallback = sum(1 for s in sources if s != 'wavedesc')
		print(f"Timestamps from scope '{scope_name}': {len(shots)} shots "
		      f"({len(shots) - n_fallback} WAVEDESC, {n_fallback} acquisition_time fallback)")
		if n_fallback:
			print("Warning: acquisition_time-fallback shots may carry the offload write "
			      "time (pre-fix runs) and can fail to match.")
	return (np.array(shots, dtype=int), np.array(stamps, dtype=float),
	        sources, scope_name)


#===============================================================================================================================================
# Interferometer-file helpers (inlined from interf_merge_datarun.py; stdlib +
# h5py + numpy only -- no read_hdf5 / interf_raw dependency)
#===============================================================================================================================================

def init_datarun_groups(datarun_path, interf_path, verbose=True):
	'''
	Initialize the interferometer groups in the datarun file.

	Creates groups for whichever interferometer datasets are present in the
	source file(s), so old files (phase_p20, phase_p29, time_array only) and
	new files (with phase_p40 and time_array_p40) both work. If multiple
	interferometer paths are given, the union of groups across all files is
	created so a multi-day run picks up channels that appear in any candidate.

	Parameters:
	datarun_path (str): The path to the datarun hdf5 file.
	interf_path (str | list[str]): The path to one or more interferometer
		hdf5 files.
	verbose (bool): If True (default), print a completion line.
	'''
	if isinstance(interf_path, (str, os.PathLike)):
		interf_paths = [interf_path]
	else:
		interf_paths = list(interf_path)

	with h5py.File(datarun_path, "a") as f_datarun:
		parent = f_datarun.require_group("diagnostics/interferometer")
		for p in interf_paths:
			with h5py.File(p, "r") as f_interf:
				if 'description' in f_interf.attrs and 'description' not in parent.attrs:
					parent.attrs['description'] = f_interf.attrs['description']

				for name in DATA_GROUPS:
					if name not in f_interf:
						continue
					sub = f_datarun.require_group(f"diagnostics/interferometer/{name}")
					for attr_name, attr_value in f_interf[name].attrs.items():
						if attr_name not in sub.attrs:
							sub.attrs[attr_name] = attr_value

	if verbose:
		print('Interferometer groups created/loaded in datarun file')


def _normalize_interf_paths(interf_path):
	'''Normalize a single path or a list of paths to a non-empty list.'''
	if isinstance(interf_path, (str, os.PathLike)):
		return [interf_path]
	interf_paths = list(interf_path)
	if not interf_paths:
		raise ValueError("interf_path must be a non-empty path or list of paths")
	return interf_paths


def _open_interf_index(stack, interf_paths):
	'''Open all candidate interferometer files and build one unified index.

	Files are opened read-only on the caller's ExitStack. The unified index is
	sorted by timestamp so a per-shot lookup is a single O(log N) bisect
	regardless of how many candidate files were supplied.

	Returns (f_interfs, all_pairs, sorted_floats, per_file_groups) where
	all_pairs is [(float_timestamp, dataset_name, file_idx)] sorted by time,
	sorted_floats are the timestamps alone, and per_file_groups[i] is the set
	of DATA_GROUPS present in file i (so per-shot reads know which channels to
	look up in which file).
	'''
	f_interfs = [stack.enter_context(h5py.File(p, "r")) for p in interf_paths]

	all_pairs = []
	per_file_groups = []
	for file_idx, f_interf in enumerate(f_interfs):
		# phase_p20 is the canonical reader index in newer files (it is
		# written last so partial shots are skipped). Fall back to
		# whichever known group exists for older formats.
		index_group = next((g for g in DATA_GROUPS if g in f_interf), None)
		if index_group is None:
			raise ValueError(f"No interferometer groups found in {interf_paths[file_idx]}")
		per_file_groups.append(set(g for g in DATA_GROUPS if g in f_interf))
		for name in f_interf[index_group].keys():
			all_pairs.append((float(name), name, file_idx))

	all_pairs.sort(key=lambda p: p[0])
	sorted_floats = [p[0] for p in all_pairs]
	return f_interfs, all_pairs, sorted_floats, per_file_groups


def _available_groups(datarun_path, per_file_groups):
	'''Groups available for writing = groups present in the datarun file AND
	present in at least one interferometer file.'''
	with h5py.File(datarun_path, "r") as f_datarun:
		datarun_groups = set(f_datarun.get("diagnostics/interferometer", {}).keys())
	union_interf_groups = set().union(*per_file_groups) if per_file_groups else set()
	return [g for g in DATA_GROUPS
	        if g in union_interf_groups and g in datarun_groups]


def _copy_shot_datasets(datarun_path, f_interf, groups_in_this_file,
                        available_groups, matching_set, shot_n,
                        extra_attrs=None):
	'''Copy one interferometer trace set into the datarun as shot ``shot_n``.

	Reads each available group's ``matching_set`` dataset from the open source
	file and writes it under diagnostics/interferometer/<group>/<shot_n>,
	copying the source attributes (plus ``extra_attrs``, if given). Datasets
	that already exist are left untouched, so re-runs are idempotent. The
	datarun file is opened just for this shot so an interruption mid-run
	leaves already-merged shots safely flushed to disk.

	Returns True if any dataset was written.
	'''
	shot_data = {}
	shot_attrs = {}
	for g in available_groups:
		# Skip groups not present in this particular source file
		# (e.g. older files without phase_p40), and shots where
		# phase_p40 is legitimately absent for that file.
		if g not in groups_in_this_file or matching_set not in f_interf[g]:
			continue
		ds = f_interf[g][matching_set]
		shot_data[g] = ds[:]
		shot_attrs[g] = dict(ds.attrs)

	wrote_any = False
	with h5py.File(datarun_path, "a") as f_datarun:
		for g, data in shot_data.items():
			dest = f_datarun[f"diagnostics/interferometer/{g}"]
			if shot_n in dest:
				continue
			new_ds = dest.create_dataset(shot_n, data=data)
			for attr_name, attr_value in shot_attrs[g].items():
				new_ds.attrs[attr_name] = attr_value
			if extra_attrs:
				for attr_name, attr_value in extra_attrs.items():
					new_ds.attrs[attr_name] = attr_value
			wrote_any = True
		if wrote_any:
			# Push h5py library buffers to the kernel, then ask the
			# kernel to push its page cache to disk. Narrows the
			# window where an OS crash or power loss could lose the
			# just-written shot.
			f_datarun.flush()
			try:
				os.fsync(f_datarun.id.get_vfd_handle())
			except (OSError, AttributeError):
				# Some VFDs don't expose a raw fd; flush() alone has
				# already done what it can.
				pass
	return wrote_any


_DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def _interf_filename_for_date(date_str):
	return f"interferometer_data_{date_str}.hdf5"


def _datarun_date(datarun_path):
	'''
	Resolve the datarun's date as a (date, source) tuple where source is
	"filename" or "ctime". The date is parsed from the filename if possible;
	otherwise it falls back to the file's creation time. On Windows
	os.path.getctime returns actual creation time. Returns (None, None) if
	the file is missing.
	'''
	m = _DATE_RE.search(os.path.basename(datarun_path))
	if m is not None:
		return datetime.datetime.strptime(m.group(1), "%Y-%m-%d"), "filename"
	try:
		ctime = os.path.getctime(datarun_path)
	except OSError:
		return None, None
	return datetime.datetime.fromtimestamp(ctime), "ctime"


def _candidate_interf_files(datarun_path, interf_dir):
	'''
	Return (candidate_paths, date_source) for a datarun file. The date is
	parsed from the filename when possible, or falls back to the file's
	creation time. Candidates are ordered same-day, prev-day, next-day; only
	paths that exist on disk are included. If no date can be resolved at all,
	date_source is None.
	'''
	date, source = _datarun_date(datarun_path)
	if date is None:
		return [], None

	candidates = [date.strftime("%Y-%m-%d"),
	              (date - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
	              (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")]
	paths = [os.path.join(interf_dir, _interf_filename_for_date(d)) for d in candidates]
	return [p for p in paths if os.path.isfile(p)], source


#===============================================================================================================================================
# Merge (first + last shot only, traces selected within the run window)
#===============================================================================================================================================

def _fmt_local(ts):
	'''Epoch seconds -> "YYYY-MM-DD HH:MM:SS" local wall time.'''
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _prompt_pad(default_pad):
	'''
	Terminal prompt asking how to resolve an empty run window.

	The user must choose: continue with the default pad, continue with a
	custom pad (type a number of seconds), or stop here (merge nothing).
	Re-asks on unrecognized input. Returns the pad in seconds, or None to
	stop. If no terminal input is available (piped/unattended), stops.
	'''
	print("Resolution required:")
	print(f"  c          continue: retry with the window padded by +/-{default_pad:.0f} s")
	print("  <number>   continue: retry with a custom pad (seconds per side)")
	print("  s          stop here: merge nothing for this datarun")
	while True:
		try:
			ans = input("Continue with pads or stop here? [c / <number> / s]: ").strip().lower()
		except (EOFError, OSError):
			print("No terminal input available; stopping here (nothing merged).")
			return None
		if ans in ('s', 'stop', 'n', 'no', 'q', 'quit'):
			return None
		if ans in ('c', 'continue', 'y', 'yes'):
			return float(default_pad)
		try:
			return float(ans)
		except ValueError:
			print(f"Unrecognized answer {ans!r} -- type 'c', a number of seconds, or 's'.")


def merge_interferometer_data(datarun_path, interf_path,
                              pad_seconds=None, interactive=True,
                              verbose=True):
	'''
	Merge interferometer traces for the first and last shots of a LAPD_DAQ run.

	Call init_datarun_groups(datarun_path, interf_path) first (it only creates
	groups and copies attributes).

	Selection rule: only interferometer traces acquired strictly within the
	run's duration (first shot trigger .. last shot trigger) are considered.
	Among them, the trace closest in time to the first shot is saved under the
	first shot's number and the trace closest to the last shot under the last
	shot's number. No timestamp tolerance is required, so ordinary clock
	mis-sync cannot make the merge come back empty as long as the two systems
	overlapped in time. If both selections land on the same trace (e.g. a
	single candidate), it is written once, under the shot it is closer to.

	If the strict window contains no trace:
	- with pad_seconds given, the window is widened by that many seconds on
	  each side and the selection retried once;
	- otherwise, when `interactive`, a terminal prompt asks for a resolution;
	- otherwise the merge returns 0.

	The interferometer time saved for each shot is printed and recorded as
	dataset attributes ('interferometer timestamp (s since epoch)',
	'interferometer time (local)', 'time difference from shot trigger (s)').

	Parameters:
	datarun_path (str): Path to the datarun hdf5 file.
	interf_path (str | list[str]): One or more interferometer hdf5 files (a run
		spanning midnight passes both days' files).
	pad_seconds (float | None): Widen the run window by this many seconds per
		side if the strict window is empty. None = strict only (prompt if
		interactive).
	interactive (bool): Allow the terminal prompt on strict-window failure.
		Set False for unattended/batch use.
	verbose (bool): Progress printing (the chosen interferometer times are
		printed regardless, unless verbose is False).

	Returns:
	int: Number of shots whose data was written (0, 1, or 2).
	'''
	def _log(msg):
		if verbose:
			print(msg)

	shot_numbers, timestamp_array, sources, ref_scope = \
		get_shot_timestamps(datarun_path, verbose=verbose)

	t_first = float(timestamp_array[0])
	t_last = float(timestamp_array[-1])
	shot_first = int(shot_numbers[0])
	shot_last = int(shot_numbers[-1])

	interf_paths = _normalize_interf_paths(interf_path)

	shots_written = 0
	with contextlib.ExitStack() as stack:
		f_interfs, all_pairs, sorted_floats, per_file_groups = \
			_open_interf_index(stack, interf_paths)

		def window_selection(pad):
			'''Indices into all_pairs of the traces (within the padded window)
			closest to the first and last shot, or None if the window is empty.'''
			lo = bisect.bisect_left(sorted_floats, t_first - pad)
			hi = bisect.bisect_right(sorted_floats, t_last + pad)
			if lo >= hi:
				return None
			i_first = min(range(lo, hi), key=lambda j: abs(sorted_floats[j] - t_first))
			i_last = min(range(lo, hi), key=lambda j: abs(sorted_floats[j] - t_last))
			return i_first, i_last, hi - lo

		_log(f"Run window: {_fmt_local(t_first)} -> {_fmt_local(t_last)} "
		     f"({t_last - t_first:.1f} s, shots {shot_first}..{shot_last})")

		pad_applied = 0.0
		selection = window_selection(0.0)
		if selection is None:
			retry_pad = None
			if pad_seconds is not None:
				_log("No interferometer trace acquired within the run window.")
				retry_pad = float(pad_seconds)
			elif interactive:
				# The prompt must be self-explanatory even with verbose=False,
				# so its context prints unconditionally here.
				print(f"\nNo interferometer trace was acquired within the run "
				      f"window {_fmt_local(t_first)} -> {_fmt_local(t_last)} "
				      f"of {os.path.basename(datarun_path)}.")
				if len(timestamp_array) > 1:
					period = float(np.median(np.diff(timestamp_array)))
				else:
					period = 0.0
				retry_pad = _prompt_pad(max(period, 10.0))
				if retry_pad is None:
					print("Stopped here: nothing merged for this datarun.")
					return 0
			else:
				_log("No interferometer trace acquired within the run window; "
				     "nothing merged (non-interactive, no pad_seconds).")
			if retry_pad is not None:
				_log(f"Retrying with window padded by +/-{retry_pad:g} s.")
				pad_applied = float(retry_pad)
				selection = window_selection(pad_applied)
			if selection is None:
				_log("Still no interferometer trace found; nothing merged.")
				return 0

		i_first, i_last, n_candidates = selection
		_log(f"{n_candidates} interferometer trace(s) inside window"
		     + (f" (pad {pad_applied:g} s)" if pad_applied else ""))

		# (shot_number, shot_trigger_time, trace_time, trace_name, file_idx);
		# if both ends picked the same trace, keep it once, under the closer shot.
		if i_first == i_last:
			ts = sorted_floats[i_first]
			if abs(ts - t_first) <= abs(ts - t_last):
				picks = [(shot_first, t_first) + all_pairs[i_first]]
			else:
				picks = [(shot_last, t_last) + all_pairs[i_last]]
			_log("Only one distinct trace selected; writing it for the closer shot.")
		else:
			picks = [(shot_first, t_first) + all_pairs[i_first],
			         (shot_last, t_last) + all_pairs[i_last]]

		for shot_num, shot_ts, trace_ts, trace_name, _fidx in picks:
			_log(f"Shot {shot_num} <- interferometer trace {trace_name} "
			     f"({_fmt_local(trace_ts)}, {trace_ts - shot_ts:+.3f} s from shot trigger)")

		available_groups = _available_groups(datarun_path, per_file_groups)

		# Provenance: how the selection was made, recorded once per merge.
		with h5py.File(datarun_path, "a") as f_datarun:
			parent = f_datarun.require_group("diagnostics/interferometer")
			parent.attrs['timestamp source'] = (
				"Shot times: WAVEDESC trigger time from stored scope headers (scope "
				"RTC, local time); 'acquisition_time' attr fallback for undecodable "
				"headers. Interferometer traces acquired within the run window were "
				"selected: the one closest to the first shot (saved under the first "
				"shot number) and the one closest to the last shot (saved under the "
				"last shot number). Sequence-mode shots are timed by their first "
				"segment only. See each dataset's attributes for the exact "
				"interferometer time saved.")
			parent.attrs['reference scope'] = ref_scope
			parent.attrs['run window (s since epoch)'] = np.array([t_first, t_last])
			parent.attrs['window pad applied (s)'] = pad_applied
			parent.attrs['merged shot numbers'] = np.array(
				[p[0] for p in picks], dtype=int)
			parent.attrs['merged interferometer timestamps (s since epoch)'] = np.array(
				[p[2] for p in picks])

		for shot_num, shot_ts, trace_ts, trace_name, file_idx in picks:
			# Clearly note which interferometer time this data is from.
			provenance = {
				'interferometer timestamp (s since epoch)': float(trace_ts),
				'interferometer time (local)': _fmt_local(trace_ts),
				'time difference from shot trigger (s)': float(trace_ts - shot_ts),
			}
			if _copy_shot_datasets(datarun_path, f_interfs[file_idx],
			                       per_file_groups[file_idx], available_groups,
			                       trace_name, str(shot_num),
			                       extra_attrs=provenance):
				shots_written += 1

			_log(f"Shot {shot_num} wrote into datarun file")

	_log(f'Interferometer data merged into datarun file ({shots_written} shots written).')
	return shots_written


#===============================================================================================================================================
# Batch wrapper
#===============================================================================================================================================

def _is_lapd_daq_format(datarun_path):
	'''True if the file has at least one root scope group with shot_* children.'''
	try:
		with h5py.File(datarun_path, 'r') as f:
			return bool(_scope_groups(f))
	except OSError:
		return False


def _merged_times_note(datarun_path):
	'''Short "shot@time" summary of what a merge just wrote, from the
	provenance attrs (for batch/status log lines). Empty string if unavailable.'''
	try:
		with h5py.File(datarun_path, "r") as f:
			parent = f["diagnostics/interferometer"]
			shots = parent.attrs['merged shot numbers']
			stamps = parent.attrs['merged interferometer timestamps (s since epoch)']
		return ", ".join(f"shot {int(s)} @ {_fmt_local(t)}"
		                 for s, t in zip(shots, stamps))
	except Exception:
		return ""


def merge_folder(datarun_dir, interf_dir, pad_seconds=None):
	'''
	Run the first+last interferometer merge for every LAPD_DAQ-format datarun
	hdf5 in a folder.

	Each datarun is paired with interferometer_data_<date>.hdf5 (date from the
	filename or the file's ctime, with prev/next-day fallback), progress goes to
	the terminal and to interf_merge_log.txt in datarun_dir, and per-file errors
	don't abort the batch. Old-format dataruns are skipped with a pointer to
	interf_merge_datarun.py.

	Batch mode never prompts: a datarun whose strict run window contains no
	interferometer trace is reported EMPTY unless pad_seconds is given, in
	which case the padded retry is applied automatically. The interferometer
	times saved for each file are included in its OK log line.

	Returns:
	dict: {datarun_path: status_string} for each file processed.
	'''
	results = {}
	if not os.path.isdir(datarun_dir):
		print(f"Datarun directory not found: {datarun_dir}")
		return results
	if not os.path.isdir(interf_dir):
		print(f"Interferometer directory not found: {interf_dir}")
		return results

	datarun_files = sorted(f for f in os.listdir(datarun_dir)
	                       if f.lower().endswith('.hdf5')
	                       and not f.lower().startswith('interferometer_data_'))

	total = len(datarun_files)
	if total == 0:
		print(f"No .hdf5 datarun files found in {datarun_dir}")
		return results

	name_w = min(60, max((len(f) for f in datarun_files), default=20))
	idx_w = len(str(total))
	counts = {"ok": 0, "empty": 0, "skipped": 0, "error": 0}

	log_path = os.path.join(datarun_dir, "interf_merge_log.txt")
	try:
		log_file = open(log_path, "w", encoding="utf-8")
	except OSError as e:
		print(f"Warning: could not open log file {log_path}: {e}")
		log_file = None

	def emit(line):
		print(line)
		if log_file is not None:
			log_file.write(line + "\n")
			log_file.flush()

	emit(f"Batch merge (LAPD_DAQ, first+last shots) started at {datetime.datetime.now().isoformat(timespec='seconds')}")
	emit(f"Batch merge: {total} datarun file(s) from {datarun_dir}")
	emit(f"             interferometer files from {interf_dir}")
	emit(f"             pad_seconds={pad_seconds}")
	emit("-" * (idx_w * 2 + name_w + 30))

	try:
		for i, fname in enumerate(datarun_files, start=1):
			datarun_path = os.path.join(datarun_dir, fname)
			prefix = f"[{i:>{idx_w}}/{total}] {fname:<{name_w}}"

			if not _is_lapd_daq_format(datarun_path):
				emit(f"{prefix}  SKIP  not LAPD_DAQ format (old datarun? use interf_merge_datarun.py)")
				results[datarun_path] = "skipped: not LAPD_DAQ format"
				counts["skipped"] += 1
				continue

			candidates, date_source = _candidate_interf_files(datarun_path, interf_dir)
			if date_source is None:
				emit(f"{prefix}  SKIP  no date in filename or ctime")
				results[datarun_path] = "skipped: no date resolvable"
				counts["skipped"] += 1
				continue
			if not candidates:
				emit(f"{prefix}  SKIP  no interf file (date from {date_source})")
				results[datarun_path] = f"skipped: no interf file (date from {date_source})"
				counts["skipped"] += 1
				continue

			interf_name = os.path.basename(candidates[0])
			if len(candidates) > 1:
				interf_name = f"{interf_name} (+{len(candidates) - 1})"
			src_tag = " [ctime]" if date_source == "ctime" else ""

			try:
				init_datarun_groups(datarun_path, candidates, verbose=False)
				n_written = merge_interferometer_data(
					datarun_path, candidates, pad_seconds=pad_seconds,
					interactive=False, verbose=False)
				if n_written == 0:
					emit(f"{prefix}  EMPTY {interf_name}{src_tag}  "
					     "(no interferometer trace in run window)")
					results[datarun_path] = f"empty ({interf_name})"
					counts["empty"] += 1
				else:
					note = _merged_times_note(datarun_path)
					emit(f"{prefix}  OK    {interf_name}{src_tag}  "
					     f"({n_written} shots: {note})")
					results[datarun_path] = f"ok: {n_written} shots ({interf_name})"
					counts["ok"] += 1
			except Exception as e:
				err = f"{type(e).__name__}: {e}"
				emit(f"{prefix}  ERROR {err}")
				results[datarun_path] = f"error: {err}"
				counts["error"] += 1

		emit("-" * (idx_w * 2 + name_w + 30))
		emit(f"Batch done: {counts['ok']} ok, {counts['empty']} empty, "
		     f"{counts['skipped']} skipped, {counts['error']} error "
		     f"(total {total})")
		emit(f"Batch merge finished at {datetime.datetime.now().isoformat(timespec='seconds')}")
	finally:
		if log_file is not None:
			log_file.close()
			print(f"Log written to {log_path}")
	return results


#===============================================================================================================================================
# Offload-facing entry point (called at the end of Offload_Run.py)
#===============================================================================================================================================

def run_interferometer_merge(hdf5_path):
	'''
	Merge first+last interferometer traces into a just-finalized datarun file.

	Always called at the end of offload, once all spool data has been written.
	Self-contained: locates the interferometer files under
	INTERFEROMETER_DATA_DIR. This function NEVER raises -- it prints and returns
	a final one-line status string so the offload terminal's last line reports
	whether the merge succeeded and, if not, exactly what failed:
	  OK      - n shot(s) merged
	  SKIPPED - the data folder or a same-date interferometer file is missing
	  NO traces - no interferometer trace fell in the run window
	  FAILED  - an exception occurred (type + message reported)

	Parameters:
	hdf5_path (str): Path to the just-finalized datarun hdf5 file.

	Returns:
	str: The status line that was printed (last line of the offload terminal).
	'''
	try:
		if not INTERFEROMETER_DATA_DIR or not os.path.isdir(INTERFEROMETER_DATA_DIR):
			msg = (f"Interferometer merge SKIPPED: data folder not found "
			       f"({INTERFEROMETER_DATA_DIR!r}).")
			print(msg)
			return msg

		candidates, date_source = _candidate_interf_files(hdf5_path, INTERFEROMETER_DATA_DIR)
		if not candidates:
			msg = (f"Interferometer merge SKIPPED: no interferometer_data_<date>.hdf5 "
			       f"for {os.path.basename(hdf5_path)} in {INTERFEROMETER_DATA_DIR}.")
			print(msg)
			return msg

		init_datarun_groups(hdf5_path, candidates, verbose=False)
		n = merge_interferometer_data(hdf5_path, candidates,
		                              pad_seconds=None, interactive=False,
		                              verbose=True)
		if n == 0:
			msg = ("Interferometer merge: NO traces in run window "
			       f"(from {os.path.basename(candidates[0])}); nothing merged.")
		else:
			note = _merged_times_note(hdf5_path)
			msg = f"Interferometer merge OK: {n} shot(s) merged ({note})."
		print(msg)
		return msg
	except Exception as e:
		msg = f"Interferometer merge FAILED: {type(e).__name__}: {e}"
		print(msg)
		return msg


#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
	# Manual re-run during development: pass a finalized datarun hdf5 as the
	# first argument (interferometer files are located under
	# INTERFEROMETER_DATA_DIR), or edit the default below.
	if len(sys.argv) > 1:
		datarun_path = sys.argv[1]
	else:
		datarun_path = r"E:\Shadow data\Electrode_Biasing\jun2026\example_datarun.hdf5"

	run_interferometer_merge(datarun_path)

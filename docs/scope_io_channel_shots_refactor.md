# Refactor plan: single-pass `read_hdf5_scope_channel_shots`

## Scope

One function: `read_hdf5_scope_channel_shots` in
[`scope_io/hdf5.py`](../scope_io/hdf5.py). Behavior-preserving structural
cleanup only — no change to the public signature, return values, or the
NaN-row semantics. This is the "two-pass loop" flagged during the `/simplify`
pass on branch `refactor/read-analyze-no-lab-scopes`.

## Problem

The function reads N shots of one channel into a `(nshot, nsamples)` float64
stack. It currently walks `shot_numbers` **twice**:

1. **Pass 1** (find scaling): loop until the first readable, non-skipped shot
   that has a `_data` + `_header`, decode its WAVEDESC once for
   `(gain, offset, dt, t0)`, then `break`.
2. **Pass 2** (read rows): loop over *all* shots, read each `_data`, scale with
   the known `gain/offset`, and build rows (NaN for skipped / missing /
   wrong-length).

Each shot's group `f[scope_name][f'shot_{s}']` is therefore looked up up to
twice. The duplication is small in the common case (Pass 1 usually `break`s on
shot 1) but the two loops repeat the same group-access and skipped-check logic,
which is the real maintenance cost: the "is this shot usable" rule lives in two
places and can drift.

## Why the naive merge is unsafe (constraints to preserve)

A correct single pass must respect three behaviors that the current structure
gets "for free" by ordering scaling-resolution before row-reading:

- **`nsamples` precedence.** When `expected_len is None`, the row width is
  defined by the **first readable shot's** sample count; every later shot must
  match it or become a NaN row. A lazy "decode scaling on first good shot, then
  fill" merge has an ordering hazard: shots *before* the first good one need
  NaN rows, but their width isn't known yet.
- **Whole-result `None`.** If no shot is readable, the function returns
  `(None, None, None)` — not an empty/zero-width stack.
- **Per-shot failure isolation.** A missing/skipped/wrong-length shot yields a
  NaN row; it must never abort the whole read.

These are pinned by the existing tests in
[`tests/test_scope_io.py`](../tests/test_scope_io.py)
(`test_read_hdf5_scope_channel_shots`,
`test_read_hdf5_scope_channel_shots_none_when_unreadable`).

## Approach

Keep a single pass over the shots, but **defer materialization** so the
`nsamples` precedence is handled without a second loop. Collect each shot's raw
int16 array (or `None`) in order, decoding the channel scaling lazily on the
first shot that yields data. Resolve the row width once at the end, then scale
and stack. This collapses the two group-access loops into one and moves the
"is this shot usable" logic into a single small helper.

### Step 1 — extract a per-shot raw reader

Add a private helper next to `_scope_channel_scaling`, matching the existing
helper style (focused, `_`-prefixed, docstring states what it returns and when
it returns `None`):

```python
def _read_shot_raw(f, scope_name, channel_name, shot_number):
    """Return one shot's raw int16 ``_data`` array, or ``None`` if unreadable.

    Unreadable means the shot group is missing, marked ``skipped``, or has no
    ``<channel>_data`` dataset. Never raises -- a bad shot is just ``None`` so
    the caller can emit a NaN row in its place.
    """
    try:
        shot_group = f[scope_name][f'shot_{shot_number}']
    except KeyError:
        return None
    if shot_group.attrs.get('skipped', False):
        return None
    if f'{channel_name}_data' not in shot_group:
        return None
    return shot_group[f'{channel_name}_data'][:]
```

This becomes the single source of truth for "is this shot usable," used by the
one remaining loop.

### Step 2 — rewrite the body as one pass

```python
def read_hdf5_scope_channel_shots(f, scope_name, channel_name, shot_numbers,
                                  expected_len=None):
    """<docstring unchanged>"""
    shot_numbers = list(shot_numbers)

    # One pass: collect raw int16 per shot (None if unreadable) and decode the
    # channel scaling once, on the first shot that actually yields data.
    raws = []
    gain = offset = dt = t0 = None
    for s in shot_numbers:
        raw = _read_shot_raw(f, scope_name, channel_name, s)
        if raw is not None and gain is None:
            try:
                gain, offset, dt, t0 = _scope_channel_scaling(
                    f, scope_name, channel_name, s)
            except (KeyError, ValueError):
                raw = None          # header unreadable -> treat shot as a gap
        raws.append(raw)

    if gain is None:                # nothing readable
        return None, None, None

    # Row width: caller's expected_len, else the first readable shot's length.
    nsamples = expected_len
    if nsamples is None:
        nsamples = next(len(r) for r in raws if r is not None)

    nan_row = np.full(nsamples, np.nan, dtype=np.float64)
    stack = np.vstack([
        r.astype(np.float64) * gain - offset if (r is not None and len(r) == nsamples)
        else nan_row
        for r in raws
    ])
    return stack, dt, t0
```

Notes on equivalence to the current code:

- **Scaling source shot.** Today scaling is decoded from the first shot that
  passes the usable check *and* whose header decodes. Here, the first shot
  whose `_data` reads sets `gain`; if its header fails to decode, that shot is
  downgraded to a gap (`raw = None`) and the next readable shot is tried —
  same outcome as the current Pass 1 `continue`-on-decode-failure.
- **`nsamples` precedence.** `next(len(r) ...)` picks the first readable shot's
  length, identical to the current `if nsamples is None: nsamples = len(raw)`
  on the first row read.
- **NaN rows.** Skipped/missing (`r is None`) and wrong-length
  (`len(r) != nsamples`) both fall to `nan_row`, as before.
- **`None` result.** `gain is None` after the loop still means "no shot
  readable" → `(None, None, None)`.

### Step 3 — tests

No new public behavior, so the existing two tests must pass unchanged. Add
**one** regression test that pins the precedence edge the merge is most likely
to break: an unreadable shot *before* the first good shot, with
`expected_len=None`, so the width is taken from a later shot and the earlier
row is NaN:

```python
def test_channel_shots_width_from_later_shot_when_first_skipped(tmp_path):
    # shot 1 skipped, shot 2 good -> width comes from shot 2, row 0 is NaN
    ...
    stack, dt, t0 = read_hdf5_scope_channel_shots(
        f, "bdotscope", "C1", [1, 2])          # no expected_len
    assert stack.shape == (2, 8)
    assert np.all(np.isnan(stack[0]))
    np.testing.assert_array_equal(stack[1], single2)
```

## Verification

- `python -m pytest tests/test_scope_io.py -q` — existing + new test green.
- `python -m pytest -q` — full suite stays at its current pass count.
- Standalone import check (the refactor adds no new imports, but re-run the
  `lab_scopes`-blocked import proof to be safe).

## Risk and rollback

Low risk, single function, fully covered by tests. The change is isolated to
`scope_io/hdf5.py` plus one added test; revert is a one-file checkout if the
edge-case test surfaces any divergence.

## Out of scope

- `read_hdf5_scope_data` and the other readers (unchanged).
- Any change to `scope_io/wavedesc.py`, the public API, or return contracts.
- Performance tuning beyond removing the duplicate group access (this is an
  offline analysis path; correctness and readability lead).
```

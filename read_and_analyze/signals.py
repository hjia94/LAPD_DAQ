# -*- coding: utf-8 -*-
"""
User-defined signals: what gets mapped, as opposed to which channel is read.

A *signal* is the quantity a plot maps. Mapping a raw channel is the special
case; the general case is simple arithmetic across channels of the same scope --
``C3 - C4`` for a two-tip difference, ``(C3-C4)/(C3+C4)`` for a normalized
asymmetry, ``sqrt(C2*C2 + C3*C3)`` for the magnitude of a two-component vector.

Every signal answers three questions::

    signal.channels        -> ("C3", "C4")       which channels to read
    signal.combine(stacks) -> (nshot, nsamples)  how to reduce them to one trace
    signal.label / .key                          how to title and name the output

``combine`` runs **per shot, before any shot averaging**. That ordering is not
incidental: averaging and combining commute only for linear operations, so
``log(a/b)`` of the shot means is not the mean of ``log(a/b)``. Callers must
combine first and average the result.

Expressions are parsed with :mod:`ast` and evaluated against a whitelist of node
types and functions -- never with ``eval``. The channel list falls out of the
parse, so it is never declared twice and cannot drift from the expression.

Created Aug.2026
@author: Jia Han
"""

import ast
import re

import numpy as np

# A name in an expression must look like a scope channel (C1, C2, ... -- the
# WAVEDESC channel labels LeCroy writes). Without this, any bare identifier
# (`__builtins__`, a typo) parses as a "channel" and fails much later with a
# KeyError in the middle of a long read instead of at parse time.
_CHANNEL_RE = re.compile(r"[A-Za-z]+[0-9]+")

# Functions an expression may call. Each maps one array to one array elementwise.
# ``log``/``sqrt`` invalidate non-positive input rather than raising, so a bad
# sample becomes NaN (which every downstream reducer already skips) instead of
# killing a run partway through a long read.
_FUNCS = {
    "abs": np.abs,
    "log": lambda a: np.log(np.where(a > 0, a, np.nan)),
    "log10": lambda a: np.log10(np.where(a > 0, a, np.nan)),
    "sqrt": lambda a: np.sqrt(np.where(a >= 0, a, np.nan)),
}

# Binary/unary operators an expression may use. Division invalidates a zero
# denominator for the same reason the functions invalidate bad input.
_BINOPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: lambda a, b: np.divide(a, np.where(b != 0, b, np.nan)),
    ast.Pow: np.power,
}


class SignalError(ValueError):
    """An expression is malformed, or names something outside the whitelist."""


class Signal:
    """One mapped quantity: the channels it needs and how to combine them.

    ``combine`` receives ``{channel: (nshot, nsamples) array}`` -- every channel
    in :attr:`channels`, already read and filtered -- and returns one
    ``(nshot, nsamples)`` array. Rows are per shot throughout, so a caller that
    averages the result is averaging combined traces, never combining averages.
    """

    def __init__(self, channels, combine, label, key):
        self.channels = tuple(channels)
        self._combine = combine
        self.label = label
        self.key = key

    def combine(self, stacks):
        return self._combine(stacks)

    def __repr__(self):
        return f"Signal({self.key!r}, channels={self.channels!r})"


# ======================================================================================
# Expression parsing  (ast walk over a whitelist -- never eval)
# ======================================================================================

def _collect_names(node):
    """Every bare Name in the tree, in first-appearance order (deduped).

    These are the channel names the expression needs. Deriving them from the
    parse -- rather than asking the caller for them -- keeps the read list and
    the arithmetic from ever disagreeing.

    A call's ``func`` is itself a Name, so ``sqrt(C2)`` would otherwise report a
    channel named ``sqrt``; callee names are skipped. Whether the callee is an
    allowed function is :func:`_eval_node`'s decision, not this one's.
    """
    callees = {sub.func for sub in ast.walk(node)
               if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}
    names = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub not in callees and sub.id not in names:
            names.append(sub.id)
    return names


def _eval_node(node, values):
    """Evaluate one whitelisted AST node against ``{name: array}``.

    Anything not explicitly handled raises SignalError, so the whitelist is the
    node types named here and nothing else -- attribute access, subscripting,
    calls to unlisted functions, comprehensions, and lambdas all fall through to
    the final raise.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, values)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise SignalError(f"only numeric constants are allowed, got {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id not in values:
            raise SignalError(f"unknown channel {node.id!r}")
        return values[node.id]

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise SignalError(f"operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.left, values), _eval_node(node.right, values))

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return np.negative(_eval_node(node.operand, values))
        if isinstance(node.op, ast.UAdd):
            return _eval_node(node.operand, values)
        raise SignalError(f"unary {type(node.op).__name__} is not allowed")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise SignalError(
                f"function {name!r} is not allowed; available: {sorted(_FUNCS)}")
        if len(node.args) != 1 or node.keywords:
            raise SignalError(f"{node.func.id}() takes exactly one argument")
        return _FUNCS[node.func.id](_eval_node(node.args[0], values))

    raise SignalError(f"{type(node).__name__} is not allowed in a signal expression")


def expression(expr, label=None, key=None):
    """Signal from an arithmetic expression over channel names.

    ``expression("C3 - C4")``, ``expression("(C3-C4)/(C3+C4)")``,
    ``expression("sqrt(C2*C2 + C3*C3)")``. Allowed: channel names, numeric
    constants, ``+ - * / **``, unary minus, and abs/log/log10/sqrt. Everything
    else is refused at parse time.

    A bare channel name is a valid expression, so ``expression("C2")`` is exactly
    :func:`raw` -- callers can accept strings uniformly without special-casing.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise SignalError(f"cannot parse signal expression {expr!r}: {exc}") from exc

    channels = _collect_names(tree)
    if not channels:
        raise SignalError(f"signal expression {expr!r} names no channel")
    bad = [c for c in channels if not _CHANNEL_RE.fullmatch(c)]
    if bad:
        raise SignalError(
            f"{bad} does not look like a channel name (expected e.g. 'C1'); "
            "signal expressions may only reference scope channels")

    # Walk once now, with dummy arrays, so a bad expression fails here rather
    # than after a long read. Shape (1, 2) exercises the elementwise path.
    _eval_node(tree, {name: np.ones((1, 2)) for name in channels})

    def combine(stacks):
        return _eval_node(tree, {name: stacks[name] for name in channels})

    text = expr.strip()
    return Signal(channels, combine,
                  label=label or text,
                  key=key or _safe_key(text))


def _safe_key(text):
    """Filename-safe fragment for an expression: keep word chars, collapse the rest."""
    return re.sub(r"[^0-9A-Za-z_]+", "-", text).strip("-") or "signal"


# ======================================================================================
# Built-in constructors
# ======================================================================================

def raw(channel):
    """The channel itself, unmodified -- the default single-channel mapping."""
    return Signal((channel,), lambda stacks: stacks[channel],
                  label=channel, key=channel)


def difference(a, b):
    """``a - b`` per shot. Two-tip / two-probe differential."""
    return expression(f"{a} - {b}", label=f"{a} - {b}", key=f"{a}-minus-{b}")


def ratio_log(a, b):
    """``log(a/b)`` per shot; NaN where either is non-positive.

    The log-ratio flow proxy. Strongly nonlinear, so the per-shot ordering that
    :class:`Signal` guarantees is what keeps this meaningful.
    """
    return expression(f"log({a}/{b})", label=f"ln({a}/{b})", key=f"log-{a}-over-{b}")


def normalized_asymmetry(a, b):
    """``(a-b)/(a+b)`` per shot -- the normalized asymmetry flow proxy."""
    return expression(f"({a}-{b})/({a}+{b})",
                      label=f"({a}-{b})/({a}+{b})",
                      key=f"asym-{a}-{b}")


def magnitude(a, b):
    """``sqrt(a^2 + b^2)`` -- magnitude of a two-component vector.

    Orientation-independent, which makes it the safe choice when the coil sign
    and direction conventions of the two channels have not been established.
    """
    return expression(f"sqrt({a}*{a} + {b}*{b})",
                      label=f"|({a}, {b})|", key=f"mag-{a}-{b}")


def as_signal(spec):
    """Coerce a user spec to a Signal: pass through, or parse a string.

    Lets callers accept ``["C2", "C3 - C4", magnitude("C2","C3")]`` uniformly.
    """
    if isinstance(spec, Signal):
        return spec
    if isinstance(spec, str):
        return expression(spec)
    raise SignalError(f"cannot interpret {spec!r} as a signal")


def as_signal_list(specs):
    """Coerce None / one spec / a sequence of specs to a list of Signals or None."""
    if specs is None:
        return None
    if isinstance(specs, (str, Signal)):
        return [as_signal(specs)]
    return [as_signal(s) for s in specs]

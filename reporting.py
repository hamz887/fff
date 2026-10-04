# -*- coding: utf-8 -*-
"""
reporting.py — console/log reporter, progress bars and the checkpoint cache used by ``--resume``.
"""
import json
import os
import shutil
import sys
import textwrap
import time
import traceback
from dataclasses import asdict

import joblib

from .config import POST_HOC_SETTINGS

try:                                                   # optional: coloured console and progress bars
    from rich.console import Console
    from rich.markup import escape
    from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

W = 108
ANSI = {"blue": "\033[1;34m", "cyan": "\033[36m", "green": "\033[1;32m", "yellow": "\033[33m",
        "red": "\033[1;31m", "magenta": "\033[35m", "dim": "\033[2m", "": ""}
RICH_STYLE = {"blue": "bold blue", "cyan": "cyan", "green": "bold green", "yellow": "yellow", "red": "bold red",
              "magenta": "magenta", "dim": "dim", "": ""}


def fmt_int(n):
    return f"{int(n):,}"


def live(msg):
    print("\r" + msg[:W].ljust(W), end="", flush=True)


def live_end():
    print("\r" + " " * W + "\r", end="", flush=True)


class Reporter:
    """Research-style, colour-coded console output. The log file always receives plain text."""

    def __init__(self, log_path, append=False, ui="auto"):
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        self.f = open(log_path, "a" if append else "w", encoding="utf-8")
        self.t0 = time.time()
        tty = sys.stdout.isatty()
        self.mode = ui if ui != "auto" else ("rich" if HAS_RICH and tty else "ansi" if tty else "plain")
        if self.mode == "rich" and not HAS_RICH:
            self.mode = "ansi"
        if self.mode == "ansi" and os.name == "nt":
            os.system("")                                   # enables ANSI escape codes on Windows terminals
        self.console = Console(soft_wrap=True, highlight=False) if self.mode == "rich" else None

    def close(self):
        if not self.f.closed:
            self.f.close()

    @staticmethod
    def auto_style(msg):
        s = msg.lstrip()
        if s.startswith(("#", "=")):
            return "blue"
        if s.startswith("WARNING") or "WARNING:" in s:
            return "yellow"
        if s.startswith(("ERROR", "[!]")):
            return "red"
        if s.startswith(("DONE", "✔")) or " saved" in s:
            return "green"
        if s.startswith(("VALIDATION", "TEST")) or " VALIDATION" in s or "TEST (" in s:
            return "cyan"
        return ""

    def out(self, msg, style=None):
        style = self.auto_style(msg) if style is None else style
        if self.mode == "rich":
            self.console.print(Text(msg, style=RICH_STYLE.get(style, style)))
        elif self.mode == "ansi" and style:
            print(f"{ANSI[style]}{msg}\033[0m", flush=True)
        else:
            print(msg, flush=True)

    def say(self, msg="", style=None):
        self.out(msg, style)
        self.note(msg)

    def note(self, msg=""):
        self.f.write(msg + "\n")
        self.f.flush()

    def exception(self, exc):
        """Writes the full traceback of a fatal error to the log (and a short message to the console)."""
        self.note("\nFATAL ERROR\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        self.out(f"ERROR: {type(exc).__name__}: {exc}  (full traceback in the log file)", "red")

    def elapsed(self):
        s = int(time.time() - self.t0)
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def stage(self, title):
        self.say("\n" + "#" * W + f"\n# {title}   [elapsed {self.elapsed()}]\n" + "#" * W, "blue")

    def section(self, title):
        self.say(f"\n  {title}\n  " + "-" * (W - 2), "blue")

    def kv(self, k, v, indent=4, style=None):
        lines = textwrap.wrap(str(v), width=W - indent - 37) or [""]
        self.say(" " * indent + f"{k:<34}: {lines[0]}", style)
        for ln in lines[1:]:
            self.say(" " * (indent + 36) + ln, style)

    def table(self, headers, rows, widths=None, indent=4, log_only=False):
        widths = widths or [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]

        def isnum(v):
            t = str(v)
            for ch in ",%±/+-.e* ★":
                t = t.replace(ch, "")
            return t == "" or t.isdigit() or t == "nan"
        al = [">" if rows and all(isnum(r[i]) for r in rows) else "<" for i in range(len(headers))]

        def fmt(vals):
            return " " * indent + " ".join(f"{str(v):{a}{w - 1}}" for v, a, w in zip(vals, al, widths)).rstrip()
        emit = self.note if log_only else self.say
        emit(fmt(headers))
        emit(" " * indent + "-" * (sum(widths) - 1))
        for r in rows:
            emit(fmt(r))

    def model_summary(self, label, hist, info, metrics=None, cached=False, unit="Epoch"):
        """Per-epoch table + stop reason + validation metrics, rendered after each model finishes."""
        has_val = bool(hist) and "val_loss" in hist[0]
        head = [unit, "Train loss"] + (["Val loss", "Val F1"] if has_val else []) + ["LR", "Time (s)", "Best"]
        rows = [(h["epoch"], f"{h['train_loss']:.5f}") +
                ((f"{h['val_loss']:.5f}", f"{h['val_f1']:.4f}") if has_val else ()) +
                (f"{h['lr']:.1e}", f"{h['time_s']:.1f}", "★" if h.get("best") else "") for h in hist]
        stop = info.get("stop_reason", f"fixed {info.get('epochs_run')} {unit.lower()}s (selected by cross-validation)")
        best = f" | best {unit.lower()} {info['best_epoch']} restored" if info.get("best_epoch") else ""
        mline = (f"VALIDATION (threshold {metrics['threshold']:g}): Accuracy {metrics['accuracy']:.4f} | "
                 f"F1 {metrics['f1']:.4f} | Recall {metrics['recall']:.4f} | Precision {metrics['precision']:.4f} | "
                 f"FAR {metrics['far']:.4f} | MCC {metrics['mcc']:.4f} | AUC {metrics['roc_auc']:.4f}") if metrics else None
        title = f"{label} — training summary{' (restored from cache)' if cached else ''}"
        self.note(f"\n    {title}")
        self.table(head, rows, widths=[8, 12] + ([11, 9] if has_val else []) + [10, 10, 6], indent=6, log_only=True)
        self.note(f"      {stop}{best}")
        if mline:
            self.note("      " + mline)
        if self.mode == "rich":
            t = Table(title=Text(title, style="bold"), title_justify="left", header_style="bold cyan",
                      show_edge=False, pad_edge=False, box=None)
            for h in head:
                t.add_column(h, justify="right" if h != "Best" else "center")
            for r in rows:
                t.add_row(*[str(x) for x in r[:-1]], Text(r[-1], style="bold yellow"))
            self.console.print(t)
            self.console.print(Text(f"      {stop}{best}", style="magenta"))
            if mline:
                self.console.print(Text("      " + mline, style="cyan"))
        else:
            self.out(f"\n    {title}", "")
            self.out(" " * 6 + " ".join(f"{h:>10}" for h in head), "cyan")
            for r in rows:
                self.out(" " * 6 + " ".join(f"{str(x):>10}" for x in r), "yellow" if r[-1] else "")
            self.out(f"      {stop}{best}", "magenta")
            if mline:
                self.out("      " + mline, "cyan")


class TrainProgress:
    """Outer bar over epochs/rounds/runs and inner bar over mini-batches (or clients), refreshed <= 10 Hz."""

    def __init__(self, R, label, total, unit="epoch", inner_unit="batches"):
        self.R, self.label, self.total, self.unit, self.inner_unit = R, label, total, unit, inner_unit
        self.last, self.info, self.p = 0.0, "", None
        self.i, self.n_inner = 0, 1

    def __enter__(self):
        if self.R is not None and self.R.mode == "rich":
            self.p = Progress(SpinnerColumn(style="cyan"), TextColumn("{task.description}"),
                              BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
                              MofNCompleteColumn(), TextColumn("{task.fields[info]}", style="cyan"),
                              TimeElapsedColumn(), TimeRemainingColumn(), console=self.R.console,
                              transient=True, refresh_per_second=10)
            self.p.start()
            self.outer = self.p.add_task(f"[bold]{escape(self.label)}[/] {self.unit}s", total=self.total, info="")
            self.inner = self.p.add_task(f"  {self.inner_unit}", total=1, info="")
        return self

    def start(self, i, n_inner):
        self.i, self.n_inner = i, max(n_inner, 1)
        if self.p is not None:
            self.p.reset(self.inner, total=self.n_inner, description=f"  {self.unit} {i}/{self.total} {self.inner_unit}")

    def step(self, k, extra=""):
        now = time.time()
        if now - self.last < 0.1 and k < self.n_inner:
            return
        self.last = now
        if self.p is not None:
            self.p.update(self.inner, completed=k, info=extra)
        elif self.R is not None and self.R.mode != "plain":
            frac = min(k / self.n_inner, 1.0)
            bar = "█" * int(24 * frac) + "░" * (24 - int(24 * frac))
            col, rst = (ANSI["cyan"], "\033[0m") if self.R.mode == "ansi" else ("", "")
            live(f"      {self.label:<20} {self.unit} {self.i:>3}/{self.total} {col}{bar}{rst} {k}/{self.n_inner} {extra}")

    def end(self, info=""):
        self.info = info
        if self.p is not None:
            self.p.update(self.outer, advance=1, info=info)

    def __exit__(self, *a):
        if self.p is not None:
            self.p.stop()
        elif self.R is not None and self.R.mode != "plain":
            live_end()
        return False


class Cache:
    """Checkpoints of completed work units (outputs/cache). --resume reuses them; every unit is seeded, so a resumed
    run gives the same numbers as an uninterrupted one. Resuming is refused if any setting changed. The checkpoints
    are joblib pickles written by this program for its own use; never resume from a cache of unknown origin."""

    FORMAT = 6                                       # bump when payload contents change

    def __init__(self, cfg, resume):
        self.dir = cfg.dirs["cache"]
        snap = json.loads(json.dumps({k: v for k, v in asdict(cfg).items()
                                      if k != "out_dir" and k not in POST_HOC_SETTINGS}, default=str))
        snap["_cache_format"] = self.FORMAT
        path = os.path.join(self.dir, "config_snapshot.json")
        if resume and os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
            diff = sorted(k for k in set(old) | set(snap) if old.get(k) != snap.get(k))
            if diff:
                raise RuntimeError(f"cannot resume: settings changed: {diff}")
        else:
            shutil.rmtree(self.dir, ignore_errors=True)
            os.makedirs(self.dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2)
        self.hits = 0

    def run(self, uid, fn):
        p = os.path.join(self.dir, f"{uid}.joblib")
        if os.path.isfile(p):
            self.hits += 1
            return joblib.load(p), True
        out = fn()
        joblib.dump(out, p + ".tmp")
        os.replace(p + ".tmp", p)
        return out, False


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)

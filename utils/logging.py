"""
TUS-GAN: Centralized Logging Utility
=====================================

Usage (any module in the project):
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("message")

For training loops:
    from utils.logging import get_logger, log_epoch, log_gradient_penalty, log_generated_sample
"""

import logging
import sys
import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


# ──────────────────────────────────────────────
# CONSTANTS — edit these to change project-wide
# ──────────────────────────────────────────────

LOG_DIR = Path("logs")          # All logs land here
LOG_LEVEL = logging.DEBUG       # Minimum severity to capture
CONSOLE_LEVEL = logging.INFO    # What gets printed to terminal

# Structured JSON log for machine-readable parsing (metrics, losses)
METRICS_LOG = LOG_DIR / "metrics.jsonl"

# Human-readable log for all modules
FULL_LOG = LOG_DIR / "tusgan.log"


# ──────────────────────────────────────────────
# FORMATTERS
# ──────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    """
    Colored terminal output so team members can scan logs fast.

    Severity color scheme:
        DEBUG   → grey     (verbose internals, ignored unless debugging)
        INFO    → green    (normal progress: epoch done, data loaded)
        WARNING → yellow   (something's off but training continues)
        ERROR   → red      (something broke; check immediately)
        CRITICAL→ bold red (training will halt)
    """

    GREY    = "\x1b[38;5;240m"
    GREEN   = "\x1b[32m"
    YELLOW  = "\x1b[33m"
    RED     = "\x1b[31m"
    BOLD_RED = "\x1b[1;31m"
    RESET   = "\x1b[0m"

    FMT = "[{asctime}] [{levelname:<8}] [{name}] {message}"
    DATEFMT = "%H:%M:%S"

    COLORS = {
        logging.DEBUG:    GREY,
        logging.INFO:     GREEN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            fmt=color + self.FMT + self.RESET,
            datefmt=self.DATEFMT,
            style="{"
        )
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    """
    No color — for writing to .log files (color escape codes look ugly in files).
    """
    def __init__(self):
        super().__init__(
            fmt="[{asctime}] [{levelname:<8}] [{name}] {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{"
        )


# ──────────────────────────────────────────────
# CORE SETUP
# ──────────────────────────────────────────────

_initialized = False  # Prevent duplicate handlers if get_logger is called multiple times


def _setup_root_logger():
    """
    Called once at import time. Attaches two handlers to the root logger:
      1. StreamHandler  → prints to terminal (INFO and above)
      2. FileHandler    → writes everything (DEBUG+) to logs/tusgan.log
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("tusgan")
    root.setLevel(LOG_LEVEL)

    # Terminal handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(CONSOLE_LEVEL)
    console.setFormatter(ColorFormatter())

    # File handler
    file_handler = logging.FileHandler(FULL_LOG, mode="a", encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(PlainFormatter())

    root.addHandler(console)
    root.addHandler(file_handler)
    root.propagate = False  # Don't bubble up to Python's root logger

    root.info(f"Logger initialized. Full log → {FULL_LOG.resolve()}")


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger namespaced under 'tusgan.<name>'.

    Convention: always call with __name__ so logs show the module path.
        from utils.logging import get_logger
        logger = get_logger(__name__)

    If called from model/generator.py, the logger name becomes:
        tusgan.model.generator
    This tells you exactly which file emitted each log line.
    """
    _setup_root_logger()

    # Strip leading project path if running as a module
    # e.g. "tusgan.model.generator" stays clean
    qualified = f"tusgan.{name}" if not name.startswith("tusgan") else name
    return logging.getLogger(qualified)


# ──────────────────────────────────────────────
# STRUCTURED METRICS LOGGER
# ──────────────────────────────────────────────

class MetricsLogger:
    """
    Writes one JSON object per line to logs/metrics.jsonl.

    Why .jsonl (JSON Lines)?
    - Each line is a valid JSON object → easy to load with pandas:
        df = pd.read_json("logs/metrics.jsonl", lines=True)
    - Append-safe: no file corruption if training crashes mid-write
    - Easy to grep, plot, or pipe into TensorBoard

    Example output in metrics.jsonl:
        {"ts": "2026-05-30T14:32:01", "epoch": 5, "step": 200,
         "critic_loss": -1.234, "gen_loss": 0.891, "gp": 0.102,
         "split": "train", "author": "venkat"}
    """

    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._path = METRICS_LOG

    def log(self, payload: Dict[str, Any]):
        payload["ts"] = datetime.utcnow().isoformat(timespec="seconds")
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


_metrics_logger = MetricsLogger()


# ──────────────────────────────────────────────
# TRAINING-SPECIFIC HELPERS
# ──────────────────────────────────────────────

def log_epoch(
    epoch: int,
    total_epochs: int,
    critic_loss: float,
    gen_loss: float,
    gp: float,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
):
    """
    Call at the end of every epoch (or every N steps) during training.

    Logs to:
      - Terminal / tusgan.log  (human-readable)
      - metrics.jsonl          (machine-readable for later plotting)

    Args:
        epoch:        Current epoch number (1-indexed)
        total_epochs: Total planned epochs
        critic_loss:  Wasserstein critic loss for this epoch
        gen_loss:     Generator loss
        gp:           Raw gradient penalty value (before λ scaling)
        step:         Global training step (optional, useful for sub-epoch logging)
        extra:        Any additional key-value pairs (e.g. FID, activity accuracy)
        author:       Who ran this experiment — important for multi-person projects!
    """
    logger = get_logger("training.loop")

    msg = (
        f"Epoch [{epoch:>4}/{total_epochs}]"
        f"  critic={critic_loss:+.4f}"
        f"  gen={gen_loss:+.4f}"
        f"  gp={gp:.4f}"
    )
    if step is not None:
        msg += f"  step={step}"
    if author:
        msg += f"  [{author}]"

    logger.info(msg)

    payload = {
        "epoch": epoch,
        "critic_loss": round(float(critic_loss), 6),
        "gen_loss": round(float(gen_loss), 6),
        "gradient_penalty": round(float(gp), 6),
        "split": "train",
    }
    if step is not None:
        payload["step"] = step
    if author:
        payload["author"] = author
    if extra:
        payload.update(extra)

    _metrics_logger.log(payload)


def log_gradient_penalty(gp_value: float, step: int, threshold: float = 5.0):
    """
    Specifically monitors the gradient penalty value during WGAN-GP training.

    Why this matters for TUS-GAN:
      GP should hover around 0 if the critic satisfies the 1-Lipschitz
      constraint. A spike means the critic is violating the constraint —
      training is becoming unstable. This helper warns you immediately.

    Args:
        gp_value:  Raw gradient penalty (before λ multiplication)
        step:      Current global step
        threshold: Warn if gp_value exceeds this (default 5.0)
    """
    logger = get_logger("training.wgan_gp")
    if gp_value > threshold:
        logger.warning(
            f"[Step {step}] Gradient penalty spike: {gp_value:.4f} "
            f"(threshold={threshold}). Critic may be violating Lipschitz constraint."
        )
    else:
        logger.debug(f"[Step {step}] GP={gp_value:.4f} ✓")


def log_generated_sample(epoch: int, slot_accuracy: Optional[float] = None):
    """
    Log when a batch of synthetic diaries is generated/saved.

    Args:
        epoch:         Epoch at which samples were saved
        slot_accuracy: Optional — fraction of 48 slots with valid activity codes
    """
    logger = get_logger("training.generator")
    msg = f"[Epoch {epoch}] Generated diary samples saved."
    if slot_accuracy is not None:
        msg += f"  Slot accuracy: {slot_accuracy:.3f}"
        if slot_accuracy < 0.7:
            logger.warning(msg + "  ← Below 0.70 — check activity code mapping.")
            return
    logger.info(msg)


# ──────────────────────────────────────────────
# DATA PIPELINE HELPERS
# ──────────────────────────────────────────────

def log_dataset_load(split: str, year: int, n_samples: int, n_dropped: int = 0):
    """
    Call after loading ITUS 2019 or ITUS 2024.

    Args:
        split:     "train" | "val" | "test"
        year:      2019 or 2024
        n_samples: Number of diaries loaded
        n_dropped: Rows dropped during preprocessing
    """
    logger = get_logger("data.loader")
    logger.info(
        f"[ITUS {year}] Loaded {n_samples:,} diaries  split={split}"
        + (f"  dropped={n_dropped:,}" if n_dropped else "")
    )
    if n_dropped > 0.05 * n_samples:
        logger.warning(
            f"[ITUS {year}] {n_dropped/n_samples:.1%} of rows dropped — "
            f"check preprocessing pipeline."
        )


def log_tensor_shape(name: str, shape: tuple):
    """
    Quick debug helper for verifying diary tensor shape (B, 3, 48, 1).

    Example:
        log_tensor_shape("diary_tensor", tuple(x.shape))
        → [DEBUG] [tusgan.data.encoder] diary_tensor: (32, 3, 48, 1) ✓
    """
    logger = get_logger("data.encoder")
    expected = (None, 3, 48, 1)
    ok = (
        len(shape) == 4
        and shape[1] == 3
        and shape[2] == 48
        and shape[3] == 1
    )
    status = "✓" if ok else "✗ UNEXPECTED — expected (B, 3, 48, 1)"
    logger.debug(f"{name}: {shape} {status}")
    if not ok:
        logger.error(
            f"Tensor shape mismatch for '{name}': got {shape}, expected (B, 3, 48, 1). "
            f"Check diary encoding pipeline."
        )


# ──────────────────────────────────────────────
# EXPERIMENT HEADER
# ──────────────────────────────────────────────

def log_experiment_start(config: Dict[str, Any], author: str):
    """
    Call once at the top of train.py.
    Prints a visible separator so multiple runs in the same log file
    are easy to distinguish.

    Args:
        config: Your training config dict (hyperparams, paths, etc.)
        author: Name of the person running the experiment
                (critical when Venkat, Harshitha, etc. share the same log file)
    """
    logger = get_logger("training.experiment")
    sep = "═" * 60
    logger.info(sep)
    logger.info(f"  TUS-GAN Experiment | Author: {author}")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(sep)
    for k, v in config.items():
        logger.info(f"  {k:<25} {v}")
    logger.info(sep)

    _metrics_logger.log({
        "event": "experiment_start",
        "author": author,
        "config": config,
    })


def log_exception(context: str = ""):
    """
    Call inside except blocks to log full tracebacks with context.

    Example:
        try:
            train_step(batch)
        except Exception:
            log_exception("train_step failed at epoch 3")
            raise
    """
    logger = get_logger("errors")
    tb = traceback.format_exc()
    msg = f"Exception" + (f" in [{context}]" if context else "") + f":\n{tb}"
    logger.error(msg)

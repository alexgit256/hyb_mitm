# - - - precision wrapper - - -

import os
from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager

from fpylll import IntegerMatrix, GSO

# ---------------------------
# MPFR precision handling
# ---------------------------

def _try_set_mpfr_precision(bits: int) -> bool:
    """
    Try to set MPFR precision in fpylll/fplll. Returns True if it likely succeeded.
    Works across multiple fpylll versions by probing attribute names.
    """
    bits = int(bits)
    if bits <= 0:
        return False

    # Try common entry points (versions vary)
    try:
        from fpylll import FPLLL  # type: ignore
        if hasattr(FPLLL, "set_precision"):
            FPLLL.set_precision(bits)
            return True
        if hasattr(FPLLL, "set_mpfr_precision"):
            FPLLL.set_mpfr_precision(bits)
            return True
    except Exception:
        pass

    try:
        from fpylll.fplll import FPLLL  # type: ignore
        if hasattr(FPLLL, "set_precision"):
            FPLLL.set_precision(bits)
            return True
        if hasattr(FPLLL, "set_mpfr_precision"):
            FPLLL.set_mpfr_precision(bits)
            return True
    except Exception:
        pass

    return False


@contextmanager
def mpfr_precision(bits: int):
    """
    Context manager: attempt to set MPFR precision for the duration of a block.
    If we can't set it (API mismatch), we proceed anyway (MPFR will use default precision).
    """
    ok = _try_set_mpfr_precision(bits)
    # No reliable cross-version "get_precision" to restore, so we do best-effort only.
    try:
        yield ok
    finally:
        pass


# ---------------------------
# Backend availability probing
# ---------------------------

@lru_cache(maxsize=1)
def _available_float_types():
    """
    Probe which float_type values work in *this* fpylll build.
    """
    B = IntegerMatrix.identity(2)
    U = IntegerMatrix.identity(2, int_type=B.int_type)
    UinvT = IntegerMatrix.identity(2, int_type=B.int_type)

    candidates = ("double", "long double", "dd", "mpfr")
    ok = []
    for ft in candidates:
        try:
            if ft == "mpfr":
                with mpfr_precision(106):
                    _ = GSO.Mat(B, float_type=ft, U=U, UinvT=UinvT)
            else:
                _ = GSO.Mat(B, float_type=ft, U=U, UinvT=UinvT)
            ok.append(ft)
        except Exception:
            pass

    if not ok:
        raise RuntimeError("No usable GSO float_type backend found in this fpylll build.")
    return tuple(ok)

def _has(ft: str) -> bool:
    return ft in _available_float_types()


# ---------------------------
# Policy with "mpfr106/mpfr237" semantics
# ---------------------------

@dataclass(frozen=True)
class FloatChoice:
    float_type: str          # "double"/"dd"/"long double"/"mpfr"
    mpfr_bits: int | None    # only used if float_type == "mpfr"
    label: str               # e.g. "dd", "mpfr106", "mpfr237"

class GSOFloatPolicyBits:
    """
    Dimension-based policy:
      - small  -> double
      - mid    -> dd if available else mpfr106 (or long double if you prefer)
      - large  -> mpfr237 (your qd-ish target) if mpfr is available, else best available
    You can override via env var FPYLLL_FLOAT_CHOICE with values:
      double | dd | long double | mpfr106 | mpfr237 | mpfr<INT>
    """

    def __init__(
        self,
        small_n: int = 140,
        large_n: int = 400,
        dd_bits: int = 106,
        qd_bits: int = 237,
        env_override: str = "FPYLLL_FLOAT_CHOICE",
        prefer_long_double_over_mpfr106: bool = False,
    ):
        self.small_n = int(small_n)
        self.large_n = int(large_n)
        self.dd_bits = int(dd_bits)
        self.qd_bits = int(qd_bits)
        self.env_override = str(env_override)
        self.prefer_long_double_over_mpfr106 = bool(prefer_long_double_over_mpfr106)

    def _parse_override(self, s: str) -> FloatChoice:
        s = s.strip()
        if not s:
            raise ValueError("empty override")
        if s in ("double", "dd", "long double", "mpfr"):
            if s == "mpfr":
                return FloatChoice("mpfr", self.dd_bits, f"mpfr{self.dd_bits}")
            return FloatChoice(s, None, s)

        if s.startswith("mpfr"):
            bits_str = s[4:]
            bits = int(bits_str)
            return FloatChoice("mpfr", bits, f"mpfr{bits}")

        raise ValueError(f"Unrecognized override {s!r}")

    def choose(self, n: int) -> FloatChoice:
        # 1) reproducibility override
        forced = os.getenv(self.env_override, "").strip()
        if forced:
            ch = self._parse_override(forced)
            if ch.float_type != "mpfr":
                if _has(ch.float_type):
                    return ch
                raise ValueError(
                    f"{self.env_override}={forced!r} requested, but {ch.float_type!r} is unavailable. "
                    f"Available: {_available_float_types()}"
                )
            # mpfr forced
            if not _has("mpfr"):
                raise ValueError(
                    f"{self.env_override}={forced!r} requested, but 'mpfr' is unavailable. "
                    f"Available: {_available_float_types()}"
                )
            return ch

        # 2) default policy
        n = int(n)

        if n < self.small_n:
            if _has("double"):
                return FloatChoice("double", None, "double")
            return FloatChoice(_available_float_types()[0], None, _available_float_types()[0])

        if n > self.large_n:
            if _has("mpfr"):
                return FloatChoice("mpfr", self.qd_bits, f"mpfr{self.qd_bits}")
            # no mpfr: fall back to best we have
            for ft in ("dd", "long double", "double"):
                if _has(ft):
                    return FloatChoice(ft, None, ft)
            ft0 = _available_float_types()[0]
            return FloatChoice(ft0, None, ft0)

        # medium
        if _has("dd"):
            return FloatChoice("dd", None, "dd")

        # dd missing: emulate dd via mpfr106 (or optionally long double first)
        if self.prefer_long_double_over_mpfr106 and _has("long double"):
            return FloatChoice("long double", None, "long double")

        if _has("mpfr"):
            return FloatChoice("mpfr", self.dd_bits, f"mpfr{self.dd_bits}")

        # no mpfr either: settle
        if _has("long double"):
            return FloatChoice("long double", None, "long double")
        if _has("double"):
            return FloatChoice("double", None, "double")
        ft0 = _available_float_types()[0]
        return FloatChoice(ft0, None, ft0)


# ---------------------------
# One-call constructor
# ---------------------------

def make_gso_mat(B: IntegerMatrix, policy: GSOFloatPolicyBits | None = None) -> GSO.Mat:
    if policy is None:
        policy = GSOFloatPolicyBits()

    n = B.nrows
    choice = policy.choose(n)

    U = IntegerMatrix.identity(n, int_type=B.int_type)
    UinvT = IntegerMatrix.identity(n, int_type=B.int_type)

    if choice.float_type == "mpfr":
        bits = int(choice.mpfr_bits or policy.dd_bits)
        with mpfr_precision(bits):
            return GSO.Mat(B, float_type="mpfr", U=U, UinvT=UinvT)

    return GSO.Mat(B, float_type=choice.float_type, U=U, UinvT=UinvT)

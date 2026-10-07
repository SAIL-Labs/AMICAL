import functools
import warnings

VIRGIL_URL = "https://benjaminpope.github.io/virgil/"


def deprecated_fitter(func):
    """Mark AMICAL's own fitting tools (smartfit) as deprecated.

    AMICAL's job ends at the calibrated OIFITS file; model fitting, binary
    searches and contrast limits are better done with a dedicated package
    such as virgil, which reads AMICAL's OIFITS output directly. (The CANDID
    and Pymask functions are kept as a legacy interface computed with virgil:
    see amical.analysis.legacy.)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"amical.{func.__name__} is deprecated and will be removed in a "
            "future release. Fit the calibrated OIFITS file with virgil "
            f"(pip install virgil-astro; {VIRGIL_URL}) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper

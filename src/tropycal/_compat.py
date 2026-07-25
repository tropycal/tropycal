r"""Internal compatibility helpers for newer/older dependency versions.

These helpers let tropycal work with the newest versions of Python and its
scientific dependencies while falling back gracefully on older versions.
"""

from datetime import datetime, timedelta

try:
    # Python 3.2+; used to construct aware UTC datetimes
    from datetime import timezone
    _HAS_TIMEZONE = True
except ImportError:  # pragma: no cover
    _HAS_TIMEZONE = False


def get_cmap(name):
    r"""Retrieve a matplotlib colormap by name.

    ``matplotlib.cm.get_cmap`` was deprecated in matplotlib 3.7 and removed in
    3.9. The replacement registry ``matplotlib.colormaps`` was added in 3.5.
    This helper works on both old and new versions. If ``name`` is already a
    colormap object, it is returned unchanged.
    """
    if not isinstance(name, str):
        return name
    try:
        from matplotlib import colormaps
        return colormaps[name]
    except ImportError:  # matplotlib < 3.5
        from matplotlib.cm import get_cmap as _mpl_get_cmap
        return _mpl_get_cmap(name)


def utcnow():
    r"""Return the current naive UTC datetime.

    ``datetime.utcnow()`` is deprecated since Python 3.12. This preserves the
    original naive-UTC semantics (tzinfo stripped) so comparisons against
    other naive datetimes in the codebase continue to work.
    """
    if _HAS_TIMEZONE:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    return datetime.utcnow()  # pragma: no cover

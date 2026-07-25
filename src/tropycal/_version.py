r"""Specify tropycal version."""

def get_version():
    r"""Get the latest version of tropycal."""
    
    try:
        from setuptools_scm import get_version
        return get_version(root='..', relative_to=__file__,
                           version_scheme='post-release', local_scheme='dirty-tag')
    except (ImportError, LookupError):
        try:
            from importlib.metadata import version, PackageNotFoundError
        except ImportError:  # Python < 3.8
            from importlib_metadata import version, PackageNotFoundError
        try:
            return version(__package__ or 'tropycal')
        except PackageNotFoundError:
            return 'unknown'
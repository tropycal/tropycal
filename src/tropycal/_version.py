r"""Specify tropycal version."""

def get_version():
    r"""Get the latest version of tropycal."""
    
    try:
        from setuptools_scm import get_version
        return get_version(root='..', relative_to=__file__,
                           version_scheme='post-release', local_scheme='dirty-tag')
    except (ImportError, LookupError):
        from pkg_resources import get_distribution
        return get_distribution(__package__).version
from importlib.metadata import version, PackageNotFoundError
__all__ = []



try:
    __version__ = version("sloeginphys")
except PackageNotFoundError:
    # package is not installed
    pass

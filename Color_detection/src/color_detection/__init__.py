import sys

# Check python version to determine import source
if sys.version_info[:2] >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Package metadata configuration
    dist_name = "Color_detection"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
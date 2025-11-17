"""DOFT Cluster Simulator package."""

# Lazy re-export to avoid runpy warning when invoking `python -m ...`.

def run_from_args(argv=None):
    from .cli import run_from_args as _run

    return _run(argv)


__all__ = ["run_from_args"]

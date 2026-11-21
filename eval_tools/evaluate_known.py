"""Resume development evaluation and apply the frozen known-review policy."""

import sys

from .evaluate import main


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:], known=True))

"""Allow ``python -m ai_stock`` to run the CLI."""

import sys

from ai_stock.cli import main

if __name__ == "__main__":
    sys.exit(main())

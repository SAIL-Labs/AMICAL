from collections.abc import Sequence
from typing import Any

from astropy.table import Table


def tabulate(rows: Sequence[Any], headers: list[str]) -> Table:
    # a drop-in replacement for tabulate.tabulate
    data = {name: [row[i] for row in rows] for i, name in enumerate(headers)}
    return Table(data)

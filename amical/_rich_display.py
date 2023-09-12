from typing import Any, List, Sequence

from astropy.table import Table


def tabulate(rows: Sequence[Any], headers: List[str]) -> Table:
    # a drop-in replacement for tabulate.tabulate
    data = {name: [row[i] for row in rows] for i, name in enumerate(headers)}
    return Table(data)

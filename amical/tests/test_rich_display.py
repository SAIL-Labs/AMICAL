from textwrap import dedent

from amical._rich_display import tabulate


def test_tabulate():
    data = [["1", 2, 34], ["5", 6, 78]]
    headers = ["name", "id", "adress"]
    table = tabulate(data, headers)
    expected = dedent(
        """
        name  id adress
        ---- --- ------
           1   2     34
           5   6     78
        """
    ).strip()
    assert str(table) == expected

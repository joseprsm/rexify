from typing import Any, List


def flatten(xss: List[List[Any]]):
    return [x for xs in xss for x in xs]

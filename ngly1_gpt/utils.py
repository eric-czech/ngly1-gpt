import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Hashable
from typing import Sequence
from typing import TypeVar

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


@dataclass
class Paths:
    raw_data: Path
    extract_data: Path
    output_data: Path
    prompts: Path


def get_paths() -> Paths:
    return Paths(
        raw_data=Path(__file__).parent.parent / "data" / "raw",
        extract_data=Path(__file__).parent.parent / "data" / "extract",
        output_data=Path(__file__).parent.parent / "data" / "output",
        prompts=Path(__file__).parent / "resources" / "prompts",
    )


paths = get_paths()

NGLY1_DEFICIENCY = "NGLY1 deficiency"


def get_colormap(values: Sequence[Hashable], cmap: str) -> dict[Hashable, str]:
    """Map any sequence of values (arbitrarily) to hex colors in a given matplotlib colorscale."""
    colormap = plt.get_cmap(cmap)
    colors = itertools.cycle(colormap(i) for i in range(colormap.N))
    res = {}
    for string in values:
        if string not in res:
            rgba_color = next(colors)
            hex_color = mcolors.rgb2hex(rgba_color)
            res[string] = hex_color
    return res


T = TypeVar("T")


def apply(obj: T, fn: Callable[[T], None]) -> T:
    fn(obj)
    return obj

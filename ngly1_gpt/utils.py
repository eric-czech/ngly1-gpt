from dataclasses import dataclass
from pathlib import Path
from typing import Hashable
from typing import Sequence


@dataclass
class Paths:
    raw_data: Path
    prompts: Path


def get_paths() -> Paths:
    return Paths(
        raw_data=Path(__file__).parent.parent / "data" / "raw",
        prompts=Path(__file__).parent / "resources" / "prompts",
    )


paths = get_paths()

NGLY1_DEFICIENCY = "NGLY1 deficiency"


def get_colormap(values: Sequence[Hashable], cmap: str) -> dict[Hashable, str]:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import itertools

    colormap = plt.get_cmap(cmap)
    colors = itertools.cycle(colormap(i) for i in range(colormap.N))
    res = {}
    for string in values:
        if string not in res:
            rgba_color = next(colors)
            hex_color = mcolors.rgb2hex(rgba_color)
            res[string] = hex_color
    return res

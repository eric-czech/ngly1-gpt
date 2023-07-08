from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    raw_data: Path


def get_paths() -> Paths:
    return Paths(raw_data=Path(__file__).parent.parent / "data" / "raw")

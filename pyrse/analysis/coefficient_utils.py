from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CoefficientSample:
    coefficient: float
    parameters: Dict[str, float]
    weight: float = 1.0
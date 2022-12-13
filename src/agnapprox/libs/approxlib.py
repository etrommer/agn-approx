from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ApproximateMultiplier:
    name: str
    performance_metric: float
    error_map: np.ndarray


class ApproxLibrary(ABC):
    @abstractmethod
    def load_lut(self, name: str) -> np.ndarray:
        """
        Load LUT for a given Approximate Multiplier

        Args:
            name: Multiplier Name
        """

    @abstractmethod
    def search_space(self) -> List[ApproximateMultiplier]:
        """
        Generate a Search Space of Approximate Multipliers

        Returns:
            List of Candidate Approximate Mulitplier
            in the Search Space
        """

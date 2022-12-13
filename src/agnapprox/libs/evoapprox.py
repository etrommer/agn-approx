from typing import List

import numpy as np
import torchapprox.utils.evoapprox as evo

from agnapprox.libs.approxlib import ApproximateMultiplier, ApproxLibrary


class EvoApprox(ApproxLibrary):
    def __init__(self, filter_str: str = "mul8s", metric: str = "PDK45_PWR"):
        self.names = evo.module_names(filter_str)
        self.metric = metric

    def load_lut(self, mul_name: str) -> np.ndarray:
        return evo.lut(mul_name)

    def search_space(self) -> List[ApproximateMultiplier]:
        return [
            ApproximateMultiplier(n, evo.attribute(n, self.metric), evo.error_map(n))
            for n in self.names
        ]

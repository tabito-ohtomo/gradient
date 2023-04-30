from dataclasses import dataclass
from typing import Dict, TypeAlias


@dataclass
class GradientMaster:
    label: str
    gradient: float


@dataclass
class GradientInformation:
    master: GradientMaster
    number: int


GradientMasterVector: TypeAlias = Dict[str, GradientMaster]
GradientInformationVector: TypeAlias = Dict[str, GradientInformation]
PartitionTrialVector: TypeAlias = Dict[str, float]


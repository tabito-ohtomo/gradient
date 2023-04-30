from math import floor
from typing import Dict

from project.domain.domain import GradientMasterVector, GradientInformationVector, GradientInformation


def round_to_given_unit(x: float, unit: int) -> int:
    return floor(x // unit) * unit


def mod_with_unit(numerator: int, denominator: int, unit: int) -> int:
    if denominator == 1:
        return numerator % unit
    divided = round_to_given_unit(numerator / denominator, unit)
    return numerator - divided


def create_gradient_information_vector(gradient_master_vector: GradientMasterVector, humans_to_label: Dict[str, str]) -> GradientInformationVector:
    gradient_information_vector: GradientInformationVector = {}
    for label in gradient_master_vector.keys():
        gradient_information_vector[label] = GradientInformation(gradient_master_vector[label], len(list(filter(lambda x: x == label, humans_to_label.values()))))
    return gradient_information_vector

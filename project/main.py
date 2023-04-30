from dataclasses import dataclass
from math import floor
from typing import Dict, TypeAlias

import numpy as np
from nptyping import NDArray, Shape, Floating


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
PartitionTrialVector: TypeAlias = Dict[str, int]


def create_gradient_information_vector(gradient_master_vector: GradientMasterVector, humans_to_label: Dict[str, str]) -> GradientInformationVector:
    gradient_information_vector: GradientInformationVector = {}
    for label in gradient_master_vector.keys():
        gradient_information_vector[label] = GradientInformation(gradient_master_vector[label], len(list(filter(lambda x: x == label, humans_to_label.values()))))
    return gradient_information_vector


def product(partition_trial_vector: PartitionTrialVector, gradient_info_vector: GradientInformationVector) -> int:
    return np.dot(np.transpose(list(partition_trial_vector.values())), list(map(lambda x: x.number, gradient_info_vector.values())))


def main():
    # inputs
    print("Hello World!")
    amount: int = 9000
    partition_minimum_unit: int = 1000
    gradient_master: Dict[str, GradientMaster] = {"1": GradientMaster("1", 1), "2": GradientMaster("2", 2)}
    human_label: Dict[str, str] = {"AAA": "1", "BBB": "1", "CCC": "2"}

    # process
    gradient_informations: GradientInformationVector = create_gradient_information_vector(gradient_master, human_label)

    total_partition_unit: float = sum(map(lambda x: x.number * x.master.gradient, gradient_informations.values()))
    print(gradient_informations)
    ideal_partition_vector: PartitionTrialVector = get_ideal_partition_vector(amount, gradient_informations)

    # here is the core of to define the gradient

    partition_current_result: PartitionTrialVector = optimize_by_simulated_annealing(amount, ideal_partition_vector, gradient_informations, partition_minimum_unit)


def get_ideal_partition_vector(amount: int, gradient_informations: GradientInformationVector) -> PartitionTrialVector:
    ideal_unit = amount / sum(map(lambda x: x.number * x.master.gradient, gradient_informations.values()))

    ideal_partition_vector: PartitionTrialVector = {}
    for label, gradient_information in gradient_informations.items():
        ideal_partition_vector[label] = gradient_information.master.gradient * ideal_unit
    return ideal_partition_vector


def optimize_by_simulated_annealing(amount: int, ideal_partition_vector: PartitionTrialVector, gradient_informations: GradientInformationVector, partition_minimum_unit: int) -> PartitionTrialVector:
    ideal_unit = amount / sum(map(lambda x: x.number * x.master.gradient, gradient_informations.values()))
    coordinated_unit: int = round_to_given_unit(ideal_unit, partition_minimum_unit)

    partition_current_result: PartitionTrialVector = {}
    for label, gradient_information in gradient_informations.items():
        partition_current_result[label] = round_to_given_unit(gradient_information.master.gradient * coordinated_unit, partition_minimum_unit)

    current_total_amount: int = product(partition_current_result, gradient_informations)
    print(current_total_amount)

    counter: int = 0
    while amount > current_total_amount and counter < 10:
        remaining: int = amount - current_total_amount
        for label, gradient_information in gradient_informations.items():
            if mod_with_unit(remaining, gradient_information.number, partition_minimum_unit) == 0:
                partition_current_result[label] += remaining // gradient_information.number
                current_total_amount += remaining
                break
        counter += 1

    print(calculate_partition_reasonability(
        amount,
        ideal_partition_vector,
        np.array(list(partition_current_result.values()))
    ))

    return partition_current_result


def calculate_partition_reasonability(amount: int, ideal_gradient: NDArray[Shape["1, *"], Floating], partitioned_amounts: NDArray[Shape["1, *"], Floating]) -> float:
    print(ideal_gradient)
    print(partitioned_amounts)
    return np.linalg.norm(np.subtract(np.multiply(1 / amount, ideal_gradient), np.multiply(1 / amount, partitioned_amounts)))


def round_to_given_unit(x: float, unit: int) -> int:
    return floor(x // unit) * unit


def mod_with_unit(numerator: int, denominator: int, unit: int) -> int:
    divided = round_to_given_unit(numerator / denominator, unit)
    return numerator - divided


if __name__ == '__main__':
    main()

from dataclasses import dataclass
from math import floor
from typing import Dict

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


def main():
    # inputs
    print("Hello World!")
    amount: int = 9000
    partition_minimum_unit: int = 1000
    gradient_master: Dict[str, GradientMaster] = {"1": GradientMaster("1", 1), "2": GradientMaster("2", 2)}
    human_label: Dict[str, str] = {"AAA": "1", "BBB": "1", "CCC": "2"}

    # process
    partition_information: Dict[str, GradientInformation] = {}
    for label in gradient_master.keys():
        partition_information[label] = GradientInformation(gradient_master[label], len(list(filter(lambda x: x == label, human_label.values()))))

    total_partition_unit: float = sum(map(lambda x: x.number * x.master.gradient, partition_information.values()))
    ideal_unit: float = amount / total_partition_unit
    print(partition_information)
    ideal_gradient: NDArray[float] = np.multiply(ideal_unit, np.array(list(map(lambda x: x.master.gradient, partition_information.values()))))

    # here is the core of to define the gradient
    coordinated_unit: int = round_to_given_unit(ideal_unit, partition_minimum_unit)

    partition_current_result: Dict[str, int] = {}
    for label, gradient_information in partition_information.items():
        partition_current_result[label] = round_to_given_unit(gradient_information.master.gradient * coordinated_unit, partition_minimum_unit)

    current_total_amount: int = np.dot(np.transpose(list(partition_current_result.values())), list(map(lambda x: x.number, partition_information.values())))
    print(current_total_amount)

    counter: int = 0
    while amount > current_total_amount and counter < 10:
        remaining: int = amount - current_total_amount
        for label, gradient_information in partition_information.items():
            if mod_with_unit(remaining, gradient_information.number, partition_minimum_unit) == 0:
                partition_current_result[label] += remaining // gradient_information.number
                current_total_amount += remaining
                break
        counter += 1

    print(calculate_partition_reasonability(
        amount,
        ideal_gradient,
        np.array(list(partition_current_result.values()))
    ))


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

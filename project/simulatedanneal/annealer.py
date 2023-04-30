import numpy as np
from nptyping import Shape, NDArray, Floating

from project.domain.domain import GradientInformationVector, PartitionTrialVector
from project.domain.util import round_to_given_unit, mod_with_unit


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
        np.array(list(ideal_partition_vector.values())),
        np.array(list(partition_current_result.values()))
    ))

    return partition_current_result


def calculate_partition_reasonability(amount: int, ideal_partition: NDArray[Shape["1, *"], Floating], partitioned_amounts: NDArray[Shape["1, *"], Floating]) -> float:
    print(ideal_partition)
    print(partitioned_amounts)
    return np.linalg.norm(np.subtract(np.multiply(1 / amount, ideal_partition), np.multiply(1 / amount, partitioned_amounts)))


def get_ideal_partition_vector(amount: int, gradient_informations: GradientInformationVector) -> PartitionTrialVector:
    ideal_unit = amount / sum(map(lambda x: x.number * x.master.gradient, gradient_informations.values()))

    ideal_partition_vector: PartitionTrialVector = {}
    for label, gradient_information in gradient_informations.items():
        ideal_partition_vector[label] = gradient_information.master.gradient * ideal_unit
    return ideal_partition_vector


def product(partition_trial_vector: PartitionTrialVector, gradient_info_vector: GradientInformationVector) -> int:
    return np.dot(np.transpose(list(partition_trial_vector.values())), list(map(lambda x: x.number, gradient_info_vector.values())))

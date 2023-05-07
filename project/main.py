import os
from typing import Dict

import importlib.util
import sys

from project.domain.domain import GradientInformationVector, GradientMasterVector, PartitionTrialVector, GradientMaster
from project.domain.util import create_gradient_information_vector
from project.quantum.qannealer import optimize_by_quantum_annealing
from project.simulatedanneal.annealer import optimize_by_simulated_annealing
import resource

def main():
    # resource.setrlimit(resource.RLIMIT_DATA, (16 * 1024**3, -1))
    print(os.environ['DWAVE_TOKEN']) # conf file should be great

    # inputs
    print("Hello World!")

    # amount: int = 8000
    # partition_minimum_unit: int = 1000
    # gradient_master: GradientMasterVector = {"1": GradientMaster("1", 1), "2": GradientMaster("2", 2)}
    # human_label: Dict[str, str] = {"A": "1", "B": "1", "F": "2"}
    #
    amount: int = 100000
    # 7200 : 12000: 16000 ~ 3: 5: 7
    partition_minimum_unit: int = 1000
    gradient_master: GradientMasterVector = {"1": GradientMaster("1", 3), "2": GradientMaster("2", 5), "3": GradientMaster("3", 7)}
    human_label: Dict[str, str] = {"A": "1", "B": "1", "C": "1", "D": "1", "E": "1", "F": "2", "G": "2", "H": "2", "I": "2", "J": "3"}
    #
    # process
    gradient_informations: GradientInformationVector = create_gradient_information_vector(gradient_master, human_label)

    total_partition_unit: float = sum(map(lambda x: x.number * x.master.gradient, gradient_informations.values()))
    print(gradient_informations)
    # ideal_partition_vector: PartitionTrialVector = get_ideal_partition_vector(amount, gradient_informations)
    ideal_partition_vector: PartitionTrialVector = get_ideal_partition_vector(gradient_informations)

    # here is the core of to define the gradient
    partition_current_result: PartitionTrialVector = optimize_by_quantum_annealing(amount, ideal_partition_vector, gradient_informations, partition_minimum_unit)
    print(partition_current_result)


# def get_ideal_partition_vector(amount: int, gradient_informations: GradientInformationVector) -> PartitionTrialVector:
#     ideal_unit = amount / sum(map(lambda x: x.number * x.master.gradient, gradient_informations.values()))
#
#     ideal_partition_vector: PartitionTrialVector = {}
#     for label, gradient_information in gradient_informations.items():
#         ideal_partition_vector[label] = gradient_information.master.gradient * ideal_unit
#     return ideal_partition_vector
#
def get_ideal_partition_vector(gradient_informations: GradientInformationVector) -> PartitionTrialVector:
    ideal_partition_vector: PartitionTrialVector = {}
    for label, gradient_information in gradient_informations.items():
        ideal_partition_vector[label] = gradient_information.master.gradient
    return ideal_partition_vector



if __name__ == '__main__':
    main()

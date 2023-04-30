from typing import Dict

from project.domain.domain import GradientInformationVector, GradientMasterVector, PartitionTrialVector, GradientMaster
from project.domain.util import create_gradient_information_vector
from project.simulatedanneal.annealer import get_ideal_partition_vector, optimize_by_simulated_annealing


def main():
    # inputs
    print("Hello World!")
    amount: int = 9000
    partition_minimum_unit: int = 1000
    gradient_master: GradientMasterVector = {"1": GradientMaster("1", 1), "2": GradientMaster("2", 2)}
    human_label: Dict[str, str] = {"AAA": "1", "BBB": "1", "CCC": "2"}

    # process
    gradient_informations: GradientInformationVector = create_gradient_information_vector(gradient_master, human_label)

    total_partition_unit: float = sum(map(lambda x: x.number * x.master.gradient, gradient_informations.values()))
    print(gradient_informations)
    ideal_partition_vector: PartitionTrialVector = get_ideal_partition_vector(amount, gradient_informations)

    # here is the core of to define the gradient
    partition_current_result: PartitionTrialVector = optimize_by_simulated_annealing(amount, ideal_partition_vector, gradient_informations, partition_minimum_unit)
    print(partition_current_result)


if __name__ == '__main__':
    main()

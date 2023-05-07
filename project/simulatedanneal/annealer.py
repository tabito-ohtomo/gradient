import numpy as np
from nptyping import Shape, NDArray, Floating
from simanneal import Annealer

from project.domain.domain import GradientInformationVector, PartitionTrialVector
from project.domain.util import round_to_given_unit, mod_with_unit


class PartitionProblem(Annealer):
    def __init__(self, ideal_partition_vector: PartitionTrialVector, gradient_informations: GradientInformationVector, amount: int, partition_minimum_unit: int):
        self.ideal_partition_vector = ideal_partition_vector
        self.gradient_informations = gradient_informations
        self.amount = amount
        self.partition_minimum_unit = partition_minimum_unit
        super().__init__(ideal_partition_vector)  # use ideal partition vector as the initial state

    def move(self):
        decrease_label = np.random.choice(list(self.state.keys()))
        increase_label = np.random.choice(list(self.state.keys()))

        # align to integer
        nearest_floor = random_zero_to_floors_with_unit(self.state[decrease_label], self.partition_minimum_unit)
        how_many_decrease = (self.state[decrease_label] - nearest_floor)

        amount_delta = how_many_decrease * self.gradient_informations[decrease_label].number

        how_many_increase = amount_delta // self.gradient_informations[increase_label].number

        self.state[decrease_label] -= how_many_decrease
        self.state[increase_label] += how_many_increase

    def energy(self) -> float:
        reasonability: float = calculate_partition_reasonability(
            self.amount,
            np.array(list(self.ideal_partition_vector.values())),
            np.array(list(self.state.values()))
        )
        division_penalty: float = sum(map(lambda x: mod_with_unit(x, 1, self.partition_minimum_unit), self.state.values()))
        total_penalty: float = min(product(self.state, self.gradient_informations) - self.amount, 0)
        return reasonability + division_penalty + total_penalty

def random_zero_to_floors_with_unit(number: int, unit: int):
    div = number // unit
    return np.random.randint(0, div + 1) * unit

def optimize_by_simulated_annealing(amount: int, ideal_partition_vector: PartitionTrialVector, gradient_informations: GradientInformationVector, partition_minimum_unit: int) -> PartitionTrialVector:
    problem = PartitionProblem(ideal_partition_vector, gradient_informations, amount, partition_minimum_unit)
    # auto_schedule = problem.auto(minutes=1)
    # problem.set_schedule(auto_schedule)
    trial_vector, energy = problem.anneal()
    print(trial_vector)
    print(energy)
    return trial_vector

def calculate_partition_reasonability(amount: int, ideal_partition: NDArray[Shape["1, *"], Floating], partitioned_amounts: NDArray[Shape["1, *"], Floating]) -> float:
    print(ideal_partition)
    print(partitioned_amounts)
    return np.linalg.norm(np.subtract(np.multiply(1 / amount, ideal_partition), np.multiply(1 / amount, partitioned_amounts)))


def product(partition_trial_vector: PartitionTrialVector, gradient_info_vector: GradientInformationVector) -> int:
    return np.dot(np.transpose(list(partition_trial_vector.values())), list(map(lambda x: x.number, gradient_info_vector.values())))

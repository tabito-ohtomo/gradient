import os
from math import ceil

import dimod
from typing import Dict
from dwave.system import LeapHybridCQMSampler

from project.domain.domain import GradientInformationVector, PartitionTrialVector


def optimize_by_quantum_annealing(amount: int, ideal_partition_vector: PartitionTrialVector,
                                  gradient_informations: GradientInformationVector,
                                  partition_minimum_unit: int) -> PartitionTrialVector:
    aligned_amount = ceil(amount // partition_minimum_unit)

    normalizing_label = list(ideal_partition_vector.keys())[0]
    print('normalizing label: ' + normalizing_label)

    normalized_ideal_partition_vector = {}
    for label in ideal_partition_vector.keys():
        normalized_ideal_partition_vector[label] = ideal_partition_vector[label] / ideal_partition_vector[
            normalizing_label]
    print('normalized_ideal_partition_vector: ' + str(normalized_ideal_partition_vector))

    cqm = dimod.ConstrainedQuadraticModel()
    obj = dimod.QuadraticModel()
    constraint = dimod.QuadraticModel()

    for label, gradient_information in gradient_informations.items():
        obj.add_variable('INTEGER', label)
        obj.set_quadratic(label, label, 1)
        if label == normalizing_label:
            obj.set_quadratic(label, label, 1 - 2 * normalized_ideal_partition_vector[label] + norm_square(normalized_ideal_partition_vector))
        else:
            obj.set_quadratic(label, normalizing_label, -2 * normalized_ideal_partition_vector[label])

    for label, gradient_information in gradient_informations.items():
        constraint.add_variable('INTEGER', label)
        constraint.set_linear(label, gradient_information.number)

    cqm.set_objective(obj)
    cqm.add_constraint(constraint, sense='==', rhs=aligned_amount)

    for label in gradient_informations.keys():
        positive_constraint = dimod.QuadraticModel()
        positive_constraint.add_variable('INTEGER', label)
        positive_constraint.set_linear(label, 1)
        cqm.add_constraint(positive_constraint, sense='>=', rhs=0)

    sampler = LeapHybridCQMSampler(token=os.environ['DWAVE_TOKEN'])
    # sampleset = sampler.sample_cqm(cqm, atol=1e-1, label='annealing')
    sampleset = sampler.sample_cqm(cqm, label='annealing')

    print(sampleset)

    parse_solution(sampleset)

    return None


def norm_square(vector: Dict[str, float]) -> float:
    return sum([value ** 2 for value in vector.values()])

def parse_solution(sampleset):
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    if not len(feasible_sampleset):
        raise ValueError("No feasible solution found")

    best = feasible_sampleset.first

    print(best)

    # print("\nFound best solution at energy {}".format(best.energy))

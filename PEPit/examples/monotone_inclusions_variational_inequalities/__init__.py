from .accelerated_proximal_point import wc_accelerated_proximal_point
from .douglas_rachford_splitting import wc_douglas_rachford_splitting
from .douglas_rachford_splitting_2 import wc_douglas_rachford_splitting_2
from .optimal_strongly_monotone_proximal_point import wc_optimal_strongly_monotone_proximal_point
from .optimistic_gradient import wc_optimistic_gradient
from .past_extragradient import wc_past_extragradient
from .proximal_point import wc_proximal_point
from .three_operator_splitting import wc_three_operator_splitting

__all__ = ['accelerated_proximal_point', 'wc_accelerated_proximal_point',
           'douglas_rachford_splitting', 'wc_douglas_rachford_splitting',
           'douglas_rachford_splitting_2', 'wc_douglas_rachford_splitting_2',
           'optimal_strongly_monotone_proximal_point', 'wc_optimal_strongly_monotone_proximal_point',
           'optimistic_gradient', 'wc_optimistic_gradient',
           'past_extragradient', 'wc_past_extragradient',
           'proximal_point', 'wc_proximal_point',
           'three_operator_splitting', 'wc_three_operator_splitting',
           ]

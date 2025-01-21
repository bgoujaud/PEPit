import unittest

import numpy as np

from PEPit.examples.unconstrained_convex_minimization import wc_conjugate_gradient
from PEPit.examples.unconstrained_convex_minimization import wc_conjugate_gradient_qg_convex
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent_lc
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent_qg_convex
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent_qg_convex_decreasing
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent_quadratics
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent_silver_stepsize_convex
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent_silver_stepsize_strongly_convex
from PEPit.examples.unconstrained_convex_minimization import wc_subgradient_method_rsi_eb
from PEPit.examples.unconstrained_convex_minimization import wc_accelerated_gradient_convex
from PEPit.examples.unconstrained_convex_minimization import wc_accelerated_gradient_strongly_convex
from PEPit.examples.unconstrained_convex_minimization import wc_accelerated_proximal_point
from PEPit.examples.unconstrained_convex_minimization import wc_proximal_point
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_exact_line_search
from PEPit.examples.unconstrained_convex_minimization import wc_heavy_ball_momentum
from PEPit.examples.unconstrained_convex_minimization import wc_heavy_ball_momentum_qg_convex
from PEPit.examples.unconstrained_convex_minimization import wc_inexact_accelerated_gradient
from PEPit.examples.unconstrained_convex_minimization import wc_inexact_gradient_descent
from PEPit.examples.unconstrained_convex_minimization import wc_inexact_gradient_exact_line_search
from PEPit.examples.unconstrained_convex_minimization import wc_optimized_gradient
from PEPit.examples.unconstrained_convex_minimization import wc_robust_momentum
from PEPit.examples.unconstrained_convex_minimization import wc_subgradient_method
from PEPit.examples.unconstrained_convex_minimization import wc_triple_momentum
from PEPit.examples.unconstrained_convex_minimization import wc_information_theoretic
from PEPit.examples.unconstrained_convex_minimization import wc_optimized_gradient_for_gradient
from PEPit.examples.unconstrained_convex_minimization import wc_epsilon_subgradient_method
from PEPit.examples.unconstrained_convex_minimization import wc_cyclic_coordinate_descent
from PEPit.examples.composite_convex_minimization import wc_accelerated_douglas_rachford_splitting
from PEPit.examples.composite_convex_minimization import wc_accelerated_proximal_gradient
from PEPit.examples.composite_convex_minimization import wc_bregman_proximal_point
from PEPit.examples.composite_convex_minimization import wc_frank_wolfe
from PEPit.examples.composite_convex_minimization import wc_douglas_rachford_splitting
from PEPit.examples.composite_convex_minimization import wc_douglas_rachford_splitting_contraction
from PEPit.examples.composite_convex_minimization import wc_improved_interior_algorithm
from PEPit.examples.composite_convex_minimization import wc_no_lips_in_bregman_divergence
from PEPit.examples.composite_convex_minimization import wc_no_lips_in_function_value
from PEPit.examples.composite_convex_minimization import wc_proximal_gradient
from PEPit.examples.composite_convex_minimization import wc_proximal_gradient_quadratics
from PEPit.examples.composite_convex_minimization import wc_three_operator_splitting
from PEPit.examples.nonconvex_optimization import wc_gradient_descent as wc_gradient_descent_non_convex
from PEPit.examples.nonconvex_optimization import wc_no_lips_1
from PEPit.examples.nonconvex_optimization import wc_no_lips_2
from PEPit.examples.stochastic_and_randomized_convex_minimization import wc_saga
from PEPit.examples.stochastic_and_randomized_convex_minimization import wc_sgd_overparametrized
from PEPit.examples.stochastic_and_randomized_convex_minimization import wc_sgd
from PEPit.examples.stochastic_and_randomized_convex_minimization import wc_point_saga
from PEPit.examples.stochastic_and_randomized_convex_minimization import \
    wc_randomized_coordinate_descent_smooth_strongly_convex
from PEPit.examples.stochastic_and_randomized_convex_minimization import wc_randomized_coordinate_descent_smooth_convex
from PEPit.examples.monotone_inclusions_variational_inequalities import \
    wc_accelerated_proximal_point as wc_accelerated_proximal_point_operators
from PEPit.examples.monotone_inclusions_variational_inequalities import \
    wc_douglas_rachford_splitting as wc_douglas_rachford_splitting_operators
from PEPit.examples.monotone_inclusions_variational_inequalities import \
    wc_douglas_rachford_splitting_2 as wc_douglas_rachford_splitting_operators_2
from PEPit.examples.monotone_inclusions_variational_inequalities import wc_optimal_strongly_monotone_proximal_point as \
    wc_optimal_strongly_monotone_proximal_point_operators
from PEPit.examples.monotone_inclusions_variational_inequalities import \
    wc_proximal_point as wc_proximal_point_method_operators
from PEPit.examples.monotone_inclusions_variational_inequalities import \
    wc_three_operator_splitting as wc_three_operator_splitting_operators
from PEPit.examples.monotone_inclusions_variational_inequalities import \
    wc_optimistic_gradient as wc_optimistic_gradient_operators
from PEPit.examples.monotone_inclusions_variational_inequalities import \
    wc_past_extragradient as wc_past_extragradient_operators
from PEPit.examples.fixed_point_problems import wc_halpern_iteration
from PEPit.examples.fixed_point_problems import wc_krasnoselskii_mann_constant_step_sizes
from PEPit.examples.fixed_point_problems import wc_krasnoselskii_mann_increasing_step_sizes
from PEPit.examples.fixed_point_problems import wc_inconsistent_halpern_iteration
from PEPit.examples.fixed_point_problems import wc_optimal_contractive_halpern_iteration
from PEPit.examples.potential_functions import wc_accelerated_gradient_method
from PEPit.examples.potential_functions import wc_gradient_descent_lyapunov_1
from PEPit.examples.potential_functions import wc_gradient_descent_lyapunov_2
from PEPit.examples.adaptive_methods import wc_polyak_steps_in_distance_to_optimum
from PEPit.examples.adaptive_methods import wc_polyak_steps_in_function_value
from PEPit.examples.low_dimensional_worst_cases_scenarios import wc_inexact_gradient as wc_inexact_gradient_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import wc_optimized_gradient as wc_optimized_gradient_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import wc_frank_wolfe as wc_frank_wolfe_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import wc_proximal_point as wc_proximal_point_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import wc_halpern_iteration as wc_halpern_iteration_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import \
    wc_gradient_descent as wc_gradient_descent_non_convex_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import \
    wc_alternate_projections as wc_alternate_projections_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import \
    wc_averaged_projections as wc_averaged_projections_low_dim
from PEPit.examples.low_dimensional_worst_cases_scenarios import wc_dykstra as wc_dykstra_low_dim
from PEPit.examples.inexact_proximal_methods import wc_accelerated_inexact_forward_backward
from PEPit.examples.inexact_proximal_methods import wc_partially_inexact_douglas_rachford_splitting
from PEPit.examples.inexact_proximal_methods import wc_relatively_inexact_proximal_point_algorithm
from PEPit.examples.continuous_time_models import wc_accelerated_gradient_flow_convex
from PEPit.examples.continuous_time_models import wc_gradient_flow_convex
from PEPit.examples.continuous_time_models import wc_accelerated_gradient_flow_strongly_convex
from PEPit.examples.continuous_time_models import wc_gradient_flow_strongly_convex
from PEPit.examples.tutorials import wc_gradient_descent_contraction


class TestExamplesCVXPY(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapper = "cvxpy"

    def setUp(self):
        self.n = 6
        self.mu = .1
        self.L = 1
        self.verbose = -1
        self.relative_precision = 10 ** -3
        self.absolute_precision = 5 * 10 ** -5

    def test_optimized_gradient(self):
        L, n = 3, 4

        wc, theory = wc_optimized_gradient(L, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_optimized_gradient_for_gradient(self):
        L, n = 3, 4

        wc, theory = wc_optimized_gradient_for_gradient(L, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_epsilon_subgradient_method(self):
        M, n, eps, R = 2, 6, 2, 1
        gamma = 1 / (np.sqrt(n + 1))

        wc, theory = wc_epsilon_subgradient_method(M=M, n=n, gamma=gamma, eps=eps, R=R, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_information_theoretic(self):
        mu, L, n = .01, 3, 3

        wc, theory = wc_information_theoretic(mu=mu, L=L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_gradient_descent(self):
        L, n = 3, 4
        gamma = 1 / L

        wc, theory = wc_gradient_descent(L=L, gamma=gamma, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_gradient_descent_lc(self):
        Lg, mug = 3, .3
        LM, muM = 1., 0.1
        gamma, n = 1 / (Lg * LM ** 2), 3
        for typeM in ["gen", "sym", "skew"]:

            wc, theory = wc_gradient_descent_lc(mug=mug, Lg=Lg,
                                                typeM=typeM, muM=muM, LM=LM,
                                                gamma=gamma, n=n,
                                                verbose=self.verbose)

            self.assertAlmostEqual(wc, theory, delta=2*self.relative_precision * theory)

    def test_gradient_descent_quadratics(self):
        L, mu, n = 3, .3, 4
        gamma = 1/L
        wc, theory = wc_gradient_descent_quadratics(mu=mu, L=L, gamma=gamma, n=n,
                                                    wrapper=self.wrapper, verbose=self.verbose)

        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_gradient_descent_silver_stepsize_convex(self):
        L, n = 2.8, 2
        wc, theory = wc_gradient_descent_silver_stepsize_convex(L=L, n=n,
                                                                verbose=self.verbose)

        self.assertLessEqual(wc, theory)

    def test_gradient_descent_silver_stepsize_strongly_convex(self):
        L, mu, n = 3.2, .1, 17
        wc, theory = wc_gradient_descent_silver_stepsize_strongly_convex(L=L, mu=mu, n=n,
                                                                         verbose=self.verbose)

        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_cyclic_coordinate_descent_one_block(self):
        n = 9
        L = 1.

        wc, _ = wc_cyclic_coordinate_descent(L=[L], n=n, wrapper=self.wrapper, verbose=self.verbose)
        wc_GD, _ = wc_gradient_descent(L, 1 / L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, wc_GD, delta=self.relative_precision * wc_GD)

    def test_cyclic_coordinate_descent(self):
        n = 9
        L = [1., 2., 10.]

        wc, _ = wc_cyclic_coordinate_descent(L=L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, 1.48928, delta=self.relative_precision * 1.48928)

    def test_gradient_descent_qg_convex(self):
        L, n = 1, 4
        gamma = .1 / L

        wc, theory = wc_gradient_descent_qg_convex(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_gradient_descent_qg_convex_decreasing(self):
        L, n = 1, 4

        wc, theory = wc_gradient_descent_qg_convex_decreasing(L, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_gradient_exact_line_search(self):
        L, mu, n = 3, .1, 1

        wc, theory = wc_gradient_exact_line_search(L=L, mu=mu, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_subgradient_method(self):
        M, n = 2, 10
        gamma = 1 / (np.sqrt(n + 1) * M)

        wc, theory = wc_subgradient_method(M=M, n=n, gamma=gamma, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_subgradient_method_rsi_eb(self):
        mu = .1
        L = 1
        gamma = mu / L ** 2
        n = 4
        wc, theory = wc_subgradient_method_rsi_eb(mu=mu, L=L, gamma=gamma, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_conjugate_gradient(self):
        L, n = 3, 2

        wc, theory = wc_conjugate_gradient(L=L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_conjugate_gradient_qg_convex(self):
        L, n = 3.5, 12

        wc, theory = wc_conjugate_gradient_qg_convex(L=L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_gradient_exact_line_search(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = wc_inexact_gradient_exact_line_search(L=L, mu=mu, epsilon=epsilon, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_gradient_descent(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = wc_inexact_gradient_descent(L=L, mu=mu, epsilon=epsilon, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_proximal_point(self):
        n, gamma = 3, .1

        wc, theory = wc_proximal_point(gamma=gamma, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_optimized_gradient_method(self):
        L, n = 3, 4

        wc, theory = wc_optimized_gradient_low_dim(L, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_gradient(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = wc_inexact_gradient_low_dim(L=L, mu=mu, epsilon=epsilon, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_frank_wolfe_low_dim(self):
        D, L, n = 1., 1., 10

        wc, theory = wc_frank_wolfe_low_dim(L, D, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_proximal_point_low_dim(self):
        n, alpha = 11, 2.2

        wc, theory = wc_proximal_point_low_dim(alpha=alpha, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_halpern_iteration_low_dim(self):
        n = 15

        wc, theory = wc_halpern_iteration_low_dim(n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=2*self.relative_precision * theory)

    def test_gradient_descent_non_convex_low_dim(self):
        L, n = 1, 5
        gamma = 1 / L

        wc, theory = wc_gradient_descent_non_convex_low_dim(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_alternate_projections_low_dim(self):
        n1 = 9
        n2 = 10

        wc1, _ = wc_alternate_projections_low_dim(n1, wrapper=self.wrapper, verbose=self.verbose)
        wc2, _ = wc_alternate_projections_low_dim(n2, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc2, wc1)

    def test_averaged_projections_low_dim(self):
        n1 = 10
        n2 = 11

        wc1, _ = wc_averaged_projections_low_dim(n1, wrapper=self.wrapper, verbose=self.verbose)
        wc2, _ = wc_averaged_projections_low_dim(n2, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc2, wc1)

    def test_dykstra_low_dim(self):
        n1 = 8
        n2 = 10

        wc1, _ = wc_dykstra_low_dim(n1, wrapper=self.wrapper, verbose=self.verbose)
        wc2, _ = wc_dykstra_low_dim(n2, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc2, wc1)

    def test_inexact_accelerated_gradient_1(self):
        L, epsilon, n = 3, 0, 5

        wc, theory = wc_inexact_accelerated_gradient(L=L, epsilon=epsilon, n=n, wrapper=self.wrapper, verbose=self.verbose)

        # Less accurate requirement due to ill conditioning of this specific SDP (no Slater point)
        local_relative_precision = 10 ** -2
        self.assertAlmostEqual(theory, wc, delta=local_relative_precision * theory)

    def test_inexact_accelerated_gradient_2(self):
        L, epsilon, n = 2, .01, 5

        wc, theory = wc_inexact_accelerated_gradient(L=L, epsilon=epsilon, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(theory, wc * (1 + self.relative_precision))

    def test_inexact_accelerated_gradient_3(self):
        L, epsilon, n = 2, .1, 5

        wc, theory = wc_inexact_accelerated_gradient(L=L, epsilon=epsilon, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(theory, wc * (1 + self.relative_precision))

    def test_heavy_ball_momentum(self):
        L, mu, n = 1, .1, 3
        alpha = 1 / (2 * L)  # alpha \in [0, 1/L]
        beta = np.sqrt((1 - alpha * mu) * (1 - L * alpha))

        wc, theory = wc_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory * (1 + self.relative_precision))

    def test_heavy_ball_momentum_qg_convex(self):
        L, n = 1, 5
        wc, theory = wc_heavy_ball_momentum_qg_convex(L=L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory * (1 + self.relative_precision))

    def test_accelerated_proximal_point(self):
        A0, n = 1, 3
        gammas = [1, 1, 1]

        wc, theory = wc_accelerated_proximal_point(A0=A0, gammas=gammas, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory * (1 + self.relative_precision))

    def test_triple_momentum(self):
        L, mu, n = 1, .1, 4

        # Compare theoretical rate in epsilon=0 case
        wc, theory = wc_triple_momentum(mu=mu, L=L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=self.relative_precision * theory)

    def test_robust_momentum(self):
        L, mu, lam = 1, .1, .5

        # Compare theoretical rate in epsilon=0 case
        wc, theory = wc_robust_momentum(mu=mu, L=L, lam=lam, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=self.relative_precision * theory)

    def test_accelerated_gradient_convex(self):
        mu, L, n = 0, 1, 10

        wc, theory = wc_accelerated_gradient_convex(mu, L, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_accelerated_gradient_strongly_convex(self):
        L, mu, n = 1, .1, 5

        wc, theory = wc_accelerated_gradient_strongly_convex(mu=mu, L=L, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_accelerated_proximal_gradient_method(self):
        mu, L, n = 0, 1, 5

        wc, theory = wc_accelerated_proximal_gradient(mu, L, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_accelerated_douglas_rachford_splitting(self):
        mu, L, alpha = 0.1, 1, 0.9

        n_list = range(1, 9)
        ref_pesto_bounds = [0.2027, 0.1929, 0.1839, 0.1737, 0.1627, 0.1514, 0.1400, 0.1289]
        for n in n_list:
            wc, _ = wc_accelerated_douglas_rachford_splitting(mu, L, alpha, n, wrapper=self.wrapper, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=self.relative_precision * ref_pesto_bounds[n - 1])

    def test_bregman_proximal_point_method(self):
        gamma, n = 3, 5

        wc, theory = wc_bregman_proximal_point(gamma=gamma, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_conditional_gradient_frank_wolfe(self):
        D, L, n = 1., 1., 10

        wc, theory = wc_frank_wolfe(L, D, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_douglas_rachford_splitting_contraction(self):
        mu, L, alpha, theta, n = 0.1, 1, 3, 1, 1

        wc, theory = wc_douglas_rachford_splitting_contraction(mu, L, alpha, theta, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_douglas_rachford_splitting(self):
        L, alpha, theta, n = 1, 1, 1, 10

        wc, theory = wc_douglas_rachford_splitting(L, alpha, theta, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_improved_interior_algorithm(self):
        L, mu, c, n = 1, 1, 1, 5
        lam = 1 / L

        wc, theory = wc_improved_interior_algorithm(L, mu, c, lam, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_no_lips_in_bregman_divergence(self):
        L, n = 0.1, 3
        gamma = 1 / L

        wc, theory = wc_no_lips_in_bregman_divergence(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_no_lips_in_function_value(self):
        L, n = 1, 3
        gamma = 1 / L / 2

        wc, theory = wc_no_lips_in_function_value(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_proximal_gradient(self):
        L, mu, gamma, n = 1, .1, 1, 2

        wc, theory = wc_proximal_gradient(L=L, mu=mu, gamma=gamma, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_proximal_gradient_quadratics(self):
        L, mu, gamma, n = 1, .1, 1, 2

        wc, theory = wc_proximal_gradient_quadratics(L=L, mu=mu, gamma=gamma, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_three_operator_splitting(self):
        mu, L1, L3, alpha, theta = 0.1, 10, 1, 1, 1
        n_list = range(1, 4)

        ref_pesto_bounds = [0.8304, 0.6895, 0.5726]
        for n in n_list:
            wc, _ = wc_three_operator_splitting(mu, L1, L3, alpha, theta, n, wrapper=self.wrapper, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=self.relative_precision * ref_pesto_bounds[n - 1])

    def test_gradient_descent_non_convex(self):
        L, n = 1, 5
        gamma = 1 / L

        wc, theory = wc_gradient_descent_non_convex(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_no_lips_1(self):
        L, n = 1, 5
        gamma = 1 / L / 2

        wc, theory = wc_no_lips_1(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_no_lips_2(self):
        L, n = 1, 3
        gamma = 1 / L

        wc, theory = wc_no_lips_2(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_saga(self):
        L, mu, n = 1, 0.1, 5

        wc, theory = wc_saga(L, mu, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_sgd(self):
        L, mu, v, R, n = 1, 0.1, 1, 2, 5
        gamma = 1 / L

        wc, theory = wc_sgd(L, mu, gamma, v, R, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_sgd_overparametrized(self):
        L, mu, n = 1, 0.1, 5
        gamma = 1 / L

        wc, theory = wc_sgd_overparametrized(L, mu, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_point_saga(self):
        L, mu, n = 1, 0.1, 10

        wc, theory = wc_point_saga(L, mu, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_randomized_coordinate(self):
        L, d, t = 1, 3, 10
        gamma = 1 / L

        wc, theory = wc_randomized_coordinate_descent_smooth_convex(L=L, gamma=gamma, d=d, t=t, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_randomized_coordinate_strongly_convex(self):
        L, mu, d = 1, 0.1, 3
        gamma = 2 / (L + mu)

        wc, theory = wc_randomized_coordinate_descent_smooth_strongly_convex(L=L, mu=mu, gamma=gamma, d=d,
                                                                             wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_accelerated_proximal_point_operators(self):
        alpha, n = 2.1, 10

        wc, theory = wc_accelerated_proximal_point_operators(alpha, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_proximal_point_method_operators(self):
        alpha, n = 2.1, 3

        wc, theory = wc_proximal_point_method_operators(alpha, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_wc_optimal_strongly_monotone_proximal_point_operators(self):
        n, mu = 3, 0.23
        wc, theory = wc_optimal_strongly_monotone_proximal_point_operators(n=n, mu=mu, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_douglas_rachford_splitting_operators(self):
        L, mu, alpha, theta = 1, 0.1, 1.3, 0.9

        wc, theory = wc_douglas_rachford_splitting_operators(L, mu, alpha, theta, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_douglas_rachford_splitting_operators_2(self):
        beta, mu, alpha, theta = 1.2, 0.1, 0.3, 1.5

        wc, theory = wc_douglas_rachford_splitting_operators_2(beta, mu, alpha, theta, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_three_operator_splitting_operators(self):
        L, mu, beta, alpha, theta = 1, 0.1, 1, 1.3, 0.9
        n_list = range(1, 2)

        ref_pesto_bounds = [0.7797]
        for n in n_list:
            wc, _ = wc_three_operator_splitting_operators(L, mu, beta, alpha, theta, wrapper=self.wrapper, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=self.relative_precision * ref_pesto_bounds[n - 1])

    def test_optimistic_gradient(self):
        n1, n2, L, gamma = 5, 6, 1, 1 / 4

        wc1, _ = wc_optimistic_gradient_operators(n=n1, gamma=gamma, L=L, wrapper=self.wrapper, verbose=self.verbose)
        wc2, _ = wc_optimistic_gradient_operators(n=n2, gamma=gamma, L=L, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc2, wc1)

    def test_past_extragradient(self):
        n1, n2, L, gamma = 5, 6, 1, 1 / 4

        wc1, _ = wc_past_extragradient_operators(n=n1, gamma=gamma, L=L, wrapper=self.wrapper, verbose=self.verbose)
        wc2, _ = wc_past_extragradient_operators(n=n2, gamma=gamma, L=L, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc2, wc1)

    def test_halpern_iteration(self):
        n = 10

        wc, theory = wc_halpern_iteration(n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_krasnoselskii_mann_constant_step_sizes(self):
        n = 10

        wc, theory = wc_krasnoselskii_mann_constant_step_sizes(n, gamma=3 / 4, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_krasnoselskii_mann_increasing_step_sizes(self):
        n = 10

        ref_pesto_bound = 0.059527
        wc, _ = wc_krasnoselskii_mann_increasing_step_sizes(n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, ref_pesto_bound, delta=self.relative_precision * ref_pesto_bound)

    def test_wc_inconsistent_halpern_iteration(self):
        n = 25
        wc, theory = wc_inconsistent_halpern_iteration(n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_wc_optimal_contractive_halpern_iteration(self):
        n, gamma = 3, 1.13
        wc, theory = wc_optimal_contractive_halpern_iteration(n=n, gamma=gamma, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_gradient_descent_lyapunov_1(self):
        L, n = 1, 10
        gamma = 1 / L

        wc, theory = wc_gradient_descent_lyapunov_1(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_gradient_descent_lyapunov_2(self):
        L, n = 1, 10
        gamma = 1 / L

        wc, theory = wc_gradient_descent_lyapunov_2(L, gamma, n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_accelerated_gradient_method(self):
        L, lam = 1, 10
        gamma = 1 / L

        wc, theory = wc_accelerated_gradient_method(L, gamma, lam, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_polyak_steps_in_distance_to_optimum(self):
        L, mu = 1, 0.1
        gamma = 2 / L

        wc, theory = wc_polyak_steps_in_distance_to_optimum(L, mu, gamma, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_polyak_steps_in_function_value(self):
        L, mu = 1, 0.1
        gamma = 2 / L

        wc, theory = wc_polyak_steps_in_function_value(L, mu, gamma, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_accelerated_inexact_forward_backward(self):
        L, zeta, n = 10, .87, 10

        wc, theory = wc_accelerated_inexact_forward_backward(L=L, zeta=zeta, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_partially_inexact_douglas_rachford_splitting(self):
        mu, L, gamma, sigma, n = 1, 5., 1.4, 0.2, 5

        wc, theory = wc_partially_inexact_douglas_rachford_splitting(mu, L, n, gamma, sigma, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_relatively_inexact_proximal_point_algorithm(self):
        gamma, sigma, n = 2, 0.3, 5

        wc, theory = wc_relatively_inexact_proximal_point_algorithm(n, gamma, sigma, wrapper=self.wrapper, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_accelerated_gradient_flow_convex(self):
        t = 3.4

        wc, theory = wc_accelerated_gradient_flow_convex(t=t, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_gradient_flow_convex(self):
        t = 3.4

        wc, theory = wc_gradient_flow_convex(t=t, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_accelerated_gradient_flow_strongly_convex(self):
        mu = 2.1
        for psd in {True, False}:
            wc, theory = wc_accelerated_gradient_flow_strongly_convex(mu=mu, psd=psd, wrapper=self.wrapper, verbose=self.verbose)
            self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_gradient_flow_strongly_convex(self):
        mu = .8

        wc, theory = wc_gradient_flow_strongly_convex(mu=mu, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_gradient_descent_contraction(self):
        L, mu, n = 1, 0.1, 1
        gamma = 1 / L

        wc, theory = wc_gradient_descent_contraction(L=L, mu=mu, gamma=gamma, n=n, wrapper=self.wrapper, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)


class TestExamplesMosek(TestExamplesCVXPY):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapper = "mosek"

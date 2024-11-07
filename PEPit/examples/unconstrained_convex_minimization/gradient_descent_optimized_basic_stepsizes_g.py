import warnings
from math import sqrt, log2
import numpy as np
from math import floor, sqrt
from typing import List
    
from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_gradient_descent_OBS_G(L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for :math:`n` steps of the **gradient descent** method using a corresponding optimized basic stepsize schedule (OBS-F).
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: \\|\nabla f(x_n)\\|^2 \\leqslant \\tau(n, L) (f(x_0) - f(x_\\star))

    is valid, where :math:`x_n` is the output of gradient descent using OBS-G stepsizes, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case
    value of :math:`\\|\nabla f(x_n)\\|^2` when :math:`f(x_0) - f(x_\\star) \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - h_t \\nabla f(x_t),

    where `h_t` is a stepsize of the :math:`t^{th}` OBS-G step-size schedule described in [1].

    **Theoretical guarantee**:
    The a tight description of the method's worstcase convergence rate (and a simpler uniform upper bound) can be found in [1, Corollary 1]:

    .. math:: \\|\nabla f(x_n)\\|^2 \\leqslant \\frac{2L}{1 + 2\\sum h_t} (f(x_0) - f(x_\\star)) < \\frac{0.84624 L}{n^\\log_2(1+\\sqrt{2})} (f(x_0) - f(x_\\star)).

    These stepsizes and rates match or beat the numerically globally minimax optimal schedules designed in [2]. As a result, they are conjectured to be the minimiax optimal stepsizes for gradient descent.
    The concurrent work of Zhang and Jiang [3] used a similar dynamical programming technique to provide an equivalent alternative construction of these potentially minimax optimal schedules.
    Therein, slightly different definitions and terms are used: focusing on “primitive”, “dominant”, and “g-bounded” schedules.

    **References**:

    `[1] Benjamin Grimmer, Kevin Shu, Alex L. Wang (2024).
    Composing Optimized Stepsize Schedules for Gradient Descent.
    arXiv preprint arXiv:2410.16249.
    <https://arxiv.org/abs/2410.16249>`_
    
    `[2] Shuvomoy Das Gupta, Bart P.G. Van Parys, and Ernest Ryu. (2023).
    Branch-and-bound performance estimation programming: A unified methodology for constructing optimal optimization methods.
    Mathematical Programming`_
    
    `[3] Zehao Zhang and Rujun Jiang (2024).
    Accelerated gradient descent by concatenation of stepsize schedules.
    arXiv preprint arXiv:2410.12395.
    <https://arxiv.org/abs/2410.12395>`_

    Args:
        L (float): the smoothness parameter.
        n (int): number of iterations (will be reset to the largest power of 2 minus 1 smaller than the provided value).
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): Level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value

    Example:
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_OBS_G(L=1, n=10, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 13x13
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                    Function 1 : Adding 132 scalar constraint(s) ...
                    Function 1 : 132 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: SCS); optimal value: 0.042490267003873934
        (PEPit) Primal feasibility check:
                The solver found a Gram matrix that is positive semi-definite
                All the primal scalar constraints are verified up to an error of 1.2494104401032137e-05
        (PEPit) Dual feasibility check:
                The solver found a residual matrix that is positive semi-definite up to an error of 2.6270319634864156e-16
                All the dual scalar values associated with inequality constraints are nonnegative up to an error of 3.6892295572041405e-18
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 6.299431562115676e-06
        (PEPit) Final upper bound (dual): 0.042489341246081315 and lower bound (primal example): 0.042490267003873934 
        (PEPit) Duality gap: absolute: -9.25757792619164e-07 and relative: -2.178752589468926e-05
        *** Example file: worst-case performance of gradient descent with Optimized Basic Stepsizes ***
            PEPit guarantee:		 ||gradf(x_n)||^2 <= 0.0424893 (f(x_0)-f(x_*))
            Tight Theoretical guarantee:	 ||gradf(x_n)||^2 <= 0.0424890122728 (f(x_0)-f(x_*))
    """


    # Define OBS_S and OBS_F stepsizes
    class SComp:
        def __init__(self, sequence: List[float], rate: float):
            self.sequence = sequence
            self.rate = rate

    class FComp:
        def __init__(self, sequence: List[float], rate: float):
            self.sequence = sequence
            self.rate = rate

    class GComp:
        def __init__(self, sequence: List[float], rate: float):
            self.sequence = sequence
            self.rate = rate    
        
    tolerance = 1e-14

    def join_scomp(L: float, R: float) -> float:
        denominator = L + R + sqrt(L**2 + 6 * L * R + R**2)
        return 2 * L * R / max(denominator, tolerance)

    def join_fcomp(L: float, R: float) -> float:
        denominator = L + 4 * R + sqrt(L**2 + 8 * L * R)
        return 2 * L * R / max(denominator, tolerance)

    def join_scomp_objects(L: SComp, R: SComp) -> SComp:
        l = L.rate
        r = R.rate
        sigmaL = (1 - L.rate) / L.rate
        sigmaR = (1 - R.rate) / R.rate
        prodL = L.rate
        prodR = R.rate

        mu = 0.5 * (-sigmaL - sigmaR + sqrt((4 + prodL * prodR * (2 + sigmaL + sigmaR) ** 2) / (prodL * prodR)))
        sequence = L.sequence + [mu] + R.sequence
        rate = 1 / (1 + sigmaL + sigmaR + mu)
        return SComp(sequence, rate)

    def join_scomp_fcomp(L: SComp, R: FComp) -> FComp:
        mu = 1 + (sqrt(1 + 8 * R.rate / L.rate) - 1) / (4 * R.rate)
        sequence = L.sequence + [mu] + R.sequence
        rate = 1 / (1 / R.rate + 2 * mu + 2 * (1 / L.rate - 1))
        return FComp(sequence, rate)

    def compute_scomp_pivots(N: int):
        s_pivot = np.zeros(N + 1, dtype=int)  # Shifted to 1-based index
        s_rate = np.zeros(N + 1)
        for i in range(1, N + 1):
            s_pivot[i] = floor(i / 2)
            L = 1.0 if s_pivot[i] == 0 else s_rate[s_pivot[i]]
            R = 1.0 if (i - 1 - s_pivot[i]) == 0 else s_rate[i - 1 - s_pivot[i]]
            s_rate[i] = join_scomp(L, R)
        return s_pivot, s_rate

    def compute_fcomp_pivots(N: int, s_pivot, s_rate):
        f_pivot = np.zeros(N + 1, dtype=int)  # Shifted to 1-based index
        f_rate = np.zeros(N + 1)
        f_pivot[1] = 0
        f_rate[1] = 0.25
        for i in range(2, N + 1):
            m = f_pivot[i - 1] if i > 1 else 0
            L = 1.0 if m == 0 else s_rate[m]
            R = 1.0 if (i - 1 - m) == 0 else f_rate[i - 1 - m]
            rate1 = join_fcomp(L, R)

            m += 1
            L = 1.0 if m == 0 else s_rate[m]
            R = 1.0 if (i - 1 - m) <= 0 else f_rate[i - 1 - m]
            rate2 = join_fcomp(L, R)

            if rate1 < rate2:
                f_pivot[i] = m - 1
                f_rate[i] = rate1
            else:
                f_pivot[i] = m
                f_rate[i] = rate2
        return f_pivot

    def OBS_S_with_pivots(N: int, s_pivot):
        if N == 0:
            return SComp([], 1.0)
        else:
            p = s_pivot[N]
            return join_scomp_objects(OBS_S_with_pivots(p, s_pivot), OBS_S_with_pivots(N - 1 - p, s_pivot))

    def OBS_S(N: int) -> SComp:
        s_pivot, _ = compute_scomp_pivots(N)
        return OBS_S_with_pivots(N, s_pivot)

    def OBS_F_with_pivots(N: int, s_pivot, f_pivot):
        if N == 0:
            return FComp([], 1.0)
        else:
            p = f_pivot[N]
            return join_scomp_fcomp(OBS_S_with_pivots(p, s_pivot), OBS_F_with_pivots(N - 1 - p, s_pivot, f_pivot))

    def OBS_F(N: int) -> FComp:
        s_pivot, s_rate = compute_scomp_pivots(N)
        f_pivot = compute_fcomp_pivots(N, s_pivot, s_rate)
        return OBS_F_with_pivots(N, s_pivot, f_pivot)

    obsSchedule = OBS_F(n)
    h = obsSchedule.sequence
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the objective gap between x0 and x^* is bounded
    problem.set_initial_condition(func(x0) - func(xs) <= 1)

    # Run n steps of the GD method with the reverse of the OBS_F pattern
    x = x0
    for i in range(n):
        x = x - h[n-1-i] / L * func.gradient(x)

    # Set the performance metric to the gradient norm accuracy
    problem.set_performance_metric(func.gradient(x) **2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with Optimized Basic Stepsizes ***')
        print('\tPEPit guarantee:\t\t ||gradf(x_n)||^2 <= {:.6} (f(x_0)-f(x_*))'.format(pepit_tau))
        print('\tTight Theoretical guarantee:\t ||gradf(x_n)||^2 <= {:.12} (f(x_0)-f(x_*))'.format(2*L*obsSchedule.rate))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, 2*obsSchedule.rate


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_gradient_descent_OBS_G(L=1, n=10,wrapper="cvxpy", solver=None, verbose=1)
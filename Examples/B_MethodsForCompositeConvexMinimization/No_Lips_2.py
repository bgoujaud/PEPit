import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Function_classes.convex_indicator import ConvexIndicatorFunction
from PEPit.Primitive_steps.bregmangradient_step import BregmanGradient_Step


def wc_no_lips2(L, gamma, n, verbose=True):
    """
    Consider the constrainted composite convex minimization problem,
        min_x { F(x) = f_1(x) + f_2(x) }
    where f_2 is a closed convex indicator function and f_1 is convex and L-smooth relatively to h,
    and h is closed proper and convex.

    This code computes a worst-case guarantee for the NoLips Method solving this problem.
    That is, it computes the smallest possible tau(n,L) such that the guarantee
        min_n (Dh(x_n, x_*)) <= tau(n,L) * Dh(x*,x0)
    is valid, where x_n is the output of the NoLips method, and where x_* is a minimizer of F,
    and where Dh is the Bregman distance generated by h.

    The detailed approach is availaible in [1]. The formulation as a PEP and the tightness are proven in [2].
    [1] Heinz H. Bauschke, Jérôme Bolte, and Marc Teboulle. "A Descent Lemma
         Beyond Lipschitz Gradient Continuity: First-Order Methods Revisited
         and Applications." (2017)

    [2] Radu-Alexandru Dragomir, Adrien B. Taylor, Alexandre d’Aspremont, and
         Jérôme Bolte. "Optimal Complexity and Certification of Bregman
         First-Order Methods". (2019)

    DISCLAIMER: This example requires some experience with PESTO and PEPs
    (see Section 4 in [2]).

    :param L: (float) relative-smoothness parameter
    :param gamma: (float) step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare two convex functions and a convex indicator function
    d = problem.declare_function(ConvexFunction,
                                 param={})
    func1 = problem.declare_function(ConvexFunction,
                                     param={})
    h = (d + func1) / L
    func2 = problem.declare_function(ConvexIndicatorFunction,
                                     param={'D': np.inf})

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    Fs = func.value(xs)
    gfs, fs = func1.oracle(xs)
    ghs, hs = h.oracle(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    gh0, h0 = h.oracle(x0)
    gf0, f0 = func1.oracle(x0)

    # Set the initial constraint that is the Bregman distance between x0 and x^*
    problem.set_initial_condition(hs - h0 - gh0 * (xs - x0) <= 1)

    # Compute n steps of the NoLips starting from x0
    x1, x2 = x0, x0
    gfx = gf0
    ghx = gh0
    hx1, hx2 = h0, h0
    for i in range(n):
        x2, _, _ = BregmanGradient_Step(gfx, ghx, func2 + h, gamma)
        gfx, _ = func1.oracle(x2)
        ghx, hx2 = h.oracle(x2)
        Dhx = hx1 - hx2 - ghx * (x1 - x2)
        # update the iterates
        x1 = x2
        hx1 = hx2
        # Set the performance metric to the Bregman distance to the optimum
        problem.set_performance_metric(Dhx)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 / (n - 1) / n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the NoLips_2 in Bregman distance ***')
        print('\tPEP-it guarantee:\t min_n Dh(y_n, y_(n-1)) <= {:.6} Dh(y_0, x_*)'.format(pepit_tau))
        print('\tTheoretical guarantee :\t min_n Dh(y_n, y_(n-1)) <= {:.6} Dh(y_0, x_*) '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    gamma = 1 / L
    n = 10

    pepit_tau, theoretical_tau = wc_no_lips2(L=L,
                                             gamma=gamma,
                                             n=n)

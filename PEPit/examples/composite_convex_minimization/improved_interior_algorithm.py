import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction
from PEPit.functions import StronglyConvexFunction
from PEPit.functions import ConvexIndicatorFunction
from PEPit.primitive_steps import bregman_gradient_step


def wc_improved_interior_algorithm(L, mu, c, lam, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the composite convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\{F(x) \\equiv f_1(x) + f_2(x)\\},

    where :math:`f_1` is a :math:`L`-smooth convex function, and :math:`f_2` is a closed convex indicator function.
    We use a kernel function :math:`h` that is assumed to be closed, proper, and strongly convex (see [1, Section 5]).

    This code computes a worst-case guarantee for **Improved interior gradient algorithm** (IGA).
    That is, it computes the smallest possible :math:`\\tau(\\mu,L,c,\\lambda,n)` such that the guarantee

    .. math:: F(x_n) - F(x_\\star) \\leqslant \\tau(\\mu,L,c,\\lambda,n)  (c D_h(x_\\star;x_0) + f_1(x_0) - f_1(x_\\star))

    is valid, where :math:`x_n` is the output of the IGA and where :math:`x_\\star` is a minimizer of :math:`F` and
    :math:`D_h` is the Bregman distance generated by :math:`h`.

    In short, for given values of :math:`\\mu`, :math:`L`, :math:`c`, :math:`\\lambda` and :math:`n`,
    :math:`\\tau(\\mu,L,c,\\lambda,n)` is computed as the worst-case value of :math:`F(x_n)-F_\\star`
    when :math:`c  D_h(x_\\star;x_0) + f_1(x_0) - f_1(x_\\star)\\leqslant 1`.

    **Algorithm**:
    The IGA is described in [1, "Improved Interior Gradient Algorithm"]. For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\alpha_t & = & \\frac{\\sqrt{(c_t\\lambda)^2+4c_t\\lambda}-\\lambda c_t}{2},\\\\
                y_t & = & (1-\\alpha_t) x_t + \\alpha_t z_t,\\\\
                c_{t+1} & = & (1-\\alpha_t)c_t,\\\\
                z_{t+1} & = & \\arg\\min_{z} \\left\\{ \\left< z;\\frac{\\alpha_t}{c_{t+1}}\\nabla f_1(y_t)\\right> +f_2(z)+D_h(z;z_t)\\right\\}, \\\\
                x_{t+1} & = & (1-\\alpha_t) x_t + \\alpha_t z_{t+1}.
            \\end{eqnarray}

    **Theoretical guarantee**:
    The following **upper** bound can be found in [1, Theorem 5.2]:

    .. math:: F(x_n) - F_\\star \\leqslant \\frac{4L}{c n^2}\\left(c  D_h(x_\\star;x_0) + f_1(x_0) - f_1(x_\\star) \\right).

    **References**:

    `[1] A. Auslender, M. Teboulle (2006).
    Interior gradient and proximal methods for convex and conic optimization.
    SIAM Journal on Optimization 16.3 (2006): 697-725.
    <https://epubs.siam.org/doi/pdf/10.1137/S1052623403427823>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong-convexity parameter.
        c (float): initial value.
        lam (float): the step-size.
        n (int): number of iterations.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> L = 1
        >>> lam = 1 / L
        >>> pepit_tau, theoretical_tau = wc_improved_interior_algorithm(L=L, mu=1, c=1, lam=lam, n=5, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 22x22
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 3 function(s)
        			Function 1 : Adding 42 scalar constraint(s) ...
        			Function 1 : 42 scalar constraint(s) added
        			Function 2 : Adding 49 scalar constraint(s) ...
        			Function 2 : 49 scalar constraint(s) added
        			Function 3 : Adding 42 scalar constraint(s) ...
        			Function 3 : 42 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.06807717876241919
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.9786985790819003e-08
        (PEPit) Final upper bound (dual): 0.06807717277007506 and lower bound (primal example): 0.06807717876241919 
        (PEPit) Duality gap: absolute: -5.992344120908655e-09 and relative: -8.802280338057462e-08
        *** Example file: worst-case performance of the Improved interior gradient algorithm in function values ***
        	PEPit guarantee:		 F(x_n)-F_* <= 0.0680772 (c * Dh(xs;x0) + f1(x0) - F_*)
        	Theoretical guarantee:	 F(x_n)-F_* <= 0.111111 (c * Dh(xs;x0) + f1(x0) - F_*)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare three convex functions
    func1 = problem.declare_function(SmoothConvexFunction, L=L)
    func2 = problem.declare_function(ConvexIndicatorFunction, D=np.inf)
    h = problem.declare_function(StronglyConvexFunction, mu=mu, reuse_gradient=True)

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func(xs)
    ghs, hs = h.oracle(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    gh0, h0 = h.oracle(x0)
    g10, f10 = func1.oracle(x0)

    # Compute n steps of the Improved Interior Algorithm starting from x0
    x = x0
    z = x0
    g = g10
    gh = gh0
    ck = c
    for i in range(n):
        alphak = (np.sqrt((ck * lam) ** 2 + 4 * ck * lam) - lam * ck) / 2
        ck = (1 - alphak) * ck
        y = (1 - alphak) * x + alphak * z
        if i >= 1:
            g, f = func1.oracle(y)
        z, _, _ = bregman_gradient_step(g, gh, h + func2, alphak / ck)
        x = (1 - alphak) * x + alphak * z
        gh, _ = h.oracle(z)

    # Set the initial constraint that is a Lyapunov distance between x0 and x^*
    problem.set_initial_condition((hs - h0 - gh0 * (xs - x0)) * c + f10 - fs <= 1)

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)
    if problem.wrapper.solver_name.casefold() != "mosek" and verbose > 0:
        print("\033[96m(PEPit) We recommend to use MOSEK solver. \033[0m")

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (4 * L) / (c * (n + 1) ** 2)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file:'
              ' worst-case performance of the Improved interior gradient algorithm in function values ***')
        print('\tPEPit guarantee:\t\t F(x_n)-F_* <= {:.6} (c * Dh(xs;x0) + f1(x0) - F_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t F(x_n)-F_* <= {:.6} (c * Dh(xs;x0) + f1(x0) - F_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    lam = 1 / L
    pepit_tau, theoretical_tau = wc_improved_interior_algorithm(L=L, mu=1, c=1, lam=lam, n=5,
                                                                wrapper="cvxpy", solver=None,
                                                                verbose=1)

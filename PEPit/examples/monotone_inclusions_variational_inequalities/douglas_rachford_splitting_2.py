from math import sqrt

from PEPit import PEP
from PEPit.operators import CocoerciveOperator
from PEPit.operators import StronglyMonotoneOperator
from PEPit.primitive_steps import proximal_step


def wc_douglas_rachford_splitting_2(beta, mu, alpha, theta, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the monotone inclusion problem

    .. math:: \\mathrm{Find}\\, x:\\, 0\\in Ax + Bx,

    where :math:`A` is :math:`\\beta`-cocoercive and maximally monotone and :math:`B` is (maximally) :math:`\\mu`-strongly
    monotone. We denote by :math:`J_{\\alpha A}` and :math:`J_{\\alpha B}` the resolvents of respectively
    :math:`\\alpha A` and :math:`\\alpha B`.

    This code computes a worst-case guarantee for the **Douglas-Rachford splitting** (DRS).
    That is, given two initial points :math:`w^{(0)}_t` and :math:`w^{(1)}_t`,
    this code computes the smallest possible :math:`\\tau(\\beta, \\mu, \\alpha, \\theta)`
    (a.k.a. "contraction factor") such that the guarantee

    .. math:: \\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\\|^2 \\leqslant \\tau(\\beta, \\mu, \\alpha, \\theta) \\|w^{(0)}_{t} - w^{(1)}_{t}\\|^2,

    is valid, where :math:`w^{(0)}_{t+1}` and :math:`w^{(1)}_{t+1}` are obtained after one iteration of DRS from
    respectively :math:`w^{(0)}_{t}` and :math:`w^{(1)}_{t}`.

    In short, for given values of :math:`\\beta`, :math:`\\mu`, :math:`\\alpha` and :math:`\\theta`, the contraction
    factor :math:`\\tau(\\beta, \\mu, \\alpha, \\theta)` is computed as the worst-case value of
    :math:`\\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\\|^2` when :math:`\\|w^{(0)}_{t} - w^{(1)}_{t}\\|^2 \\leqslant 1`.

    **Algorithm**: One iteration of the Douglas-Rachford splitting is described as follows,
    for :math:`t \in \\{ 0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & J_{\\alpha B} (w_t),\\\\
                y_{t+1} & = & J_{\\alpha A} (2x_{t+1}-w_t),\\\\
                w_{t+1} & = & w_t - \\theta (x_{t+1}-y_{t+1}).
            \\end{eqnarray}

    **Theoretical guarantee**: Theoretical worst-case guarantees can be found in [1, section 4, Theorem 4.1].

    **References**: The detailed PEP methodology for studying operator splitting is provided in [1].

    `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020). Operator splitting performance estimation:
    Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3), 2251-2271.
    <https://arxiv.org/pdf/1812.00146.pdf>`_

    Args:
        beta (float): the Lipschitz parameter.
        mu (float): the strongly monotone parameter.
        alpha (float): the step-size in the resolvent.
        theta (float): algorithm parameter.
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
        >>> pepit_tau, theoretical_tau = wc_douglas_rachford_splitting(beta=1, mu=.1, alpha=1.3, theta=.9, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 6x6
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
        			Function 1 : Adding 2 scalar constraint(s) ...
        			Function 1 : 2 scalar constraint(s) added
        			Function 2 : Adding 1 scalar constraint(s) ...
        			Function 2 : 1 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.928770707839351
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.297473722026212e-09
        		All the primal scalar constraints are verified up to an error of 1.64989273354621e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.4855088898444464e-07
        (PEPit) Final upper bound (dual): 0.9287707057295752 and lower bound (primal example): 0.928770707839351 
        (PEPit) Duality gap: absolute: -2.109775798508906e-09 and relative: -2.2715787445719413e-09
        *** Example file: worst-case performance of the Douglas Rachford Splitting***
        	PEPit guarantee:	 ||w_(t+1)^0 - w_(t+1)^1||^2 <= 0.928771 ||w_(t)^0 - w_(t)^1||^2
        	Theoretical guarantee:	 ||w_(t+1)^0 - w_(t+1)^1||^2 <= 0.928771 ||w_(t)^0 - w_(t)^1||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(CocoerciveOperator, beta=beta)
    B = problem.declare_function(StronglyMonotoneOperator, mu=mu)

    # Then define starting points w0 and w1
    w0 = problem.set_initial_point()
    w1 = problem.set_initial_point()

    # Set the initial constraint that is the distance between w0 and w1
    problem.set_initial_condition((w0 - w1) ** 2 <= 1)

    # Compute one step of the Douglas Rachford Splitting starting from w0
    x0, _, _ = proximal_step(w0, B, alpha)
    y0, _, _ = proximal_step(2 * x0 - w0, A, alpha)
    z0 = w0 - theta * (x0 - y0)

    # Compute one step of the Douglas Rachford Splitting starting from w1
    x1, _, _ = proximal_step(w1, B, alpha)
    y1, _, _ = proximal_step(2 * x1 - w1, A, alpha)
    z1 = w1 - theta * (x1 - y1)

    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric((z0 - z1) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison), see [1, Theorem 4.1]
    mu = alpha * mu
    beta = alpha * beta
    if mu * beta - mu + beta < 0 and theta <= 2 * ( beta + 1 ) * ( mu - beta - mu * beta ) / ( mu + mu * beta - beta - beta ** 2 - 2 * mu * beta ** 2):
        theoretical_tau = ( 1 - theta * beta / ( beta + 1 ) ) ** 2
    elif mu * beta - mu - beta > 0 and theta <=  2 * ( mu ** 2 + beta ** 2 + mu * beta + mu + beta - mu ** 2 * beta ** 2) / (mu ** 2 + beta ** 2 + mu ** 2 * beta + mu * beta ** 2 + mu + beta - 2 * mu ** 2 * beta ** 2):
        theoretical_tau = ( 1 - theta * ( 1 + mu * beta ) / ( mu + 1 ) / ( beta + 1 ) ) ** 2
    elif theta >= 2 * ( mu * beta + mu + beta ) / ( 2 * mu * beta + mu + beta ) :
    	theoretical_tau = ( 1 - theta ) ** 2
    elif mu * beta + mu - beta < 0 and theta <= 2 * ( mu + 1 ) * ( beta - mu - mu * beta) / ( beta + mu * beta - mu - mu ** 2 - 2 * mu ** 2 * beta ):
    	theoretical_tau = ( 1 - theta * mu / ( mu + 1 ) ) ** 2
    else :
        theoretical_tau = (2 - theta) / 4 / mu * ( ( 2 - theta ) * mu * ( beta + 1 ) + theta * beta * ( 1 - mu ) ) * ( ( 2 - theta ) * beta * ( mu + 1 ) + theta * mu * ( 1 - beta ) ) / mu / beta / ( 2 * mu * beta * ( 1 - theta ) + ( 2 - theta ) * ( mu + beta + 1 ) )

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Douglas Rachford Splitting***')
        print('\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= {:.6} ||w_(t)^0 - w_(t)^1||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= {:.6} ||w_(t)^0 - w_(t)^1||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_douglas_rachford_splitting_2(beta=1.2, mu=.1, alpha=.3, theta=1.5,
                                                                 wrapper="cvxpy", solver=None,
                                                                 verbose=1)

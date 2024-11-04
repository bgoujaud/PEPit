import numpy as np

from PEPit.function import Function

class ConvexIndicatorFunction(Function):
    """
    The :class:`ConvexIndicatorFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing interpolation constraints for the class of closed convex indicator functions.

    Attributes:
        D (float): upper bound on the diameter of the feasible set, possibly set to np.inf
        R (float): upper bound on the radius of the feasible set, possibly set to np.inf
        center (Point): Center of the feasible set spanned by the radius constraint, possibly set to None.
    Convex indicator functions are characterized by a parameter `D` (or `R`), hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit import Point
        >>> from PEPit.functions import ConvexIndicatorFunction
        >>> problem = PEP()
        >>> func1 = problem.declare_function(function_class=ConvexIndicatorFunction, D=1)
        >>> func2 = problem.declare_function(function_class=ConvexIndicatorFunction, R=1)
        >>> omega = Point()
        >>> func3 = problem.declare_function(function_class=ConvexIndicatorFunction, R=1, center=omega)
        
    References:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Exact worst-case performance of first-order methods for composite convex optimization.
    SIAM Journal on Optimization, 27(3):1283–1313.
    <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 D=np.inf,
                 R=np.inf,
                 center=None,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None):
        """

        Args:
            D (float): Diameter of the support of self. Default value set to infinity.
            R (float): Radius of the support of self. Default value set to infinity.
            center: Center of the feasible set spanned by the radius constraint of self. Default value set to None.
                    If the value is None, the feasible set is centered on the origin.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient,
                         name=name,
                         )

        # Store the diameter D in an attribute
        self.D = D
        # Store the radius R in an attribute
        self.R = R
        # Store the center in an attribute 
        self.center = center

    @staticmethod
    def set_value_constraint_i(xi, gi, fi):
        """
        Set the value of the function to 0 everywhere on the support.

        """
        # Value constraint
        constraint = (fi == 0)

        return constraint

    @staticmethod
    def set_convexity_constraint_i_j(xi, gi, fi,
                                     xj, gj, fj,
                                     ):
        """
        Formulates the list of interpolation constraints for self (CCP function).
        """
        # Interpolation conditions of convex functions class
        constraint = (0 >= gj * (xi - xj))

        return constraint

    def set_diameter_constraint_i_j(self,
                                    xi, gi, fi,
                                    xj, gj, fj,
                                    ):
        """
        Formulate the constraints bounding the diameter of the support of self.

        """
        # Diameter constraint
        constraint = ((xi - xj) ** 2 <= self.D ** 2)

        # Radius constraint 
        if self.R < np.inf:
            # No self.center provided centers the ball on the origin
            if self.center is None: 
                constraint = ((xi)**2 <= self.R ** 2)
            # Centering the ball on self.center
            else: 
                constraint = ((self.center - xi)**2 <= self.R ** 2)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (closed convex indicator function),
        see [1, Theorem 3.6].
        """
        self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                     constraint_name="value",
                                                     set_class_constraint_i=self.set_value_constraint_i,
                                                     )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="convexity",
                                                      set_class_constraint_i_j=self.set_convexity_constraint_i_j,
                                                      )
        if (self.D != np.inf) or (self.R != np.inf):
            self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                          list_of_points_2=self.list_of_points,
                                                          constraint_name="diameter",
                                                          set_class_constraint_i_j=self.set_diameter_constraint_i_j,
                                                          )

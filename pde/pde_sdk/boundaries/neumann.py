from .base import BoundaryCondition


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition: ∂u/∂n = g

    Specifies the normal derivative (gradient) at the boundary.
    For 1D: ∂u/∂x = g
    For 2D: ∂u/∂n = g (normal derivative)
    """

    def __init__(self, value: float):
        """
        Parameters
        ----------
        value : float
            The gradient value g such that ∂u/∂n = g at the boundary
        """
        self.value = value

    def apply(self, values):
        """
        Apply Neumann boundary condition.

        Note: This is a placeholder method. Actual application
        is solver-specific and requires grid spacing information.
        Solvers will implement their own Neumann BC logic.

        Parameters
        ----------
        values : np.ndarray
            Array of values (1D or 2D)

        Returns
        -------
        np.ndarray
            Modified values (for consistency with DirichletBC interface)
        """
        # This method exists for interface consistency
        # Actual Neumann BC application requires dx/dy and is solver-specific
        return values


class BoundaryCondition:
    """Base class for boundary conditions."""

    def apply(self, values):
        raise NotImplementedError

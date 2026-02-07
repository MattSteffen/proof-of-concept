"""Progress tracking utilities for solvers."""



class ProgressTracker:
    """
    Progress tracker for solver iterations.

    Supports different verbosity levels:
    - 'none': No progress output
    - 'summary': Show summary at end
    - 'steps': Show progress bar for each step

    Parameters
    ----------
    verbosity : {'none', 'summary', 'steps'}, default='summary'
        Verbosity level for progress output
    total_steps : int, optional
        Total number of steps (for progress bar)
    description : str, optional
        Description to show in progress output

    Examples
    --------
    >>> tracker = ProgressTracker(verbosity='steps', total_steps=100)
    >>> for i in range(100):
    ...     tracker.update(1)
    """

    def __init__(
        self,
        verbosity: str = "summary",
        total_steps: int | None = None,
        description: str | None = None,
    ):
        self.verbosity = verbosity
        self.total_steps = total_steps
        self.description = description or "Solving"
        self.current_step = 0
        self._tqdm = None

        if verbosity == "steps" and total_steps is not None:
            try:
                from tqdm import tqdm

                self._tqdm = tqdm(
                    total=total_steps,
                    desc=self.description,
                    unit="step",
                    ncols=80,
                )
            except ImportError:
                # Fallback if tqdm not available
                self.verbosity = "summary"
                self._tqdm = None

    def update(self, n: int = 1, **kwargs):
        """
        Update progress by n steps.

        Parameters
        ----------
        n : int
            Number of steps to advance
        **kwargs
            Additional information to display
        """
        self.current_step += n

        if self.verbosity == "steps" and self._tqdm is not None:
            self._tqdm.update(n)
            if kwargs:
                # Update description with additional info
                info_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                self._tqdm.set_postfix_str(info_str)

    def set_description(self, desc: str):
        """Update the progress description."""
        self.description = desc
        if self._tqdm is not None:
            self._tqdm.set_description(desc)

    def close(self):
        """Close the progress tracker."""
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def print_summary(self, message: str = ""):
        """
        Print a summary message.

        Parameters
        ----------
        message : str
            Additional message to print
        """
        if self.verbosity in ("summary", "steps"):
            if message:
                print(f"{self.description}: {message}")
            elif self.verbosity == "summary":
                print(f"{self.description}: Completed {self.current_step} steps")


def create_progress_tracker(
    verbosity: str | None = None,
    total_steps: int | None = None,
    description: str | None = None,
) -> ProgressTracker | None:
    """
    Create a progress tracker with optional parameters.

    Returns None if verbosity is 'none' or not provided.

    Parameters
    ----------
    verbosity : {'none', 'summary', 'steps'}, optional
        Verbosity level
    total_steps : int, optional
        Total number of steps
    description : str, optional
        Description for progress output

    Returns
    -------
    ProgressTracker or None
        Progress tracker instance or None if not needed
    """
    if verbosity is None or verbosity == "none":
        return None

    return ProgressTracker(
        verbosity=verbosity,
        total_steps=total_steps,
        description=description,
    )


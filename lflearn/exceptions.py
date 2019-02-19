"""This module includes all custom warnings and error classes used across lift-learn."""


class UpliftModelError(Exception):
    """Base class for exceptions of uplift model."""


class NotFittedError(UpliftModelError):
    """Exception raised for errors in the prediction before model fitting."""


class MultiTreatmentError(UpliftModelError):
    """Exception raised for errors in the treatment assignment input."""

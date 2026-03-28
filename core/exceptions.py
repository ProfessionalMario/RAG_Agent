"""
Exceptions – Developer Guide

Custom errors for clear failure handling in the system.

--------------------------------------------------------------------------------
Classes:
--------------------------------------------------------------------------------
BaseAppException(message: str, code: str = "APP_ERROR")
    - Base class for all custom exceptions.
    - Use code to categorize errors.

ParserError
    - Raised when parsing fails.

RetrievalError
    - Raised when retrieval fails.

AgentError
    - Raised when the decision agent fails.

ValidationError
    - Raised for invalid input.
"""

class BaseAppException(Exception):
    """Base exception for all custom errors"""
    def __init__(self, message: str, code: str = "APP_ERROR"):
        super().__init__(message)
        self.code = code


class ParserError(BaseAppException):
    """Raised when parsing fails"""
    pass


class RetrievalError(BaseAppException):
    """Raised when retrieval fails"""
    pass


class AgentError(BaseAppException):
    """Raised when decision agent fails"""
    pass


class ValidationError(BaseAppException):
    """Raised for invalid input"""
    pass
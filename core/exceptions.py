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
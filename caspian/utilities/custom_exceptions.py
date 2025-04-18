class UnsafeMemoryAccessException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class InvalidDataException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class ShapeIncompatibilityException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
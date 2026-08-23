class ModelNotReadyError(Exception):
    """
    Raised when the model is not available
    when a generation request is received.
    """

    pass


class GenerationError(Exception):
    """
    Raised when model inference fails
    while generating a response.
    """

    pass
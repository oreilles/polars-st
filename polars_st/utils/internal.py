from collections.abc import Callable


def is_empty_method(func: Callable) -> bool:
    """Return True if the function body is only '...'."""

    def _empty_with_docstring() -> None:
        """Docstring."""
        ...

    def _empty_without_docstring() -> None: ...

    empty_bytecodes = {
        _empty_with_docstring.__code__.co_code,
        _empty_without_docstring.__code__.co_code,
    }

    return func.__code__.co_code in empty_bytecodes

"""Tests for annotations module."""

import warnings

from spacenit.annotations import unstable


class TestUnstableDecorator:
    """Tests for @unstable decorator."""

    def test_unstable_function(self) -> None:
        """Test that unstable decorator works on functions."""

        @unstable()
        def test_func() -> int:
            """Test function."""
            return 42

        # Check marker attribute
        assert hasattr(test_func, "_unstable_")
        assert test_func._unstable_ is True

        # Check docstring updated
        assert test_func.__doc__ is not None
        assert "UNSTABLE" in test_func.__doc__
        assert "Test function" in test_func.__doc__

        # Check warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "unstable" in str(w[0].message).lower()

    def test_unstable_function_with_note(self) -> None:
        """Test unstable decorator with note."""

        @unstable("Still testing performance")
        def test_func() -> int:
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()
            assert len(w) == 1
            assert "Still testing performance" in str(w[0].message)

    def test_unstable_class(self) -> None:
        """Test that unstable decorator works on classes."""

        @unstable()
        class TestClass:
            """Test class."""

            def __init__(self, value: int) -> None:
                self.value = value

        # Check marker attribute
        assert hasattr(TestClass, "_unstable_")
        assert TestClass._unstable_ is True

        # Check docstring updated
        assert TestClass.__doc__ is not None
        assert "UNSTABLE" in TestClass.__doc__
        assert "Test class" in TestClass.__doc__

        # Check warning is raised on instantiation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = TestClass(42)
            assert obj.value == 42
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)

    def test_unstable_class_with_note(self) -> None:
        """Test unstable decorator on class with note."""

        @unstable("API may change")
        class TestClass:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TestClass()
            assert len(w) == 1
            assert "API may change" in str(w[0].message)

    def test_unstable_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function metadata."""

        @unstable()
        def my_function(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        assert my_function.__name__ == "my_function"
        assert "my_function" in my_function.__qualname__

        # Function should still work
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert my_function(1, 2) == 3

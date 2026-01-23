"""Example test file to verify testing setup."""

import pytest


class TestExample:
    """Example test class."""

    def test_basic_assertion(self):
        """Test basic assertion."""
        assert 1 + 1 == 2

    def test_with_fixture(self, sample_config):
        """Test with sample_config fixture."""
        assert "resolution" in sample_config
        assert sample_config["fps"] == 60

    @pytest.mark.slow
    def test_slow_test(self):
        """A test marked as slow."""
        assert True

    def test_skip_example(self):
        """Example of skipping a test."""
        pytest.skip("This test is skipped for demonstration")


def test_basic_function():
    """Test a basic function."""
    def add(a, b):
        return a + b

    assert add(2, 3) == 5

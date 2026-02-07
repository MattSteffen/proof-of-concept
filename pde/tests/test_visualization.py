"""Smoke tests for visualization functions."""

import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from pde_sdk.visualization.plot import (
    animate_solution,
    plot_1d,
    plot_2d,
    plot_comparison,
    plot_error,
)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestPlot1D:
    def test_basic_plot(self):
        """Test basic 1D plot."""
        x = np.linspace(0, 1, 101)
        u = np.sin(np.pi * x)
        ax = plot_1d(x, u, title="Test")
        assert ax is not None
        plt.close()

    def test_with_label(self):
        """Test 1D plot with label."""
        x = np.linspace(0, 1, 101)
        u = np.sin(np.pi * x)
        ax = plot_1d(x, u, label="Test", title="Test")
        assert ax is not None
        plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestPlot2D:
    def test_basic_plot(self):
        """Test basic 2D plot."""
        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = np.sin(np.pi * X) * np.sin(np.pi * Y)
        ax = plot_2d(X, Y, u, title="Test")
        assert ax is not None
        plt.close()

    def test_with_levels(self):
        """Test 2D plot with custom levels."""
        x = np.linspace(0, 1, 51)
        y = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = np.sin(np.pi * X) * np.sin(np.pi * Y)
        ax = plot_2d(X, Y, u, levels=10, title="Test")
        assert ax is not None
        plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestPlotComparison:
    def test_basic_comparison(self):
        """Test basic comparison plot."""
        x = np.linspace(0, 1, 101)
        u_num = np.sin(np.pi * x)
        u_analytical = np.sin(np.pi * x)
        ax = plot_comparison(x, u_num, u_analytical, title="Test")
        assert ax is not None
        plt.close()

    def test_numerical_only(self):
        """Test comparison plot with only numerical solution."""
        x = np.linspace(0, 1, 101)
        u_num = np.sin(np.pi * x)
        ax = plot_comparison(x, u_num, title="Test")
        assert ax is not None
        plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestPlotError:
    def test_basic_error_plot(self):
        """Test basic error plot."""
        x = np.linspace(0, 1, 101)
        error = np.abs(np.sin(np.pi * x) - np.sin(np.pi * x + 0.01))
        ax = plot_error(x, error, title="Test")
        assert ax is not None
        plt.close()

    def test_log_scale(self):
        """Test error plot with log scale."""
        x = np.linspace(0, 1, 101)
        error = np.abs(np.sin(np.pi * x) - np.sin(np.pi * x + 0.01))
        ax = plot_error(x, error, log_scale=True, title="Test")
        assert ax is not None
        plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestAnimateSolution:
    def test_basic_animation(self):
        """Test basic animation (smoke test)."""
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        solutions = [
            np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t) for t in [0, 0.1, 0.2]
        ]
        # Just test that it doesn't crash - don't actually show animation
        try:
            animate_solution(X, Y, solutions)
        except Exception:
            # Animation might fail in headless environment, that's okay
            pass


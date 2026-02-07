"""Tests for progress tracking functionality."""

from pde_sdk.utils.progress import ProgressTracker, create_progress_tracker


class TestProgressTracker:
    def test_basic_usage(self):
        """Test basic progress tracker usage."""
        tracker = ProgressTracker(verbosity="summary", total_steps=100)
        assert tracker.current_step == 0
        tracker.update(10)
        assert tracker.current_step == 10
        tracker.close()

    def test_steps_verbosity(self):
        """Test steps verbosity (without tqdm)."""
        tracker = ProgressTracker(verbosity="steps", total_steps=100)
        tracker.update(50)
        assert tracker.current_step == 50
        tracker.close()

    def test_summary_verbosity(self):
        """Test summary verbosity."""
        tracker = ProgressTracker(verbosity="summary", total_steps=100)
        tracker.update(100)
        tracker.print_summary("Test complete")
        tracker.close()

    def test_context_manager(self):
        """Test progress tracker as context manager."""
        with ProgressTracker(verbosity="summary") as tracker:
            tracker.update(50)
            assert tracker.current_step == 50

    def test_set_description(self):
        """Test setting description."""
        tracker = ProgressTracker(verbosity="summary", description="Test")
        tracker.set_description("New description")
        assert tracker.description == "New description"
        tracker.close()


class TestCreateProgressTracker:
    def test_none_verbosity(self):
        """Test creating tracker with none verbosity."""
        tracker = create_progress_tracker(verbosity="none")
        assert tracker is None

    def test_summary_verbosity(self):
        """Test creating tracker with summary verbosity."""
        tracker = create_progress_tracker(verbosity="summary", total_steps=100)
        assert tracker is not None
        assert tracker.verbosity == "summary"
        tracker.close()

    def test_steps_verbosity(self):
        """Test creating tracker with steps verbosity."""
        tracker = create_progress_tracker(verbosity="steps", total_steps=100)
        assert tracker is not None
        # If tqdm is not available, it falls back to summary
        # So we check that it's either steps or summary
        assert tracker.verbosity in ("steps", "summary")
        tracker.close()

    def test_none_verbosity_default(self):
        """Test default behavior (no verbosity)."""
        tracker = create_progress_tracker()
        assert tracker is None


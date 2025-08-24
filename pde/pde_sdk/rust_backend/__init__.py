"""
Rust Backend Module

**Phase 2: Future Rust Integration**

This module will provide access to high-performance Rust implementations
via PyO3/maturin when the Python SDK is mature and ready for acceleration.

Currently in Phase 1: Pure Python development
- Focus on building a solid, testable SDK foundation
- All functionality implemented in pure Python with NumPy
- Ready for modular Rust integration when needed

When ready for Phase 2, this will automatically import from pde_sdk_rust.
"""

__rust_available__ = False
__phase__ = "Phase 1: Pure Python Development"

# Future implementation will be:
# try:
#     from pde_sdk_rust import *
#     __rust_available__ = True
# except ImportError:
#     __rust_available__ = False
#     import warnings
#     warnings.warn(
#         "Rust backend not available. Run: cd ../rust/pde_sdk_rs && make build-python",
#         UserWarning,
#         stacklevel=2
#     )


def info():
    """Get information about Rust backend availability."""
    return {
        "rust_available": __rust_available__,
        "phase": __phase__,
        "message": "Currently in Phase 1. Rust integration will be available when the Python SDK is mature.",
    }

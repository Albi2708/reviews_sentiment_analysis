"""Pytest configuration for the test suite.

This file exists at the repository root so that root lands on ``sys.path``:
under pytest's default (prepend) import mode, a conftest's own directory is
inserted at import time. That lets the modules under ``tests/`` import the
top-level ``pipeline``, ``api``, and ``db`` modules directly.
"""

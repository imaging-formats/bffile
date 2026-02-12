"""Test that all python code blocks in docs/usage.md execute correctly."""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import numpy as np

USAGE_MD = Path(__file__).parent.parent / "docs" / "usage.md"


def _extract_python_blocks(md_text: str) -> list[str]:
    """Extract all ```python code blocks from markdown text."""
    return re.findall(r"```python\n(.*?)```", md_text, re.DOTALL)


def _adapt_block(block: str, test_file: str) -> str:
    """Adapt a doc code block to work with the test data file.

    The test file (ND2_dims_p4z5t3c2y32x32.nd2) has:
    4 series, each T=3, C=2, Z=5, Y=32, X=32, uint16.
    """
    # Replace example file paths with actual test data
    block = block.replace('"image.nd2"', repr(test_file))
    block = block.replace('"multi_scene.czi"', repr(test_file))
    # Test file is 32x32 pixels; scale down illustrative sub-regions
    block = block.replace("100:200", "10:20")
    block = block.replace("50:150", "5:15")
    block = block.replace("200:300", "10:20")
    block = block.replace("300:400", "15:25")
    block = block.replace("slice(200, 300)", "slice(10, 20)")
    # Test file has Z=5 (valid indices 0-4)
    block = block.replace("z=5,", "z=4,")
    return dedent(block)


def test_usage_docs(multiseries_file: Path) -> None:
    """Execute all python code blocks from usage.md in sequence."""
    blocks = _extract_python_blocks(USAGE_MD.read_text())
    assert blocks, "No python code blocks found in usage.md"

    path = str(multiseries_file)
    # Pre-seed with numpy (assumed available in doc examples)
    ns: dict = {"np": np}

    for i, block in enumerate(blocks):
        block = _adapt_block(block, path)
        try:
            exec(compile(block, f"<usage.md block {i}>", "exec"), ns)
        except ImportError:
            # Skip blocks requiring optional dependencies (e.g. dask)
            continue
        except Exception as e:
            raise RuntimeError(f"Error executing block {i}:\n{block}") from e

        # After each block, reopen any BioFile that was closed by a context
        # manager exit, so that continuation snippets can use `bf` and `arr`.
        bf = ns.get("bf")
        if bf is not None and hasattr(bf, "closed") and bf.closed:
            bf.open()

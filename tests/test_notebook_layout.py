"""Keep the public notebook surface small and portable."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT_ROOT / "notebooks"


class NotebookLayoutTests(unittest.TestCase):
    def test_no_notebooks_remain_in_the_notebook_root(self) -> None:
        self.assertEqual(list(NOTEBOOKS.glob("*.ipynb")), [])

    def test_notebooks_are_classified(self) -> None:
        self.assertEqual(len(list((NOTEBOOKS / "active").glob("*.ipynb"))), 4)
        self.assertEqual(len(list((NOTEBOOKS / "labs").glob("*.ipynb"))), 4)
        self.assertEqual(len(list((NOTEBOOKS / "archive").glob("*.ipynb"))), 12)

    def test_every_notebook_is_valid_and_finds_ancestor_root(self) -> None:
        for path in NOTEBOOKS.glob("*/*.ipynb"):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                code = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "code"
                )
                self.assertIn("Path.cwd().parents", code)
                self.assertIn('(path / "src").is_dir()', code)


if __name__ == "__main__":
    unittest.main()

"""Tests for the expanded CLI."""

import pytest

from quantllm.cli import main


class TestCLICommands:
    def test_version_command(self):
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ["quantllm", "version"]
        sys.stdout = StringIO()

        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code is None or exc.value.code == 0
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

    def test_info_command(self):
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ["quantllm", "info"]
        sys.stdout = StringIO()

        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code is None or exc.value.code == 0
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

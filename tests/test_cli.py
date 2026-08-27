"""Command-line surface promised by the public contract."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from femtools import cli as cli_module


def test_cli_help_lists_all_contract_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["--help"])

    assert result.exit_code == 0, result.output
    plain_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for command in ("solve-modes", "mac", "frf", "update", "pretest", "script"):
        assert command in plain_output

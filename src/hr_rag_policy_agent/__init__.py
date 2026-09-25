"""Installed entry point for the HR Policy Assistant."""


def main() -> None:
    """Launch the interactive command-line assistant."""
    from main import main as run_assistant

    run_assistant()

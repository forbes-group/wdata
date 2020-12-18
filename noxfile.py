import nox
import nox_poetry.patch
from nox.sessions import Session


@nox.session(python=["3.6", "3.7", "3.8"], reuse_venv=False)
def test(session: Session) -> None:
    """Run the test suite."""
    session.install(".")
    # session.run("poetry", "shell", external=True)
    # session.run("poetry", "install", external=True)
    session.run("pytest")

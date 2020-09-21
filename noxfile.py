import nox


@nox.session(
    #python=[#"3.5", "3.6", "3.7", "3.8"]
    ## These need to be installed already... could use Conda.
    reuse_venv=True)
def test(session):
    session.install(".[test]")
    session.run('pytest')

'''
@nox.session(
    venv_backend="conda",
    python=["3.5", "3.6", "3.7", "3.8"])
def test_conda(session):
    #session.conda_env_update("environment.yml")
    #session.conda("env", "update", "--f", "environment.yml",
    #              conda="mamba", external=True)
    session.install(".[test]")
    session.run('pytest')
'''

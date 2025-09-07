import subprocess
def test_cli_help():
    out = subprocess.check_output(["python","-m","spectramind","--help"]).decode()
    assert "calibrate" in out and "train" in out

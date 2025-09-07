from typer import Typer
from rich import print

app = Typer(no_args_is_help=True, add_completion=False)

@app.command()
def calibrate(config_name: str = "train"):
    """Run calibration chain per Hydra config."""
    print(f"[bold cyan]Calibrating with config:[/bold cyan] {config_name}")

@app.command()
def train(config_name: str = "train"):
    """Train SpectraMind V50 model."""
    print(f"[bold green]Training with config:[/bold green] {config_name}")

@app.command()
def predict(config_name: str = "predict", ckpt: str | None = None):
    """Run inference and produce predictions CSV."""
    print(f"[bold yellow]Predicting with config:[/bold yellow] {config_name}; ckpt={ckpt}")

@app.command()
def submit(config_name: str = "submit"):
    """Validate + package Kaggle submission."""
    print(f"[bold magenta]Submitting with config:[/bold magenta] {config_name}")

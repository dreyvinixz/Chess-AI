"""Command line interface for local training and evaluation."""

import csv
import json
import tempfile
from pathlib import Path

import chess
import torch
import typer
from rich.console import Console

from chess_ai import __version__
from chess_ai.config import load_config
from chess_ai.core import encode_board
from chess_ai.data import prepare_pgn
from chess_ai.doctor import write_report
from chess_ai.evaluation import play_match, save_report
from chess_ai.model import PolicyValueNet
from chess_ai.search import policy_only, puct
from chess_ai.teacher import label_positions
from chess_ai.training import load_model
from chess_ai.training import train as train_model

app = typer.Typer(
    help="Train, inspect, and play with a compact chess student.", no_args_is_help=True
)
console = Console()


@app.command()
def info() -> None:
    """Show package version and the separation between teacher and student."""
    console.print(
        f"Chess-AI {__version__} | Stockfish: offline teacher only | "
        "student: neural policy/value + search"
    )


@app.command()
def doctor(
    report: Path = typer.Option(Path("reports/system.json"), help="Local JSON report path."),
) -> None:
    """Check Python, PyTorch, CUDA, GPU memory, Stockfish, and Playwright."""
    console.print("Chess-AI System Doctor")
    console.print_json(data=write_report(str(report)))


@app.command("prepare-data")
def prepare_data(
    source: Path = typer.Argument(..., exists=True, help="Licensed standard-chess PGN."),
    output: Path = typer.Option(Path("data/processed/pgn")),
    min_elo: int = typer.Option(1800),
    max_games: int = typer.Option(0),
    seed: int = typer.Option(42),
    deduplicate: bool = typer.Option(True, help="Skip repeated board positions across splits."),
) -> None:
    """Stream PGN into game-separated train, validation, and test JSONL."""
    console.print_json(data=prepare_pgn(source, output, min_elo, max_games, seed, deduplicate))


@app.command("label-data")
def label_data(
    source: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option(Path("data/processed/teacher.jsonl")),
    engine: str = typer.Option("stockfish", help="UCI engine executable."),
    depth: int = typer.Option(8),
    multipv: int = typer.Option(3),
    temperature_cp: float = typer.Option(100.0, help="Policy softmax temperature in centipawns."),
    limit: int = typer.Option(0),
    sample_seed: int | None = typer.Option(
        None, help="Reservoir-sample --limit positions across the source."
    ),
) -> None:
    """Generate offline Stockfish policy/value labels for prepared positions."""
    count = label_positions(
        source, output, engine, depth, multipv, temperature_cp, limit, sample_seed
    )
    console.print(f"Labeled {count} positions: {output}")


@app.command()
def train(
    dataset: Path = typer.Option(..., exists=True, help="Training JSONL."),
    config: Path = typer.Option(Path("configs/gtx1650.yaml"), exists=True),
    run_dir: Path = typer.Option(Path("runs/latest")),
    resume: Path | None = typer.Option(None, exists=True),
    init_checkpoint: Path | None = typer.Option(
        None, exists=True, help="Warm-start from model weights with fresh optimizer state."
    ),
    max_steps: int | None = typer.Option(None, min=1, help="Override training.max_steps."),
    report_dir: Path = typer.Option(Path("reports/training"), help="Local JSON summary directory."),
) -> None:
    """Train the student from PGN imitation or teacher labels; save a resumable checkpoint."""
    experiment = load_config(config)
    if max_steps is not None:
        experiment["training"]["max_steps"] = max_steps
    checkpoint = train_model(experiment, dataset, run_dir, resume, init_checkpoint)
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    epochs = [
        json.loads(line)
        for line in (run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        last_metrics = None
        for row in csv.DictReader(stream):
            last_metrics = row
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        "environment": environment,
        "epochs": epochs,
        "last_metrics": last_metrics,
    }
    save_report(report, report_dir / f"{run_dir.name}.json")
    console.print(str(checkpoint))


def _student(checkpoint: Path) -> tuple[PolicyValueNet, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    return model, device


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., exists=True),
    opponent: str = typer.Option("random", help="random or greedy"),
    games: int = typer.Option(2, min=1),
    search: str = typer.Option("policy", help="policy or puct"),
    simulations: int = typer.Option(16),
    output: Path = typer.Option(Path("reports/evaluation/latest.json")),
) -> None:
    """Play local games against a baseline and save honest W/D/L results."""
    if opponent not in {"random", "greedy"} or search not in {"policy", "puct"}:
        raise typer.BadParameter("Use opponent=random|greedy and search=policy|puct")
    model, device = _student(checkpoint)
    report = play_match(model, device, opponent, games, search, simulations)
    report["checkpoint"] = str(checkpoint)
    save_report(report, output)
    console.print_json(data=report)


@app.command()
def benchmark(
    checkpoint: Path = typer.Option(..., exists=True),
    iterations: int = typer.Option(20, min=1),
    output: Path = typer.Option(Path("reports/benchmark.json")),
) -> None:
    """Measure model size and local inference latency. Does not measure chess strength."""
    from time import perf_counter

    model, device = _student(checkpoint)
    board = chess.Board()
    for _ in range(3):
        policy_only(model, board, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for _ in range(iterations):
        policy_only(model, board, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    from datetime import datetime, timezone

    from chess_ai.training import git_commit, git_dirty

    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint": str(checkpoint),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "device": str(device),
        "inference_ms": 1000 * (perf_counter() - started) / iterations,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1)
        if device.type == "cuda"
        else None,
        "iterations": iterations,
    }
    save_report(report, output)
    summary = (
        "# Model summary\n\n"
        f"Checkpoint: `{checkpoint}`  \n"
        f"Git commit: `{report['git_commit']}`  \n"
        f"Device: {report['gpu'] or 'CPU'}  \n"
        f"Parameters: {report['parameters']:,}  \n"
        f"Checkpoint size: {report['checkpoint_bytes'] / 1048576:.1f} MiB  \n"
        f"Policy inference: {report['inference_ms']:.2f} ms/position "
        f"over {iterations} iterations  \n"
        f"Peak allocated VRAM: {report['peak_vram_mb']} MiB\n\n"
        "This benchmark measures speed and memory, not playing strength.\n"
    )
    summary_path = (
        output.parent / "model_summary.md"
        if output.name == "benchmark.json"
        else output.with_suffix(".md")
    )
    summary_path.write_text(summary, encoding="utf-8")
    console.print_json(data=report)


@app.command()
def play(
    checkpoint: Path = typer.Option(..., exists=True),
    human_color: str = typer.Option("white"),
    search: str = typer.Option("policy"),
    simulations: int = typer.Option(16),
) -> None:
    """Play a complete terminal game against the local student; enter UCI or SAN moves."""
    model, device = _student(checkpoint)
    board = chess.Board()
    human = chess.WHITE if human_color.lower() == "white" else chess.BLACK
    while not board.is_game_over(claim_draw=True):
        console.print(board)
        if board.turn == human:
            typed = typer.prompt("Your move (SAN or UCI; quit to exit)")
            if typed.lower() == "quit":
                return
            try:
                move = board.parse_san(typed)
            except ValueError:
                try:
                    move = board.parse_uci(typed)
                except ValueError:
                    console.print("Illegal move; try again.")
                    continue
        else:
            result = (
                puct(model, board, device, simulations)
                if search == "puct"
                else policy_only(model, board, device)
            )
            move = result.move
            console.print(
                f"Chess-AI: {board.san(move)} | value {result.value:+.2f} | nodes {result.nodes}"
            )
        board.push(move)
    console.print(board)
    console.print(f"Result: {board.result(claim_draw=True)}")


@app.command("smoke-test")
def smoke_test() -> None:
    """Check encoding, CPU/CUDA inference, one training batch, checkpoint reload, and local play."""
    board = chess.Board()
    assert encode_board(board).shape == (18, 8, 8)
    config = load_config(Path(__file__).resolve().parents[2] / "configs/debug.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(**config["model"]).to(device)
    assert policy_only(model, board, device).move in board.legal_moves
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        dataset = root / "tiny.jsonl"
        rows = []
        for move in list(board.legal_moves)[:4]:
            rows.append(json.dumps({"fen": board.fen(), "move": move.uci(), "value": 0.0}))
        dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
        checkpoint = train_model(config, dataset, root / "run")
        restored, _ = load_model(checkpoint, device)
        for _ in range(8):
            board.push(puct(restored, board, device, simulations=2).move)
    console.print(f"Smoke test passed on {device}: training, checkpoint, search, 8 legal plies")


if __name__ == "__main__":
    app()

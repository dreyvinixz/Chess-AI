"""Command line interface for local training and evaluation."""

import csv
import json
import random
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
from chess_ai.evaluation import baseline_move, play_match, save_report
from chess_ai.experience import ExperienceStore
from chess_ai.model import PolicyValueNet
from chess_ai.search import policy_only, puct
from chess_ai.self_play import generate_self_play
from chess_ai.teacher import label_positions
from chess_ai.training import load_model
from chess_ai.training import train as train_model
from chess_ai.validation import validate_dataset

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
    opponent: str = typer.Option("random", help="random, greedy, or stockfish"),
    games: int = typer.Option(2, min=1),
    search: str = typer.Option("policy", help="policy or puct"),
    simulations: int = typer.Option(16),
    engine: str = typer.Option("stockfish", help="UCI executable for Stockfish baseline."),
    engine_depth: int = typer.Option(4, min=0),
    engine_nodes: int = typer.Option(0, min=0),
    engine_time_sec: float = typer.Option(0.0, min=0),
    engine_elo: int | None = typer.Option(None, help="Optional UCI_LimitStrength Elo."),
    output: Path = typer.Option(Path("reports/evaluation/latest.json")),
) -> None:
    """Play local games against a baseline and save honest W/D/L results."""
    if opponent not in {"random", "greedy", "stockfish"} or search not in {"policy", "puct"}:
        raise typer.BadParameter("Use opponent=random|greedy|stockfish and search=policy|puct")
    model, device = _student(checkpoint)
    report = play_match(
        model, device, opponent, games, search, simulations,
        engine_path=engine, engine_depth=engine_depth, engine_nodes=engine_nodes,
        engine_time_sec=engine_time_sec, engine_elo=engine_elo,
    )
    report["checkpoint"] = str(checkpoint)
    save_report(report, output)
    console.print_json(data=report)


@app.command()
def validate(
    checkpoint: Path = typer.Option(..., exists=True),
    dataset: Path = typer.Option(..., exists=True, help="Held-out prepared JSONL."),
    batch_size: int = typer.Option(32, min=1),
    output: Path = typer.Option(Path("reports/evaluation/validation.json")),
) -> None:
    """Measure policy agreement and value error on a held-out dataset."""
    model, device = _student(checkpoint)
    report = validate_dataset(model, device, dataset, checkpoint, batch_size)
    save_report(report, output)
    console.print_json(data=report)


@app.command("self-play")
def self_play(
    checkpoint: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/self-play.jsonl")),
    games: int = typer.Option(1, min=1),
    search: str = typer.Option("puct", help="puct or policy"),
    simulations: int = typer.Option(16, min=1),
    max_plies: int = typer.Option(300, min=1),
    temperature: float = typer.Option(1.0, min=0),
    temperature_plies: int = typer.Option(20, min=0),
    noise_alpha: float | None = typer.Option(0.3, help="PUCT root noise; set to 0 to disable."),
    seed: int = typer.Option(42),
) -> None:
    """Create student-only self-play examples from completed local games."""
    if noise_alpha == 0:
        noise_alpha = None
    model, device = _student(checkpoint)
    report = generate_self_play(
        model, device, checkpoint, output, games, search, simulations, max_plies,
        temperature, temperature_plies, noise_alpha, seed,
    )
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


@app.command()
def assist(
    checkpoint: Path = typer.Option(..., exists=True),
    database: Path = typer.Option(Path("data/processed/experience.sqlite")),
    opponent: str = typer.Option("greedy", help="Local random or greedy opponent."),
    human_color: str = typer.Option("white"),
    search: str = typer.Option("policy", help="policy or puct"),
    simulations: int = typer.Option(16, min=1),
    annotate_after: bool = typer.Option(True, help="Run offline Stockfish review after a game."),
    engine_depth: int = typer.Option(8, min=1),
    export: Path = typer.Option(Path("data/processed/human-experience.jsonl")),
    train_after: bool = typer.Option(
        False, help="Fine-tune from this checkpoint after annotation."
    ),
    fine_tune_steps: int = typer.Option(10, min=1),
) -> None:
    """Play a local opponent with student suggestions and record your actual moves."""
    if opponent not in {"random", "greedy"}:
        raise typer.BadParameter("Local assist opponent must be random or greedy")
    if human_color not in {"white", "black"} or search not in {"policy", "puct"}:
        raise typer.BadParameter("Use human-color white|black and search policy|puct")
    if train_after and not annotate_after:
        raise typer.BadParameter("--train-after requires --annotate-after")
    model, device = _student(checkpoint)
    human = chess.WHITE if human_color == "white" else chess.BLACK
    board = chess.Board()
    rng = random.Random(42)
    with ExperienceStore(database) as store:
        game_id = store.create_game(
            "local_assist", f"local-{opponent}", str(checkpoint), human_color
        )
        while not board.is_game_over(claim_draw=True):
            console.print(board)
            if board.turn == human:
                suggestion = (
                    puct(model, board, device, simulations)
                    if search == "puct"
                    else policy_only(model, board, device)
                )
                console.print(f"FEN: {board.fen()} | side: {human_color}")
                console.print(
                    "Student top 5: "
                    + ", ".join(
                        f"{board.san(chess.Move.from_uci(uci))} {probability:.3f}"
                        for uci, probability in suggestion.candidates
                    )
                )
                console.print(
                    f"Value: {suggestion.value:+.3f} | nodes: {suggestion.nodes} "
                    f"| time: {suggestion.elapsed:.3f}s"
                )
                typed = typer.prompt("Your move (SAN or UCI; quit to exit)")
                if typed.lower() == "quit":
                    store.finish_game(game_id, "*")
                    console.print(f"Game {game_id} saved as unfinished in {database}")
                    return
                try:
                    move = board.parse_san(typed)
                except ValueError:
                    try:
                        move = board.parse_uci(typed)
                    except ValueError:
                        console.print("Illegal move; try again.")
                        continue
                store.record_human_move(
                    game_id, board, move, suggestion.policy, suggestion.value,
                    suggestion.nodes, suggestion.elapsed,
                )
                console.print(
                    f"Recorded: {board.san(move)} | student agreement: "
                    f"{'yes' if move == suggestion.move else 'no'}"
                )
            else:
                move = baseline_move(board, opponent, rng)
                console.print(f"Opponent: {board.san(move)}")
            board.push(move)
        outcome = board.result(claim_draw=True)
        store.finish_game(game_id, outcome)
        console.print(f"Game {game_id} result: {outcome}")
        if annotate_after:
            try:
                annotation = store.annotate_game(game_id, depth=engine_depth)
            except FileNotFoundError as error:
                console.print(str(error))
                console.print("Game retained; run chess-ai annotate-experience later.")
                return
            console.print(f"Annotated {annotation['positions']} human moves offline.")
            manifest = store.export_training(export)
            console.print(f"Exported {manifest['positions']} positions to {export}")
            if train_after and manifest["positions"]:
                _, saved = load_model(checkpoint, device)
                config = saved["config"].copy()
                config["training"] = {
                    **config["training"],
                    "max_steps": fine_tune_steps,
                    "learning_rate": min(config["training"]["learning_rate"], 1e-4),
                }
                run_dir = Path("runs") / f"assist-{game_id}"
                trained = train_model(config, export, run_dir, init_checkpoint=checkpoint)
                console.print(f"Fine-tuned checkpoint: {trained}")


@app.command("annotate-experience")
def annotate_experience(
    game_id: int = typer.Argument(..., min=1),
    database: Path = typer.Option(Path("data/processed/experience.sqlite"), exists=True),
    engine: str = typer.Option("stockfish"),
    depth: int = typer.Option(8, min=1),
) -> None:
    """Review saved human moves with Stockfish after the game is finished."""
    with ExperienceStore(database) as store:
        console.print_json(data=store.annotate_game(game_id, engine, depth))


@app.command("export-experience")
def export_experience(
    database: Path = typer.Option(Path("data/processed/experience.sqlite"), exists=True),
    output: Path = typer.Option(Path("data/processed/human-experience.jsonl")),
    min_weight: float = typer.Option(0.02, min=0, max=1),
) -> None:
    """Export annotated human moves with teacher values and quality weights."""
    with ExperienceStore(database) as store:
        console.print_json(data=store.export_training(output, min_weight))


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

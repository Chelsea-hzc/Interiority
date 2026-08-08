"""Command-line entry point for the local Gemini evaluation project."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .gemini_client import GeminiConfig, GeminiRunnerClient
from .io_utils import ensure_dirs, load_annotations
from .runners import EvaluationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini-only interiority diagnostics and zero-shot tests.")
    parser.add_argument("--data", type=Path, default=Path("df_annotation_final.csv"), help="Annotation CSV path.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/local_gemini"), help="Output folder.")
    parser.add_argument("--models", nargs="+", required=True, help="Gemini model names/IDs to evaluate.")
    parser.add_argument(
        "--run",
        default="all",
        choices=["all", "diagnostics", "final"],
        help="Which evaluation group to run. Default: all.",
    )
    parser.add_argument(
        "--backend",
        default="developer",
        choices=["developer", "vertex"],
        help="developer = Gemini Developer API key; vertex = Google Cloud / Vertex AI / enterprise environment.",
    )
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Environment variable for Developer API key.")
    parser.add_argument("--api-key", default=None, help="Developer API key. Prefer --api-key-env for shared machines.")
    parser.add_argument("--project", default=None, help="Google Cloud project for Vertex backend.")
    parser.add_argument("--location", default=None, help="Google Cloud location, e.g. global.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum output budget for each request. Default: 4096.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default="high",
        help="Thinking level for label-classification calls. Default: high.",
    )
    parser.add_argument(
        "--thinking-levels",
        nargs="+",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help=(
            "Run every model at each listed thinking level in one invocation. "
            "When provided, this takes precedence over --thinking-level."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries per request for transient 429/5xx failures. Default: 5.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for each API attempt in seconds. Use 0 to disable. Default: 60.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay after each request. Useful for preview/pro model RPM limits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true", help="Use small samples: dev sample for diagnostics and first test rows for final.")
    parser.add_argument("--pilot-n", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted final run from its saved per-row checkpoint.",
    )
    parser.add_argument(
        "--status-every",
        "--status_every",
        dest="status_every",
        type=int,
        default=30,
        help="Print progress/cost and checkpoint final-run outputs every N rows. Default: 30.",
    )
    parser.add_argument(
        "--input-cost-per-1m-tokens",
        "--input_cost_per_1m_tokens",
        dest="input_cost_per_1m_tokens",
        type=float,
        default=None,
        help="Override estimated input-token price used in terminal cost reporting.",
    )
    parser.add_argument(
        "--output-cost-per-1m-tokens",
        "--output_cost_per_1m_tokens",
        dest="output_cost_per_1m_tokens",
        type=float,
        default=None,
        help="Override estimated output-token price used in terminal cost reporting.",
    )
    return parser.parse_args()


def build_resume_command(argv: list[str]) -> str:
    safe_args = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--api-key":
            skip_next = True
            continue
        if arg.startswith("--api-key=") or arg == "--resume":
            continue
        safe_args.append(arg)
    return shlex.join([sys.executable, "-m", "gemini_interiority.cli", *safe_args, "--resume"])


def main() -> None:
    args = parse_args()
    thinking_levels = args.thinking_levels or [args.thinking_level]
    df = load_annotations(args.data)
    paths = ensure_dirs(args.output_dir)
    config = GeminiConfig(
        backend=args.backend,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        project=args.project,
        location=args.location,
        max_output_tokens=args.max_output_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        classify_thinking_level=thinking_levels[0],
        max_retries=args.max_retries,
        request_delay_seconds=args.request_delay_seconds,
    )
    client = GeminiRunnerClient(config)
    runner = EvaluationRunner(
        df=df,
        models=args.models,
        client=client,
        output_paths=paths,
        seed=args.seed,
        pilot=args.pilot,
        pilot_n=args.pilot_n,
        max_output_tokens=args.max_output_tokens,
        status_every=args.status_every,
        input_cost_per_1m_tokens=args.input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=args.output_cost_per_1m_tokens,
        resume=args.resume,
        resume_command=build_resume_command(sys.argv[1:]),
        thinking_levels=thinking_levels,
    )

    outputs = {}
    if args.run in {"all", "diagnostics"}:
        outputs.update(runner.run_diagnostics())
    if args.run in {"all", "final"}:
        outputs.update(runner.run_final_zero_shot())

    print("\nSaved outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

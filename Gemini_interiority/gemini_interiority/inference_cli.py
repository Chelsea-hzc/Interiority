"""Command-line interface for sampled-passage inference."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .inference import InferenceRunner, compact_input, sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference-only Gemini interiority classification on sampled novel passages."
    )
    parser.add_argument("--data", type=Path, default=Path("final_sample_29books.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results"))
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--backend", choices=["developer", "vertex"], default="developer")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"], default="high")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--max-invalid-retries",
        type=int,
        default=5,
        help="Retries after a malformed non-error label response. Default: 5.",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--pilot-n", type=int, default=5)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument(
        "--status-every",
        "--status_every",
        dest="status_every",
        type=int,
        default=100,
        help="Print progress and refresh final CSV/JSON status files every N rows. Default: 100.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create compact input and empty output artifacts without making API calls.",
    )
    return parser.parse_args(argv)


def build_resume_command(argv: list[str]) -> str:
    safe_args: list[str] = []
    skip_next = False
    for argument in argv:
        if skip_next:
            skip_next = False
            continue
        if argument == "--api-key":
            skip_next = True
            continue
        if argument.startswith("--api-key=") or argument in {"--resume", "--prepare-only"}:
            continue
        safe_args.append(argument)
    return shlex.join(
        [sys.executable, "-m", "gemini_interiority.inference_cli", *safe_args, "--resume"]
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = args.output_dir / "llm_input.csv"
    input_frame = compact_input(args.data, compact_path)
    client = None
    if not args.prepare_only:
        from .gemini_client import GeminiConfig, GeminiRunnerClient

        client = GeminiRunnerClient(
            GeminiConfig(
                backend=args.backend,
                api_key=args.api_key,
                api_key_env=args.api_key_env,
                project=args.project,
                location=args.location,
                max_output_tokens=args.max_output_tokens,
                request_timeout_seconds=args.request_timeout_seconds,
                request_delay_seconds=args.request_delay_seconds,
                max_retries=args.max_retries,
                classify_thinking_level=args.thinking_level,
            )
        )
    command_args = list(sys.argv[1:] if argv is None else argv)
    runner = InferenceRunner(
        input_frame=input_frame,
        client=client,
        output_dir=args.output_dir,
        model=args.model,
        backend=args.backend,
        thinking_level=args.thinking_level,
        max_output_tokens=args.max_output_tokens,
        max_invalid_retries=args.max_invalid_retries,
        pilot_n=args.pilot_n,
        status_every=args.status_every,
        resume=args.resume,
        pilot_only=args.pilot_only,
        prepare_only=args.prepare_only,
        input_sha256=sha256_file(compact_path),
        resume_command=build_resume_command(command_args),
    )
    outputs = runner.run()
    print("\nInference outputs:")
    print(f"- llm_input: {compact_path}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

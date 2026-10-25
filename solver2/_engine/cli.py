"""命令行接口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation.benchmark import run_benchmark
from .evaluation.evaluator import evaluate_solution
from .evaluation.self_test import run_evaluator_self_test
from .evaluation.noise_sweep import run_noise_sweep
from .evaluation.oracle_diagnosis import run_oracle_diagnosis
from .evaluation.performance_benchmark import run_profile_case
from .evaluation.appearance_smoke import run_appearance_smoke
from .evaluation.adaptive_uncertainty_smoke import (
    run_adaptive_uncertainty_smoke,
)
from .generator.noise import NOISE_MODELS
from .generator.partition_gallery import generate_partition_gallery
from .generator.general_partition_gallery import generate_general_partition_gallery
from .generator.random_generator import GENERATION_MODES
from .generator.export import generate_case
from .solver.solve import solve_observations
from .runtime_config import RuntimeConfig, resolve_runtime_config


def _add_common_generate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["fixed", "random"], required=True)
    parser.add_argument(
        "--surface-mode",
        choices=["white", "patterned"],
        default="white",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--noise-mm", type=float, default=5.0)
    parser.add_argument("--piece-count", type=int, choices=range(2, 5))
    parser.add_argument("--noise-model", choices=sorted(NOISE_MODELS), default="disk_bounded_vertex")
    parser.add_argument(
        "--generation-mode", "--generator-mode",
        dest="generation_mode",
        choices=GENERATION_MODES,
        default="mixed",
        help="random 模式的分割后端",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m solver2", description="A4 拼图几何与运动规划 Demo")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="生成 public/truth 隔离案例")
    _add_common_generate(generate)
    generate.add_argument("--output", required=True)
    generate.add_argument("--target-width", type=float)
    generate.add_argument("--target-height", type=float)

    solve = sub.add_parser("solve", help="仅用 observations.json 求解")
    solve.add_argument("--input", required=True)
    solve.add_argument("--output", required=True)
    solve.add_argument("--machine-calibration")
    solve.add_argument("--verbose", action="store_true")
    solve.add_argument("--runtime-config")
    solve.add_argument(
        "--recovery",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    solve.add_argument("--recovery-deadline-seconds", type=float)
    solve.add_argument(
        "--surface-mode", choices=["white", "patterned"]
    )
    solve.add_argument("--geometry-ambiguity-threshold", type=float)
    solve.add_argument("--appearance-winner-margin", type=float)
    solve.add_argument(
        "--diagnostics-level", choices=["off", "summary", "full"]
    )
    solve.add_argument("--target-seam-gap-mm", type=float)
    solve.add_argument(
        "--execution-policy",
        choices=[
            "strict",
            "competition_best_effort",
            "competition_force_output",
        ],
    )

    evaluate = sub.add_parser("evaluate", help="用真值评估求解输出")
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--solution", required=True)

    demo = sub.add_parser("demo", help="一键生成、求解、评估")
    _add_common_generate(demo)
    demo.add_argument("--output")
    demo.add_argument("--machine-calibration")

    benchmark = sub.add_parser("benchmark", help="批量基准")
    _add_common_generate(benchmark)
    benchmark.add_argument("--cases", type=int, required=True)
    benchmark.add_argument("--output")
    benchmark.add_argument("--runtime-config")

    evaluator_self_test = sub.add_parser("evaluator-self-test", help="用 perfect truth 验证评估器")
    evaluator_self_test.add_argument("--mode", choices=["fixed", "random"], required=True)
    evaluator_self_test.add_argument("--cases", type=int, required=True)
    evaluator_self_test.add_argument("--seed", type=int, required=True)
    evaluator_self_test.add_argument("--output")

    noise_sweep = sub.add_parser("noise-sweep", help="按噪声等级批量评估")
    noise_sweep.add_argument("--mode", choices=["fixed", "random"], required=True)
    noise_sweep.add_argument("--noise-values", required=True)
    noise_sweep.add_argument("--cases-per-level", type=int, required=True)
    noise_sweep.add_argument("--seed", type=int, required=True)
    noise_sweep.add_argument("--output")
    noise_sweep.add_argument("--noise-model", choices=sorted(NOISE_MODELS), default="disk_bounded_vertex")
    noise_sweep.add_argument(
        "--generation-mode", "--generator-mode",
        dest="generation_mode",
        choices=GENERATION_MODES,
        default="mixed",
        help="random 噪声阶梯使用的分割生成后端",
    )
    noise_sweep.add_argument("--runtime-config")

    oracle = sub.add_parser("oracle-diagnosis", help="仅供离线评估的 Oracle 分层诊断")
    oracle.add_argument("--mode", choices=["fixed", "random"], required=True)
    oracle.add_argument("--noise-values", required=True)
    oracle.add_argument("--cases-per-level", type=int, required=True)
    oracle.add_argument("--seed", type=int, required=True)
    oracle.add_argument("--output")

    gallery = sub.add_parser("partition-gallery", help="仅生成随机铁皮分割效果图")
    gallery.add_argument("--count", type=int, required=True)
    gallery.add_argument("--seed", type=int, required=True)
    gallery.add_argument("--output", required=True)
    general_gallery = sub.add_parser(
        "general-partition-gallery",
        help="生成 mixed 通用直线分割图库及多样性统计",
    )
    general_gallery.add_argument("--count", type=int, default=1000)
    general_gallery.add_argument("--seed", type=int, default=10000)
    general_gallery.add_argument(
        "--output", default="outputs/general_partition_gallery"
    )
    profile_case = sub.add_parser(
        "profile-case", help="重复运行单案例并输出分阶段性能报告"
    )
    profile_case.add_argument("--input", required=True)
    profile_case.add_argument("--output", required=True)
    profile_case.add_argument("--repeat", type=int, default=10)
    profile_case.add_argument("--runtime-config")
    appearance_smoke = sub.add_parser(
        "appearance-smoke",
        help="运行白色/牌面候选消歧的小规模确定性冒烟测试",
    )
    appearance_smoke.add_argument(
        "--output", default="outputs/appearance_ambiguity_smoke"
    )
    adaptive_smoke = sub.add_parser(
        "adaptive-uncertainty-smoke",
        help="运行隐藏测试噪声标签的0/1/2mm小规模渐进搜索验证",
    )
    adaptive_smoke.add_argument(
        "--output", default="outputs/adaptive_uncertainty_smoke"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        path = generate_case(
            args.mode, args.seed, args.noise_mm, args.output, args.piece_count,
            args.target_width, args.target_height, noise_model=args.noise_model,
            generation_mode=args.generation_mode,
            surface_mode=args.surface_mode,
        )
        print(f"案例已生成：{path.resolve()}")
    elif args.command == "solve":
        runtime_config = args.runtime_config
        if (
            args.recovery is not None
            or args.recovery_deadline_seconds is not None
            or args.geometry_ambiguity_threshold is not None
            or args.appearance_winner_margin is not None
            or args.target_seam_gap_mm is not None
            or args.execution_policy is not None
        ):
            loaded = resolve_runtime_config(runtime_config)
            values = loaded.as_dict()
            if args.recovery is not None:
                values["enable_recovery"] = bool(args.recovery)
            if args.recovery_deadline_seconds is not None:
                values["recovery_total_deadline_seconds"] = float(
                    args.recovery_deadline_seconds
                )
            if args.geometry_ambiguity_threshold is not None:
                values["geometry_ambiguity_threshold"] = float(
                    args.geometry_ambiguity_threshold
                )
            if args.appearance_winner_margin is not None:
                values["appearance_winner_margin"] = float(
                    args.appearance_winner_margin
                )
            if args.target_seam_gap_mm is not None:
                values["target_seam_gap_mm"] = float(
                    args.target_seam_gap_mm
                )
            if args.execution_policy is not None:
                values["execution_policy"] = args.execution_policy
            runtime_config = RuntimeConfig.from_mapping(values)
        result = solve_observations(
            args.input, args.output, args.machine_calibration, args.verbose,
            runtime_config=runtime_config,
            diagnostics_level=args.diagnostics_level,
            surface_mode=args.surface_mode,
        )
        if not result.get("success"):
            raise SystemExit(2)
    elif args.command == "evaluate":
        result = evaluate_solution(args.truth, args.solution)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "demo":
        output = Path(args.output or f"outputs/{args.mode}_seed_{args.seed:04d}")
        generate_case(
            args.mode, args.seed, args.noise_mm, output, args.piece_count,
            noise_model=args.noise_model, generation_mode=args.generation_mode,
            surface_mode=args.surface_mode,
        )
        solution = solve_observations(
            output / "public" / "observations.json",
            output / "solution",
            args.machine_calibration,
            surface_mode=args.surface_mode,
        )
        evaluation = evaluate_solution(output / "truth" / "ground_truth.json", output / "solution" / "solution.json")
        (output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n评估通过推荐阈值：{'是' if evaluation.get('passes_recommended_thresholds') else '否'}")
        print(f"输出目录：{output.resolve()}")
        if not solution.get("success"):
            raise SystemExit(2)
    elif args.command == "benchmark":
        output = Path(args.output or f"outputs/benchmark_{args.mode}_{args.seed}")
        result = run_benchmark(
            args.mode, args.cases, args.noise_mm, args.seed, output,
            args.piece_count, args.noise_model, args.generation_mode,
            args.runtime_config,
            surface_mode=args.surface_mode,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "cases_detail"}, ensure_ascii=False, indent=2))
    elif args.command == "evaluator-self-test":
        output = Path(args.output or f"outputs/evaluator_self_test_{args.mode}_{args.seed}")
        result = run_evaluator_self_test(args.mode, args.cases, args.seed, output)
        print(json.dumps({k: v for k, v in result.items() if k != "cases_detail"}, ensure_ascii=False, indent=2))
        if result["failed"]:
            raise SystemExit(2)
    elif args.command == "noise-sweep":
        values = [float(value.strip()) for value in args.noise_values.split(",") if value.strip()]
        output = Path(args.output or f"outputs/noise_sweep_{args.mode}_{args.seed}")
        result = run_noise_sweep(
            args.mode, values, args.cases_per_level, args.seed, output,
            args.noise_model, args.generation_mode,
            args.runtime_config,
        )
        concise = [
            {
                k: v
                for k, v in level.items()
                if k not in {
                    "cases_detail", "by_topology", "by_adjacency_signature"
                }
            }
            for level in result["levels"]
        ]
        print(json.dumps(concise, ensure_ascii=False, indent=2))
    elif args.command == "oracle-diagnosis":
        values = [float(value.strip()) for value in args.noise_values.split(",") if value.strip()]
        output = Path(args.output or f"outputs/oracle_diagnosis_{args.mode}_{args.seed}")
        result = run_oracle_diagnosis(args.mode, values, args.cases_per_level, args.seed, output)
        concise = [{k: v for k, v in level.items() if k != "cases_detail"} for level in result["levels"]]
        print(json.dumps(concise, ensure_ascii=False, indent=2))
    elif args.command == "partition-gallery":
        result = generate_partition_gallery(args.count, args.seed, args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False, indent=2))
    elif args.command == "general-partition-gallery":
        result = generate_general_partition_gallery(
            args.count, args.seed, args.output
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "cases"},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "profile-case":
        result = run_profile_case(
            args.input, args.output, args.repeat, args.runtime_config
        )
        print(json.dumps(
            {key: value for key, value in result.items() if key != "runs"},
            ensure_ascii=False, indent=2,
        ))
    elif args.command == "appearance-smoke":
        result = run_appearance_smoke(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "adaptive-uncertainty-smoke":
        result = run_adaptive_uncertainty_smoke(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))

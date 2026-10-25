#!/usr/bin/env python3
"""Q6A 拼图任务入口：默认无外设模式，只报告状态。

显式 --hardware 才允许设备操作；本地图片 HTP 后端已接入，相机任务待验收。
配置完整后可使用 --hardware --execute 进入原有人工建立原点/停靠流程。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from q6a_platform import VISION_UNAVAILABLE, add_mode_arguments, load_config, validate_motor_port
from hardware_discovery import resolve_writer_port
from task1_adapter import (
    TangramAdapterConfig,
    TangramHardwarePorts,
    TangramPlanExecutor,
    adapt_move_plan,
    load_rotation_calibration,
    load_safe_parking_config,
    project_root,
)


REFERENCE_DIR = project_root() / "reference-vision" / "ec_26_e"
REFERENCE_SRC = REFERENCE_DIR / "src"
COLLEAGUE_DIR = project_root() / "vision"
COLLEAGUE_TASK_CONFIG = COLLEAGUE_DIR / "configs" / "tasks" / "task1.json"
DEFAULT_PLAN_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "task1_plans"
DEFAULT_ROTATION_CALIBRATION = Path(__file__).with_name(
    "task1.tangram.rotation.json"
)
DEFAULT_SAFE_PARKING_CONFIG = project_root() / "host" / "safe_parking.json"
# 全部任务默认使用现场确认的快速参数。上位机可显式传入现场参数；随机题
# 即使求解耗时较长，也绝不能在其后静默改写操作者选择的执行速度。
RANDOM_SPEEDUP_AFTER_SOLVER_SECONDS = 60.0
RANDOM_LATE_SOLVER_XY_FEED_MM_MIN = 4500.0
RANDOM_LATE_SOLVER_ROTATION_RPM = 29


def _execution_xy_feed_mm_min(
    task_profile: str,
    solver_elapsed_seconds: float,
    normal_xy_feed_mm_min: float = 4500.0,
) -> float:
    """保留调用契约，但始终尊重上位机明确传入的 XY 速度。"""

    # 旧逻辑在随机题超过 60 秒后改用常量 4500；常量恰好等于旧默认值，
    # 却会覆盖操作者在上位机设置的现场速度。现在默认值仍是 4500，只有
    # 使用者主动选择时才改变，绝不按求解耗时静默改写。
    del task_profile, solver_elapsed_seconds
    return normal_xy_feed_mm_min


def _execution_motion_settings(
    task_profile: str,
    solver_elapsed_seconds: float,
    normal_xy_feed_mm_min: float,
    normal_rotation_rpm: int,
) -> tuple[float, int]:
    """统一选择随机题后段执行的 XY 与 F32C 转速。"""

    xy_feed = _execution_xy_feed_mm_min(
        task_profile, solver_elapsed_seconds, normal_xy_feed_mm_min
    )
    rotation_rpm = (
        RANDOM_LATE_SOLVER_ROTATION_RPM
        if xy_feed != normal_xy_feed_mm_min
        else normal_rotation_rpm
    )
    return xy_feed, rotation_rpm


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """上位机任务与求解坐标、视觉配置的固定对应关系。"""

    name: str
    task_config: Path
    plan_output_dir: Path
    reference_width_mm: float
    reference_height_mm: float
    surface_mode: str
    use_new_random_solver: bool = False
    use_patterned_solver: bool = False
    planning_start_xy_mm: tuple[float, float] | None = None


TASK_PROFILES = {
    "fixed": TaskProfile(
        name="固定四片拼图",
        task_config=COLLEAGUE_TASK_CONFIG,
        plan_output_dir=DEFAULT_PLAN_OUTPUT_DIR,
        reference_width_mm=210.0,
        reference_height_mm=297.0,
        surface_mode="white",
    ),
    "random-white": TaskProfile(
        name="白色随机碎片（自动识别 2～4 片）",
        task_config=COLLEAGUE_DIR / "configs" / "tasks" / "task2.json",
        plan_output_dir=Path(__file__).resolve().parent / "outputs" / "task2_white_plans",
        reference_width_mm=207.0,
        reference_height_mm=292.0,
        surface_mode="white",
        use_new_random_solver=True,
        # 真实起点在纸外安全停靠点；这里是其在随机求解坐标中的纸内投影，
        # 仅决定抓取排序，不会生成实际运动坐标。
        planning_start_xy_mm=(207.0, 292.0),
    ),
    "random-patterned": TaskProfile(
        name="扑克牌图案随机碎片（几何 + 现场花纹消歧）",
        task_config=COLLEAGUE_DIR / "configs" / "tasks" / "task2.json",
        plan_output_dir=Path(__file__).resolve().parent / "outputs" / "task2_patterned_plans",
        reference_width_mm=207.0,
        reference_height_mm=292.0,
        surface_mode="patterned",
        use_patterned_solver=True,
        planning_start_xy_mm=(207.0, 292.0),
    ),
}


def _reference_imports() -> tuple[Any, ...]:
    raise RuntimeError("旧传统视觉参考工程未包含在本包中；" + VISION_UNAVAILABLE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_mode_arguments(parser)
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(__file__).with_name("task1.tangram.env"),
        help="参考 Task1 相机/视觉环境配置",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        help="覆盖当前任务档案的 JSON 配置；traditional 使用旧参考配置",
    )
    parser.add_argument(
        "--task-profile",
        choices=tuple(TASK_PROFILES),
        default="fixed",
        help="上位机任务档案：固定或两种随机外轮廓拼图",
    )
    parser.add_argument(
        "--vision",
        choices=("yolo", "traditional"),
        default="yolo",
        help="视觉/规划链路；yolo 为已实测鲁棒链路，traditional 为旧参考链路",
    )
    parser.add_argument(
        "--params",
        choices=("current", "new"),
        default="new",
        help="A4 seed 参数预设",
    )
    parser.add_argument(
        "--landscape-rotation",
        choices=("clockwise", "counterclockwise"),
        default="clockwise",
        help=(
            "本机横向蓝纸转为竖向题目 A4 的方向；"
            "本机碎片在左、目标在右，默认 clockwise"
        ),
    )
    parser.add_argument(
        "--process-fps",
        type=float,
        default=6.0,
        help="YOLO 视觉处理频率，Hz",
    )
    parser.add_argument(
        "--acquisition-timeout",
        type=float,
        help="稳定视觉采集总超时，秒；省略时读取 Task1 JSON",
    )
    parser.add_argument(
        "--camera-device",
        help=(
            "由上位机现场设备选择器确认的摄像头路径；省略时使用环境文件并自动发现"
        ),
    )
    parser.add_argument(
        "--plan-output-dir",
        type=Path,
        default=DEFAULT_PLAN_OUTPUT_DIR,
        help="保存求解图、solution.json 和诊断日志的目录",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="显示鲁棒视觉实时诊断；默认全屏",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="视觉诊断使用窗口模式；默认全屏",
    )
    parser.add_argument("--execute", action="store_true", help="经 RUN 确认后连接本机硬件")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="仅采集、识别、解算和输出计划；硬性禁止连接任何运动硬件",
    )
    parser.add_argument(
        "--reuse-machine-origin",
        action="store_true",
        help=(
            "复用同一次上电时已建立的坐标；串口重连后会把当前安全停靠点重新锚定为"
            "其业务坐标，不会错误地把它设为机械零点"
        ),
    )
    parser.add_argument(
        "--park-only",
        action="store_true",
        help="不采集视觉或执行拼图；复用本次上电坐标，直接回记录的安全停靠点",
    )
    parser.add_argument(
        "--initialize-and-park",
        action="store_true",
        help=(
            "机构已人工位于机械限位角时，建立本次上电原点并直接前往"
            "记录的安全停靠点；不采集视觉或执行拼图"
        ),
    )
    parser.add_argument(
        "--allow-competition-compliant",
        action="store_true",
        help=(
            "测试时允许人工放行已通过竞赛名义几何门、但未通过严格鲁棒门的方案；"
            "默认仍拒绝"
        ),
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        help="测试时只执行计划前 N 片；首次实机建议设为 1",
    )
    parser.add_argument(
        "--travel-lift-mm",
        type=float,
        default=10.0,
        help="携件与 XY 运动时的正数抬高高度，mm",
    )
    parser.add_argument(
        "--pickup-lift-mm",
        type=float,
        default=2.0,
        help="吸取时电磁铁距自然下垂 Z0 的正数抬高，mm",
    )
    parser.add_argument(
        "--place-lift-mm",
        type=float,
        default=2.0,
        help="释放时电磁铁距自然下垂 Z0 的正数抬高，mm",
    )
    parser.add_argument(
        "--rest-lift-mm",
        type=float,
        default=10.0,
        help="每片完成后的正数抬高高度，mm",
    )
    parser.add_argument(
        "--xy-feed-mm-min",
        type=float,
        default=4500.0,
        help="正式搬运 XY 进给速度，mm/min",
    )
    parser.add_argument(
        "--z-feed-mm-min",
        type=float,
        default=1500.0,
        help="正式搬运 Z 进给速度，mm/min",
    )
    parser.add_argument(
        "--rotation-rpm",
        type=int,
        default=29,
        help="F32C 拼图旋转转速，RPM",
    )
    parser.add_argument(
        "--magnet-pickup-dwell-seconds",
        type=float,
        default=0.5,
        help="电磁铁吸合后等待时间，秒",
    )
    parser.add_argument(
        "--magnet-release-dwell-seconds",
        type=float,
        default=0.5,
        help="电磁铁释放后等待时间，秒",
    )
    parser.add_argument(
        "--continuous-transfers",
        action="store_true",
        help=(
            "连续搬运：成功放下一片并抬到安全 Z 后直接去下一取件点；"
            "默认每片回蓝纸原点，异常时始终回原点"
        ),
    )
    parser.add_argument(
        "--safe-parking-config",
        type=Path,
        default=DEFAULT_SAFE_PARKING_CONFIG,
        help="host 记录的任务结束和异常恢复安全停靠点 JSON",
    )
    parser.add_argument(
        "--rotation-sign",
        type=int,
        choices=(-1, 1),
        help="视觉顺时针到 F32C 机械正方向：+1 同向，-1 反向；覆盖标定文件",
    )
    parser.add_argument(
        "--rotation-calibration",
        type=Path,
        default=DEFAULT_ROTATION_CALIBRATION,
        help="由 calibrate_f32c_rotation.py 生成的旋转方向标定 JSON",
    )
    parser.add_argument(
        "--writer-port",
        default=load_config()["writer_port"],
        help="写字机 USB 串口；配置路径失效时自动采用唯一候选",
    )
    parser.add_argument("--motor-port", default=load_config()["motor_port"])
    parser.add_argument("--motor-address", type=int, default=2)
    return parser


def _hardware_ports(args: argparse.Namespace) -> TangramHardwarePorts:
    """执行前才解析可变 USB 写字机；F32C 板载 UART 保持原配置。"""

    motor_port = validate_motor_port(args.motor_port)
    return TangramHardwarePorts(
        writer_port=resolve_writer_port(args.writer_port),
        motor_port=motor_port,
        motor_address=args.motor_address,
    )


def _capture_traditional_plan(args: argparse.Namespace) -> object:
    if args.task_profile != "fixed":
        raise ValueError("随机任务只支持已实测的 yolo 鲁棒视觉链路")
    (
        camera_factory,
        load_task1_config,
        preview_factory,
        observer_factory,
        run_task1,
        detector_factory,
        resize_factory,
    ) = _reference_imports()
    task_config = args.task_config or REFERENCE_DIR / "configs" / "tasks" / "task1.json"
    config = load_task1_config(task_config, env_path=args.env)
    # 本装置当前识别的是白色拼图；不修改参考 JSON，只在运行时覆盖其模式。
    config = replace(config, detect1=replace(config.detect1, piece_mode=1))
    # 现场预览必须足够显眼，避免普通窗口被 Plasma 桌面或其他窗口遮挡。
    observer = (
        observer_factory(preview_factory(fullscreen=True))
        if args.draw
        else None
    )
    return run_task1(
        camera_factory(config.camera),
        resize_factory(config.resize),
        detector_factory(config.detect1),
        options=config.options,
        observer=observer,
    )


def _capture_yolo_plan(args: argparse.Namespace) -> object:
    from task1_yolo_flow import acquire_yolo_move_plan, load_yolo_task_config

    profile = TASK_PROFILES[args.task_profile]
    config = load_yolo_task_config(
        args.env,
        args.task_config or profile.task_config,
        params_name=args.params,
        process_fps=args.process_fps,
        acquisition_timeout_seconds=args.acquisition_timeout,
        plan_output_dir=(
            args.plan_output_dir
            if args.plan_output_dir != DEFAULT_PLAN_OUTPUT_DIR
            else profile.plan_output_dir
        ),
        draw=args.draw,
        windowed=args.windowed,
        allow_competition_compliant=args.allow_competition_compliant,
        landscape_rotation=args.landscape_rotation,
        reference_width_mm=profile.reference_width_mm,
        reference_height_mm=profile.reference_height_mm,
        surface_mode=profile.surface_mode,
        use_new_random_solver=profile.use_new_random_solver,
        use_patterned_solver=profile.use_patterned_solver,
        planning_start_xy_mm=profile.planning_start_xy_mm,
        camera_override=args.camera_device,
    )
    return acquire_yolo_move_plan(config)


def _capture_plan(args: argparse.Namespace) -> object:
    if args.vision == "yolo":
        return _capture_yolo_plan(args)
    return _capture_traditional_plan(args)


def _clamp_execution_parameter(
    args: argparse.Namespace,
    attribute: str,
    option: str,
    lower: float,
    upper: float,
    fallback: float,
) -> float:
    """限幅可恢复的现场执行参数，避免一项超界使整题未启动。"""

    original = float(getattr(args, attribute))
    bounded = fallback if not math.isfinite(original) else min(upper, max(lower, original))
    if bounded != original:
        print(
            f"⚠ 参数限幅：{option}={original:g} → {bounded:g} "
            f"（允许范围 {lower:g}～{upper:g}）；继续执行。",
            flush=True,
        )
    setattr(args, attribute, bounded)
    return bounded


def _normalise_execution_parameters(args: argparse.Namespace) -> None:
    """统一处理 GUI/命令行的可限幅执行参数，不改变正常默认值。"""

    pickup = _clamp_execution_parameter(
        args, "pickup_lift_mm", "--pickup-lift-mm", 0.0, 14.5, 2.0
    )
    place = _clamp_execution_parameter(
        args, "place_lift_mm", "--place-lift-mm", 0.0, 14.5, 2.0
    )
    _clamp_execution_parameter(
        args, "rest_lift_mm", "--rest-lift-mm", 0.0, 15.0, 10.0
    )
    # safe_z 必须严格高于（数值更负于）取/放 Z；保留 0.5 mm 余量，
    # 否则即使坐标在范围内也会因无安全抬高而无法建立控制器配置。
    travel_floor = max(0.5, pickup + 0.5, place + 0.5)
    _clamp_execution_parameter(
        args, "travel_lift_mm", "--travel-lift-mm", travel_floor, 15.0, 10.0
    )
    _clamp_execution_parameter(
        args, "xy_feed_mm_min", "--xy-feed-mm-min", 500.0, 6000.0, 4500.0
    )
    _clamp_execution_parameter(
        args, "z_feed_mm_min", "--z-feed-mm-min", 500.0, 3000.0, 1500.0
    )
    _clamp_execution_parameter(
        args, "magnet_pickup_dwell_seconds", "--magnet-pickup-dwell-seconds", 0.1, 2.0, 0.5
    )
    _clamp_execution_parameter(
        args, "magnet_release_dwell_seconds", "--magnet-release-dwell-seconds", 0.1, 2.0, 0.5
    )
    original_rpm = int(args.rotation_rpm)
    bounded_rpm = min(30, max(1, original_rpm))
    if bounded_rpm != original_rpm:
        print(
            f"⚠ 参数限幅：--rotation-rpm={original_rpm} → {bounded_rpm} "
            "（允许范围 1～30）；继续执行。",
            flush=True,
        )
    args.rotation_rpm = bounded_rpm


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.hardware:
        if args.execute or args.park_only or args.initialize_and_park or args.reuse_machine_origin:
            print("无外设模式禁止执行、停靠或建立原点；需要显式 --hardware。", file=sys.stderr)
            return 2
        print("Q6A 无外设模式：未打开摄像头、串口、GPIO 或 NPU。")
        print(VISION_UNAVAILABLE)
        return 0
    if not (args.park_only or args.initialize_and_park):
        print(VISION_UNAVAILABLE, file=sys.stderr)
        return 2
    try:
        validate_motor_port(args.motor_port)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    _normalise_execution_parameters(args)
    if args.max_moves is not None and args.max_moves <= 0:
        print("⛔ --max-moves 必须是正整数", file=sys.stderr)
        return 2
    if args.park_only and args.initialize_and_park:
        print("⛔ --park-only 与 --initialize-and-park 不能同时指定", file=sys.stderr)
        return 2
    if args.plan_only and args.execute:
        print("⛔ --plan-only 与 --execute 不能同时指定", file=sys.stderr)
        return 2
    if args.plan_only and (
        args.park_only or args.initialize_and_park or args.reuse_machine_origin
    ):
        print(
            "⛔ --plan-only 不允许停靠、初始化或复用机械原点参数",
            file=sys.stderr,
        )
        return 2
    if (args.park_only or args.initialize_and_park) and not args.execute:
        print("⛔ 停靠操作必须同时指定 --execute", file=sys.stderr)
        return 2
    if args.park_only and not args.reuse_machine_origin:
        print(
            "⛔ --park-only 只能复用本次上电已建立的机械原点；"
            "请先完成一次初始化任务。",
            file=sys.stderr,
        )
        return 2
    if args.initialize_and_park and args.reuse_machine_origin:
        print(
            "⛔ --initialize-and-park 会建立机械原点，不能同时复用安全停靠点坐标。",
            file=sys.stderr,
        )
        return 2
    rotation_sign = args.rotation_sign
    if args.execute and not (args.park_only or args.initialize_and_park) and rotation_sign is None:
        try:
            rotation_sign = load_rotation_calibration(
                args.rotation_calibration
            ).vision_clockwise_to_motor_sign
        except FileNotFoundError:
            print(
                "⛔ 未指定 --rotation-sign，且找不到旋转标定文件；"
                "请先运行 calibrate_f32c_rotation.py",
                file=sys.stderr,
            )
            return 2
        except Exception as error:
            print(f"⛔ 无法使用旋转标定文件：{error}", file=sys.stderr)
            return 2
    try:
        parking_x, parking_y = load_safe_parking_config(args.safe_parking_config)
        profile = TASK_PROFILES[args.task_profile]
        adapter_config = TangramAdapterConfig(
            vision_clockwise_to_motor_sign=rotation_sign,
            # 顺时针模式的竖向 A4 坐标与 example 的横向视觉坐标相差 180°。
            # 在适配到写字机时翻转两轴，保持 source 在左、target 在右。
            reverse_reference_axes=args.landscape_rotation == "clockwise",
            reference_width_mm=profile.reference_width_mm,
            reference_height_mm=profile.reference_height_mm,
            safe_z_mm=-args.travel_lift_mm,
            pickup_z_mm=-args.pickup_lift_mm,
            place_z_mm=-args.place_lift_mm,
            rest_z_mm=-args.rest_lift_mm,
            continuous_transfers=args.continuous_transfers,
            safe_parking_x_mm=parking_x,
            safe_parking_y_mm=parking_y,
            xy_feed_mm_min=args.xy_feed_mm_min,
            z_feed_mm_min=args.z_feed_mm_min,
            rotation_rpm=args.rotation_rpm,
            magnet_pickup_dwell_seconds=args.magnet_pickup_dwell_seconds,
            magnet_release_dwell_seconds=args.magnet_release_dwell_seconds,
        )
        if not (args.park_only or args.initialize_and_park):
            reference_plan = _capture_plan(args)
            solver_elapsed_seconds = float(
                getattr(reference_plan, "solver_elapsed_seconds", 0.0)
            )
            selected_xy_feed, selected_rotation_rpm = _execution_motion_settings(
                args.task_profile,
                solver_elapsed_seconds,
                adapter_config.xy_feed_mm_min,
                adapter_config.rotation_rpm,
            )
            if (
                selected_xy_feed != adapter_config.xy_feed_mm_min
                or selected_rotation_rpm != adapter_config.rotation_rpm
            ):
                adapter_config = replace(
                    adapter_config,
                    xy_feed_mm_min=selected_xy_feed,
                    rotation_rpm=selected_rotation_rpm,
                )
                print(
                    "随机求解耗时 "
                    f"{solver_elapsed_seconds:.1f} 秒（≥"
                    f"{RANDOM_SPEEDUP_AFTER_SOLVER_SECONDS:g} 秒）；"
                    "本次后续 XY 自动提速至 "
                    f"{RANDOM_LATE_SOLVER_XY_FEED_MM_MIN:g} mm/min，"
                    "F32C 无刷转速自动提至 "
                    f"{RANDOM_LATE_SOLVER_ROTATION_RPM} RPM。",
                    flush=True,
                )
            tangram_plan = adapt_move_plan(reference_plan, adapter_config)
            if args.max_moves is not None:
                tangram_plan = replace(
                    tangram_plan,
                    moves=tangram_plan.moves[: args.max_moves],
                )
    except KeyboardInterrupt:
        print("\n视觉预检已取消；未连接任何运动硬件。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"⛔ 视觉、规划或本机限位预检失败：{error}", file=sys.stderr)
        return 1

    if args.park_only or args.initialize_and_park:
        if args.initialize_and_park:
            description = (
                "初始化安全停靠：把当前位置定义为本次上电机械原点，"
                f"再移动到蓝纸安全点=({parking_x:g}, {parking_y:g}) mm。"
            )
        else:
            description = (
                "安全停靠：复用本次上电已建立的机械坐标，"
                f"目标蓝纸坐标=({parking_x:g}, {parking_y:g}) mm。"
            )
        print(description)
        if input("确认机构路径安全后输入 RUN： ").strip().upper() != "RUN":
            print("已取消：未连接硬件。")
            return 0
        try:
            executor = TangramPlanExecutor(
                adapter_config,
                _hardware_ports(args),
            )
            if args.initialize_and_park:
                executor.initialize_and_park(
                    origin_ready_callback=lambda _reused: print(
                        "硬件坐标基准状态：已在当前位置建立本次上电原点",
                        flush=True,
                    )
                )
            else:
                executor.park_safely(
                    origin_ready_callback=lambda _reused: print(
                        "硬件坐标基准状态：已将当前位置重新锚定为安全停靠点",
                        flush=True,
                    )
                )
        except KeyboardInterrupt:
            print("\n安全停靠已中断；底层控制器已尝试安全恢复。", file=sys.stderr)
            return 130
        except Exception as error:
            print(f"⛔ 安全停靠失败：{error}", file=sys.stderr)
            return 1
        print("安全停靠完成。")
        return 0

    print(json.dumps(tangram_plan.to_dict(), ensure_ascii=False, indent=2))
    print(
        "本次动作配置："
        f"执行 {len(tangram_plan.moves)} 片；"
        f"XY 进给 {adapter_config.xy_feed_mm_min:g} mm/min；"
        f"搬运抬高 {args.travel_lift_mm:g} mm；"
        f"吸取/释放抬高 {args.pickup_lift_mm:g}/"
        f"{args.place_lift_mm:g} mm；"
        f"结束抬高 {args.rest_lift_mm:g} mm。"
    )
    if not args.execute:
        print(
            "纯检测解算完成：未连接写字机、F32C 或电磁铁。"
            if args.plan_only
            else "预览完成：未连接写字机、F32C 或电磁铁。"
        )
        return 0
    if args.allow_competition_compliant:
        print(
            "⚠ 当前启用了竞赛合规测试放行；仅满足受限放行条件的方案"
            "才会进入硬件确认。",
            file=sys.stderr,
            flush=True,
        )
    if input("确认计划、机构原点与路径安全后输入 RUN： ").strip().upper() != "RUN":
        print("已取消：未连接硬件。")
        return 0
    try:
        executor = TangramPlanExecutor(
            adapter_config,
            _hardware_ports(args),
        )
        results = executor.execute(
            tangram_plan,
            reuse_machine_origin=args.reuse_machine_origin,
            origin_ready_callback=lambda reused: print(
                "硬件坐标基准状态："
                + (
                    "已将当前位置重新锚定为安全停靠点"
                    if reused
                    else "已在当前位置建立本次上电原点"
                ),
                flush=True,
            ),
            completion_callback=lambda: print(
                "拼图完成信号：最后一片已释放。",
                flush=True,
            ),
        )
    except KeyboardInterrupt:
        print("\n任务已中断；底层控制器已尝试安全恢复。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"⛔ 硬件任务失败：{error}", file=sys.stderr)
        return 1
    print(f"任务完成：共执行 {len(results)} 片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sys
from time import perf_counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pso_core import PSOConfig, make_quadratic, make_rastrigin, pso, summarize_run


ASSETS = ROOT / "assets"
TIMED_RUNS = 20


def surface_grid(objective, lim: float, n: int = 220):
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(x, y)
    pts = np.stack([xx, yy], axis=-1)
    zz = objective(pts)
    return xx, yy, zz


def save_pso_scheme() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    pos = np.array([2.0, 1.55])
    pbest = np.array([3.7, 4.05])
    gbest = np.array([8.25, 3.65])
    vel_end = np.array([3.35, 1.0])
    new_pos = np.array([5.25, 2.55])

    ax.scatter(*pos, s=130, color="#1f77b4", label="текущее положение")
    ax.scatter(*pbest, s=120, color="#ff7f0e", label="личный лучший опыт")
    ax.scatter(*gbest, s=140, color="#2ca02c", marker="*", label="лучшее в рое")
    ax.scatter(*new_pos, s=140, color="#d62728", label="следующее положение")

    arrows = [
        (pos, vel_end, "#4c78a8", "инерция", (2.65, 0.55)),
        (pos, pbest, "#f58518", "личная память", (2.35, 3.25)),
        (pos, gbest, "#54a24b", "притяжение к лучшему роя", (5.65, 4.12)),
        (pos, new_pos, "#d62728", "итоговый шаг", (4.0, 2.05)),
    ]
    for start, end, color, label, text_pos in arrows:
        delta = end - start
        ax.arrow(
            start[0],
            start[1],
            delta[0] * 0.9,
            delta[1] * 0.9,
            head_width=0.12,
            length_includes_head=True,
            color=color,
            alpha=0.9,
        )
        ax.text(
            text_pos[0],
            text_pos[1],
            label,
            color=color,
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.8},
        )

    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "pso_step_scheme.png", bbox_inches="tight")
    plt.close(fig)


def save_surfaces() -> None:
    specs = [
        ("quadratic", make_quadratic(10.0), 4.0, "Сильно выпуклая квадратичная функция"),
        ("rastrigin", make_rastrigin(10.0), 5.12, "Функция Растригина"),
    ]
    for name, objective, lim, title in specs:
        xx, yy, zz = surface_grid(objective, lim)

        fig = plt.figure(figsize=(7.2, 5.2), dpi=180)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        ax.set_title(title, pad=12)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$f(x)$")
        ax.view_init(elev=35, azim=-55)
        fig.tight_layout()
        fig.savefig(ASSETS / f"{name}_surface.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.3, 5.2), dpi=180)
        levels = 28 if name == "quadratic" else 45
        contour = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis")
        ax.contour(xx, yy, zz, levels=levels, colors="white", alpha=0.25, linewidths=0.35)
        ax.scatter([0], [0], marker="*", s=160, color="#ffcc00", edgecolor="black", linewidth=0.6)
        ax.set_title(title + ": линии уровня")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(contour, ax=ax, label="$f(x)$")
        fig.tight_layout()
        fig.savefig(ASSETS / f"{name}_contour.png", bbox_inches="tight")
        plt.close(fig)


def save_convergence_plot(histories: dict[str, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.3), dpi=180)
    for label, history in histories.items():
        ax.plot(history["best_values"], label=label, linewidth=2)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("номер итерации")
    ax.set_ylabel("лучшее найденное значение")
    ax.set_title("Базовая сходимость PSO")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "pso_convergence.png", bbox_inches="tight")
    plt.close(fig)


def save_parameter_comparison(experiments: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=180)
    groups = [
        ("quadratic", "Квадратичная функция"),
        ("rastrigin", "Растригин"),
    ]
    for ax, (prefix, title) in zip(axes, groups):
        for key, item in experiments.items():
            if not key.startswith(prefix):
                continue
            ax.plot(item["history"]["best_values"], label=item["label"], linewidth=1.9)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title(title)
        ax.set_xlabel("итерация")
        ax.set_ylabel("$f(g^t)-f^*$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Как параметры и начальные условия меняют скорость PSO", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "pso_parameter_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_hitting_times(experiments: dict[str, dict], eps: float = 1e-3) -> None:
    labels = []
    values = []
    colors = []
    max_iter = max(item["config"].n_iter for item in experiments.values())
    for key, item in experiments.items():
        summary = item["summary"]
        labels.append(item["short"])
        if summary["hit_eps"] is None:
            values.append(max_iter + 8)
            colors.append("#d95f02")
        else:
            values.append(summary["hit_eps"])
            colors.append("#1b9e77")

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(max_iter, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    ax.set_ylabel(f"$T_{{\\varepsilon}}$, $\\varepsilon={eps:g}$")
    ax.set_title("Время достижения заданной точности")
    ax.grid(axis="y", alpha=0.25)
    for i, item in enumerate(experiments.values()):
        hit = item["summary"]["hit_eps"]
        ax.text(i, values[i] + 1.2, "нет" if hit is None else str(hit), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "pso_hitting_times.png", bbox_inches="tight")
    plt.close(fig)


def save_animation_frames(
    name: str,
    objective,
    history: dict[str, np.ndarray],
    lim: float,
    levels: int,
    every: int = 1,
    gif_duration: float = 0.22,
) -> None:
    frame_dir = ASSETS / f"frames_{name}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()

    xx, yy, zz = surface_grid(objective, lim, n=260)
    positions = history["positions"]
    best_positions = history["best_positions"]
    best_values = history["best_values"]
    frame_paths = []

    for frame_no, iteration in enumerate(range(0, len(positions), every)):
        fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=140)
        contour = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis")
        ax.contour(xx, yy, zz, levels=levels, colors="white", linewidths=0.25, alpha=0.25)
        ax.scatter(
            positions[iteration, :, 0],
            positions[iteration, :, 1],
            s=30,
            c="#f4e04d",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.95,
        )
        ax.scatter(
            best_positions[iteration, 0],
            best_positions[iteration, 1],
            marker="*",
            s=180,
            c="#e63946",
            edgecolors="white",
            linewidths=0.8,
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"PSO: итерация {iteration}, лучшее f={best_values[iteration]:.3g}")
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = frame_dir / f"frame_{frame_no:03d}.png"
        fig.savefig(path, bbox_inches="tight")
        frame_paths.append(path)
        plt.close(fig)

    gif_frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(ASSETS / f"pso_{name}.gif", gif_frames, duration=gif_duration)


def run_experiments() -> dict[str, dict]:
    bounds_quad = np.array([[-4, 4], [-4, 4]], dtype=float)
    bounds_rast = np.array([[-5.12, 5.12], [-5.12, 5.12]], dtype=float)
    quad10 = make_quadratic(10.0)
    quad100 = make_quadratic(100.0)
    rast10 = make_rastrigin(10.0)
    rast20 = make_rastrigin(20.0)

    specs = {
        "quadratic_baseline": {
            "label": "N=35, устойчивые параметры",
            "short": "quad base",
            "function": "квадратичная, k=10",
            "objective": quad10,
            "bounds": bounds_quad,
            "config": PSOConfig(n_particles=35, n_iter=120, omega=0.72, c1=1.49, c2=1.49, seed=2),
        },
        "quadratic_small_swarm": {
            "label": "N=8, мало частиц",
            "short": "quad N=8",
            "function": "квадратичная, k=10",
            "objective": quad10,
            "bounds": bounds_quad,
            "config": PSOConfig(n_particles=8, n_iter=120, omega=0.72, c1=1.49, c2=1.49, seed=2),
        },
        "quadratic_ill_conditioned": {
            "label": "k=100, узкая долина",
            "short": "quad k=100",
            "function": "квадратичная, k=100",
            "objective": quad100,
            "bounds": bounds_quad,
            "config": PSOConfig(n_particles=35, n_iter=120, omega=0.72, c1=1.49, c2=1.49, seed=5),
        },
        "quadratic_slow": {
            "label": "малая инерция и слабое притяжение",
            "short": "quad slow",
            "function": "квадратичная, k=10",
            "objective": quad10,
            "bounds": bounds_quad,
            "config": PSOConfig(n_particles=25, n_iter=120, omega=0.20, c1=0.35, c2=0.35, seed=9),
        },
        "rastrigin_baseline": {
            "label": "N=35, устойчивые параметры",
            "short": "rast base",
            "function": "Растригин, A=10",
            "objective": rast10,
            "bounds": bounds_rast,
            "config": PSOConfig(n_particles=35, n_iter=120, omega=0.72, c1=1.49, c2=1.49, seed=4),
        },
        "rastrigin_small_swarm": {
            "label": "N=8, мало частиц",
            "short": "rast N=8",
            "function": "Растригин, A=10",
            "objective": rast10,
            "bounds": bounds_rast,
            "config": PSOConfig(n_particles=8, n_iter=120, omega=0.72, c1=1.49, c2=1.49, seed=4),
        },
        "rastrigin_local_start": {
            "label": "частицы стартуют около (3,3)",
            "short": "rast local",
            "function": "Растригин, A=10",
            "objective": rast10,
            "bounds": bounds_rast,
            "config": PSOConfig(
                n_particles=12,
                n_iter=120,
                omega=0.25,
                c1=0.55,
                c2=0.55,
                seed=11,
                init_center=(3.0, 3.0),
                init_spread=0.06,
                velocity_scale=0.03,
            ),
        },
        "rastrigin_harder": {
            "label": "A=20, более резкий рельеф",
            "short": "rast A=20",
            "function": "Растригин, A=20",
            "objective": rast20,
            "bounds": bounds_rast,
            "config": PSOConfig(n_particles=35, n_iter=120, omega=0.72, c1=1.49, c2=1.49, seed=12),
        },
    }

    experiments = {}
    for key, spec in specs.items():
        history = pso(spec["objective"], spec["bounds"], spec["config"])
        start = perf_counter()
        for _ in range(TIMED_RUNS):
            pso(spec["objective"], spec["bounds"], spec["config"])
        runtime_ms = (perf_counter() - start) * 1000.0 / TIMED_RUNS
        experiments[key] = {
            **spec,
            "history": history,
            "runtime_ms": runtime_ms,
            "summary": summarize_run(history, eps=1e-3),
        }
    return experiments


def write_summary(experiments: dict[str, dict]) -> None:
    serializable = {}
    for key, item in experiments.items():
        cfg = item["config"]
        serializable[key] = {
            "label": item["label"],
            "function": item["function"],
            "n_particles": cfg.n_particles,
            "n_iter": cfg.n_iter,
            "omega": cfg.omega,
            "c1": cfg.c1,
            "c2": cfg.c2,
            "seed": cfg.seed,
            "init_center": cfg.init_center,
            "init_spread": cfg.init_spread,
            "runtime_ms": item["runtime_ms"],
            "summary": item["summary"],
        }
    (ASSETS / "experiment_summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    animation_stride = 3
    animation_frame_duration = 0.22
    save_pso_scheme()
    save_surfaces()

    experiments = run_experiments()
    save_convergence_plot(
        {
            "квадратичная, базовый запуск": experiments["quadratic_baseline"]["history"],
            "Растригин, базовый запуск": experiments["rastrigin_baseline"]["history"],
        }
    )
    save_parameter_comparison(experiments)
    save_hitting_times(experiments)
    write_summary(experiments)

    save_animation_frames(
        "quadratic",
        make_quadratic(10.0),
        experiments["quadratic_baseline"]["history"],
        lim=4.0,
        levels=32,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )
    save_animation_frames(
        "quadratic_small_swarm",
        make_quadratic(10.0),
        experiments["quadratic_small_swarm"]["history"],
        lim=4.0,
        levels=32,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )
    save_animation_frames(
        "quadratic_ill_conditioned",
        make_quadratic(100.0),
        experiments["quadratic_ill_conditioned"]["history"],
        lim=4.0,
        levels=32,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )
    save_animation_frames(
        "quadratic_slow",
        make_quadratic(10.0),
        experiments["quadratic_slow"]["history"],
        lim=4.0,
        levels=32,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )
    save_animation_frames(
        "rastrigin_harder",
        make_rastrigin(20.0),
        experiments["rastrigin_harder"]["history"],
        lim=5.12,
        levels=55,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )
    save_animation_frames(
        "rastrigin",
        make_rastrigin(10.0),
        experiments["rastrigin_baseline"]["history"],
        lim=5.12,
        levels=55,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )
    save_animation_frames(
        "rastrigin_small_swarm",
        make_rastrigin(10.0),
        experiments["rastrigin_small_swarm"]["history"],
        lim=5.12,
        levels=55,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )
    save_animation_frames(
        "rastrigin_local_start",
        make_rastrigin(10.0),
        experiments["rastrigin_local_start"]["history"],
        lim=5.12,
        levels=55,
        every=animation_stride,
        gif_duration=animation_frame_duration,
    )


if __name__ == "__main__":
    main()

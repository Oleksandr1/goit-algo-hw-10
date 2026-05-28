"""
ДЗ 10, Завдання 2: Обчислення інтеграла методом Монте-Карло.

Функція:    f(x) = x²
Межі:       [0, 2]
Аналітично: ∫₀² x² dx = x³/3 |₀² = 8/3 ≈ 2.6666666667

Реалізовано два варіанти Монте-Карло:
  1) Mean value method  — I ≈ (b-a) · mean(f(X)),  X ~ Uniform(a, b)
  2) Hit-or-miss         — частка точок під кривою × площа прямокутника
"""

from __future__ import annotations

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

A, B = 0.0, 2.0


def f(x):
    return x ** 2


def monte_carlo_mean_value(func, a: float, b: float, n: int, rng=None) -> float:
    """Метод середнього значення. Зазвичай має нижчу дисперсію за hit-or-miss."""
    if rng is None:
        rng = np.random.default_rng(42)
    x = rng.uniform(a, b, size=n)
    return (b - a) * float(np.mean(func(x)))


def monte_carlo_hit_or_miss(
    func, a: float, b: float, n: int, rng=None
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Hit-or-miss: кидаємо n точок у [a,b]×[0, max f], рахуємо частку під кривою.

    Повертає (оцінка інтеграла, x-координати, y-координати, маска "влучили").
    """
    if rng is None:
        rng = np.random.default_rng(42)
    y_max = float(np.max(func(np.linspace(a, b, 1000))))
    x = rng.uniform(a, b, size=n)
    y = rng.uniform(0, y_max, size=n)
    hits = y < func(x)
    area_box = (b - a) * y_max
    return area_box * float(np.mean(hits)), x, y, hits


def _plot_hit_or_miss(n_points: int = 2000, filename: str = "monte_carlo_plot.png") -> None:
    """Малюємо hit-or-miss для невеликого N — щоб видно було точки."""
    estimate, x_pts, y_pts, hits = monte_carlo_hit_or_miss(f, A, B, n_points)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_curve = np.linspace(-0.2, B + 0.2, 400)
    ax.plot(x_curve, f(x_curve), "r", linewidth=2, label="f(x) = x²")

    ix = np.linspace(A, B, 200)
    ax.fill_between(ix, f(ix), color="gray", alpha=0.3, label="Шукана площа")

    ax.scatter(x_pts[hits], y_pts[hits], s=3, c="green", alpha=0.6,
               label=f"Під кривою ({hits.sum()})")
    ax.scatter(x_pts[~hits], y_pts[~hits], s=3, c="red", alpha=0.3,
               label=f"Над кривою ({(~hits).sum()})")

    ax.axvline(A, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(B, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(-0.2, B + 0.2)
    ax.set_ylim(0, max(f(x_curve)) + 0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Монте-Карло (hit-or-miss), N = {n_points}, оцінка ≈ {estimate:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"\nГрафік збережено: {filename}")


def main() -> None:
    analytical = (B ** 3 - A ** 3) / 3
    quad_result, quad_err = spi.quad(f, A, B)

    print(f"Аналітично:     {analytical:.10f}")
    print(f"scipy.quad:     {quad_result:.10f}   (оцінка похибки ≈ {quad_err:.2e})")
    print()

    print(f"{'N':>10} | {'Mean Value':>12} | {'|Δ|':>10} | {'Hit-or-Miss':>12} | {'|Δ|':>10}")
    print("-" * 64)
    for n in (100, 1_000, 10_000, 100_000, 1_000_000):
        # Окремий rng на кожен метод, але фіксований — щоб результат був відтворюваний
        mv = monte_carlo_mean_value(f, A, B, n, rng=np.random.default_rng(42))
        hm, *_ = monte_carlo_hit_or_miss(f, A, B, n, rng=np.random.default_rng(42))
        print(f"{n:>10} | {mv:>12.6f} | {abs(mv - analytical):>10.6f} | "
              f"{hm:>12.6f} | {abs(hm - analytical):>10.6f}")

    _plot_hit_or_miss(n_points=2000)


if __name__ == "__main__":
    main()

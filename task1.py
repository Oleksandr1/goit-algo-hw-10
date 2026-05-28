"""
ДЗ 10, Завдання 1: Розмін монет.

Дві реалізації касової системи видачі решти:
- find_coins_greedy: жадібний алгоритм, O(k) де k = кількість номіналів
- find_min_coins:    динамічне програмування, O(n*k) де n = сума

Порівняння та висновки — у README.md.
"""

from __future__ import annotations

import time
from typing import Iterable

DEFAULT_COINS = (50, 25, 10, 5, 2, 1)


def find_coins_greedy(amount: int, coins: Iterable[int] = DEFAULT_COINS) -> dict[int, int]:
    """Жадібний підхід: на кожному кроці беремо найбільший номінал, що влазить.

    Працює оптимально лише для канонічних систем монет (як [50, 25, 10, 5, 2, 1]
    або більшість реальних валют). Для довільного набору може давати неоптимум.

    Складність: O(k log k) на сортування + O(k) на сам прохід. Від суми НЕ залежить.
    """
    result: dict[int, int] = {}
    for coin in sorted(coins, reverse=True):
        if amount <= 0:
            break
        count, amount = divmod(amount, coin)
        if count:
            result[coin] = count
    return result


def find_min_coins(amount: int, coins: Iterable[int] = DEFAULT_COINS) -> dict[int, int]:
    """DP-підхід: dp[i] = мінімальна кількість монет, щоб скласти суму i.

    Гарантовано знаходить оптимум для БУДЬ-ЯКОГО набору монет.
    Складність: O(n * k) за часом, O(n) за пам'яттю.
    """
    coins = list(coins)
    INF = float("inf")

    # dp[i] — мінімум монет для суми i; last_coin[i] — яку монету взяли останньою
    dp = [0] + [INF] * amount
    last_coin = [0] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                last_coin[i] = coin

    # Реконструюємо словник, "розкручуючи" останні монети
    result: dict[int, int] = {}
    remaining = amount
    while remaining > 0:
        coin = last_coin[remaining]
        result[coin] = result.get(coin, 0) + 1
        remaining -= coin

    # Сортуємо за зростанням номіналу — як у прикладі з умови
    return dict(sorted(result.items()))


def _benchmark() -> None:
    """Порівняння часу виконання на різних сумах."""
    print(f"{'Сума':>10} | {'Жадібний (мс)':>15} | {'DP (мс)':>12} | {'DP / Greedy':>12}")
    print("-" * 60)
    for amount in (113, 1_000, 10_000, 50_000, 100_000):
        # Усереднюємо greedy по багатьох запусках — він занадто швидкий для одного виміру
        t0 = time.perf_counter()
        for _ in range(10_000):
            find_coins_greedy(amount)
        t_greedy = (time.perf_counter() - t0) * 1000 / 10_000

        t0 = time.perf_counter()
        find_min_coins(amount)
        t_dp = (time.perf_counter() - t0) * 1000

        speedup = t_dp / t_greedy if t_greedy > 0 else float("inf")
        print(f"{amount:>10} | {t_greedy:>15.5f} | {t_dp:>12.2f} | {speedup:>11.0f}x")


if __name__ == "__main__":
    # Демонстрація з умови
    print("Приклад з умови (сума = 113):")
    print(f"  Жадібний: {find_coins_greedy(113)}")
    print(f"  DP:       {find_min_coins(113)}")
    print()

    # Класичний контрприклад, де жадібний провалюється
    print("Контрприклад (монети [1, 3, 4], сума = 6):")
    print(f"  Жадібний: {find_coins_greedy(6, [1, 3, 4])}  ← 3 монети, НЕ оптимум")
    print(f"  DP:       {find_min_coins(6, [1, 3, 4])}  ← 2 монети, оптимум")
    print()

    print("Бенчмарк (greedy усереднено по 10 000 запусках):")
    _benchmark()

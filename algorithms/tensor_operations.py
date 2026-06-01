# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"формы тензоров не совпадают: {tt1.shape} != {tt2.shape}")

    if tt1.order != tt2.order:
        raise ValueError("порядки тензоров не совпадают")

    d = tt1.order
    cores = []

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        if k == 0:
            result = backend.zeros((1, n, r1_right + r2_right))

            for i in range(n):
                for a in range(r1_right):
                    result[0, i, a] = core1[0, i, a]
                for b in range(r2_right):
                    result[0, i, r1_right + b] = core2[0, i, b]

        elif k == d - 1:
            result = backend.zeros((r1_left + r2_left, n, 1))

            for a in range(r1_left):
                for i in range(n):
                    result[a, i, 0] = core1[a, i, 0]

            for b in range(r2_left):
                for i in range(n):
                    result[r1_left + b, i, 0] = core2[b, i, 0]

        else:
            result = backend.zeros(
                (r1_left + r2_left, n, r1_right + r2_right)
            )

            for a_left in range(r1_left):
                for i in range(n):
                    for a_right in range(r1_right):
                        result[a_left, i, a_right] = core1[
                            a_left,
                            i,
                            a_right
                        ]

            for b_left in range(r2_left):
                for i in range(n):
                    for b_right in range(r2_right):
                        result[
                            r1_left + b_left,
                            i,
                            r1_right + b_right
                        ] = core2[b_left, i, b_right]

        cores.append(result)
    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    cores = [backend.copy(core) for core in tt.cores]
    cores[0] = backend.scale(cores[0], alpha)
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"формы тензоров не совпадают: {tt1.shape} != {tt2.shape}")
    if tt1.order != tt2.order:
        raise ValueError("порядки тензоров не совпадают")

    cores = []

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        result = backend.zeros(
            (r1_left * r2_left, n, r1_right * r2_right)
        )

        for a_left in range(r1_left):
            for b_left in range(r2_left):
                left_index = a_left * r2_left + b_left

                for i in range(n):
                    for a_right in range(r1_right):
                        for b_right in range(r2_right):
                            right_index = a_right * r2_right + b_right
                            result[left_index, i, right_index] = (
                                core1[a_left, i, a_right]
                                * core2[b_left, i, b_right]
                            )

        cores.append(result)

    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"формы тензоров не совпадают: {tt1.shape} != {tt2.shape}")

    if tt1.order != tt2.order:
        raise ValueError("порядки тензоров не совпадают")

    Z = backend.ones((1, 1))

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        new_Z = backend.zeros((r1_right, r2_right))

        for a_right in range(r1_right):
            for b_right in range(r2_right):
                value = 0.0

                for a_left in range(r1_left):
                    for b_left in range(r2_left):
                        z_value = Z[a_left, b_left]

                        for i in range(n):
                            value += (
                                z_value
                                * core1[a_left, i, a_right]
                                * core2[b_left, i, b_right]
                            )

                new_Z[a_right, b_right] = value

        Z = new_Z

    return Z[0, 0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    value = tt_dot(tt, tt, backend)
    return math.sqrt(max(value, 0.0))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"формы тензоров не совпадают: {tt1.shape} != {tt2.shape}")

    value = (
        tt_dot(tt1, tt1, backend)
        - 2.0 * tt_dot(tt1, tt2, backend)
        + tt_dot(tt2, tt2, backend)
    )

    return math.sqrt(max(value, 0.0))
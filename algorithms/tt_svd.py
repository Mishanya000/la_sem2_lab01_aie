# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    shape = tensor.shape
    d = tensor.ndim

    if d == 1:
        return TTTensor([backend.reshape(tensor, (1, shape[0], 1))])

    norm = backend.norm(tensor)
    if norm > 1e-30:
        delta = eps * norm / math.sqrt(d - 1)
    else:
        delta = 0.0

    cores = []
    C = backend.copy(tensor)
    r_prev = 1

    for k in range(d - 1):
        n_k = shape[k]

        left_size = r_prev * n_k
        right_size = C.size // left_size

        matrix = backend.reshape(C, (left_size, right_size))
        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_truncated_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        core = backend.reshape(U_trunc, (r_prev, n_k, rank))
        cores.append(core)

        C = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)
        r_prev = rank

    last_core = backend.reshape(C, (r_prev, shape[-1], 1))
    cores.append(last_core)
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("ожидает 1D вектор")

    k = S.shape[0]
    if k == 0:
        return 1

    threshold = max(1e-12, 1e-8 * abs(S[0]))

    numerical_rank = 0
    for value in S.data:
        if abs(value) > threshold:
            numerical_rank += 1

    rank = max(1, numerical_rank)

    if delta > 0.0:
        tail_sq = 0.0
        while rank > 1:
            candidate_value = S[rank - 1]
            if tail_sq + candidate_value * candidate_value <= delta * delta:
                tail_sq += candidate_value * candidate_value
                rank -= 1
            else:
                break

    if max_rank is not None:
        rank = min(rank, max_rank)

    return max(1, rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("ожидает 2D матрицу")

    m, n = matrix.shape
    if rank < 0 or rank > n:
        raise ValueError(f"некорректный rank={rank} для матрицы {matrix.shape}")

    result = backend.zeros((m, rank))
    for i in range(m):
        for j in range(rank):
            result[i, j] = matrix[i, j]

    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("ожидает 2D матрицу")

    m, n = matrix.shape
    if rank < 0 or rank > m:
        raise ValueError(f"некорректный rank={rank} для матрицы {matrix.shape}")

    result = backend.zeros((rank, n))
    for i in range(rank):
        for j in range(n):
            result[i, j] = matrix[i, j]

    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError("ожидает 1D вектор")

    if rank < 0 or rank > vector.shape[0]:
        raise ValueError(f"некорректный rank={rank} для вектора {vector.shape}")

    result = backend.zeros((rank,))
    for i in range(rank):
        result[i] = vector[i]

    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1 or matrix.ndim != 2:
        raise ValueError("ожидает 1D и 2D тензоры")

    rows, cols = matrix.shape
    if rank > diag_vec.shape[0] or rank > rows:
        raise ValueError("rank несовместим с размерами аргументов")

    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[i, j] = diag_vec[i] * matrix[i, j]

    return result
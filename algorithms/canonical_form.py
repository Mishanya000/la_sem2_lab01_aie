# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [backend.copy(core) for core in tt.cores]

    for k in range(tt.order - 1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = backend.reshape(core, (r_left * n, r_right))
        Q, R = backend.qr(matrix)

        new_rank = Q.shape[1]
        cores[k] = backend.reshape(Q, (r_left, n, new_rank))

        next_core = cores[k + 1]
        _, next_n, next_r = next_core.shape
        next_matrix = backend.reshape(next_core, (r_right, next_n * next_r))

        updated_next = backend.matmul(R, next_matrix)
        cores[k + 1] = backend.reshape(
            updated_next,
            (new_rank, next_n, next_r)
        )

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [backend.copy(core) for core in tt.cores]

    for k in range(tt.order - 1, 0, -1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = backend.reshape(core, (r_left, n * r_right))

        Q_t, R_t = backend.qr(backend.transpose(matrix))

        Q = backend.transpose(Q_t)
        R = backend.transpose(R_t)

        new_rank = Q.shape[0]
        cores[k] = backend.reshape(Q, (new_rank, n, r_right))

        prev_core = cores[k - 1]
        prev_r_left, prev_n, _ = prev_core.shape
        prev_matrix = backend.reshape(
            prev_core,
            (prev_r_left * prev_n, r_left)
        )

        updated_prev = backend.matmul(prev_matrix, R)
        cores[k - 1] = backend.reshape(
            updated_prev,
            (prev_r_left, prev_n, new_rank)
        )

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.ndim != 1:
        raise ValueError(f"ожидает 1D, получено ndim={S.ndim}")

    if S.size == 0:
        return 0

    max_s = max(abs(x) for x in S.data)
    threshold = max(abs_tol, rel_tol * max_s)

    rank = 0
    for value in S.data:
        if abs(value) > threshold:
            rank += 1

    return rank


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
        rank:     длина диагонального вектора
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


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    if matrix.ndim != 2 or diag_vec.ndim != 1:
        raise ValueError("ожидает 2D и 1D тензоры")

    m, n = matrix.shape
    if diag_vec.shape[0] != n:
        raise ValueError("длина diag_vec должна совпадать с числом столбцов")

    result = backend.zeros((m, n))
    for i in range(m):
        for j in range(n):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result
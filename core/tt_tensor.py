# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not isinstance(cores, list):
            raise TypeError("cores должен быть списком DenseTensor")

        if len(cores) == 0:
            raise ValueError("список ядер не может быть пустым")

        for core in cores:
            if not isinstance(core, DenseTensor):
                raise TypeError("каждое ядро должно быть DenseTensor")
            if core.ndim != 3:
                raise ValueError(
                    f"каждое ядро должно быть 3D, получено ndim={core.ndim}"
                )

        if cores[0].shape[0] != 1:
            raise ValueError("первый ранг должен быть равен 1")

        if cores[-1].shape[2] != 1:
            raise ValueError("последний ранг должен быть равен 1")

        for k in range(len(cores) - 1):
            if cores[k].shape[2] != cores[k + 1].shape[0]:
                raise ValueError(
                    "соседние ядра имеют несовместимые ранги: "
                    f"{cores[k].shape[2]} != {cores[k + 1].shape[0]}"
                )

        self.cores = [core.copy() for core in cores]
        self.order = len(cores)
        self.shape = tuple(core.shape[1] for core in cores)
        self.ranks = tuple([cores[0].shape[0]] + [core.shape[2] for core in cores])

        validate_shape(self.shape)

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        shape = validate_shape(shape)
        d = len(shape)

        ranks = tuple(ranks)

        if len(ranks) == d - 1:
            full_ranks = (1,) + ranks + (1,)
        elif len(ranks) == d + 1:
            full_ranks = ranks
        else:
            raise ValueError(
                "ranks должен быть либо списком внутренних рангов длины d-1, "
                "либо полным списком рангов длины d+1"
            )

        if full_ranks[0] != 1 or full_ranks[-1] != 1:
            raise ValueError("Граничные ранги должны быть 1")

        for rank in full_ranks:
            if not isinstance(rank, int) or rank <= 0:
                raise ValueError("Все ранги должны быть положительными")
        cores = []
        for k in range(d):
            core_shape = (full_ranks[k], shape[k], full_ranks[k + 1])
            cores.append(DenseTensor.random(core_shape, seed=None if seed is None else seed + k))
        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if len(indices) != self.order:
            raise ValueError(
                f"Ожидалось {self.order} индексов, получено {len(indices)}"
            )

        indices = tuple(indices)

        for index, dim in zip(indices, self.shape):
            if index < 0 or index >= dim:
                raise IndexError(f"Индекс {index} выходит за границы размера {dim}")

        current = [1.0]

        for k, index in enumerate(indices):
            core = self.cores[k]
            r_left, _, r_right = core.shape

            next_current = [0.0] * r_right

            for right_rank in range(r_right):
                value = 0.0
                for left_rank in range(r_left):
                    value += (
                        current[left_rank]
                        * core[left_rank, index, right_rank]
                    )
                next_current[right_rank] = value

            current = next_current

        return current[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = DenseTensor.zeros(self.shape)

        for flat_index in range(result.size):
            multi_index = flat_to_multi_index(flat_index, self.shape)
            result.data[flat_index] = self.get_element(multi_index)

        return result

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        storage = self.total_storage()
        if storage == 0:
            raise ZeroDivisionError("тензор имеет нулевой объём хранения")

        return compute_size(self.shape) / storage

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        return (
            "TTTensor(\n"
            f"  order={self.order},\n"
            f"  shape={self.shape},\n"
            f"  ranks={self.ranks},\n"
            f"  cores={self.core_sizes()},\n"
            f"  total_storage={self.total_storage()}\n"
            ")"
        )

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()
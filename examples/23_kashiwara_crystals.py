# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Decompose the tensor square of the type-A2 letter crystal."""

from noether.combin import KashiwaraCrystal


def main() -> None:
    letters = KashiwaraCrystal.type_a_letters(2)
    tensor_square = letters.tensor(letters)
    highest_weights = sorted(
        tensor_square.weight(vertex) for vertex in tensor_square.highest_weight_vertices()
    )
    assert tensor_square.vertex_count == 9
    assert highest_weights == [(0, 1), (2, 0)]
    print(f"B(omega_1)^2 has highest weights {highest_weights}")


if __name__ == "__main__":
    main()

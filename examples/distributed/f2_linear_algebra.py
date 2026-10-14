# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Run exact packed F2 linear algebra under ``mpirun`` or ``srun``."""

from noether import capabilities
from noether.linalg import DistributedMatrix, Matrix, World
from noether.numthy import FiniteField

assert "distributed" in capabilities().compiled_features
assert capabilities().mpi_runtime is None
world = World.initialize()
runtime = capabilities().mpi_runtime
assert runtime is not None
assert runtime.world_size == world.size
assert runtime.mpi_rank == world.mpi_rank
field = FiniteField.prime(2)

# Every process executes the same collective sequence. Only rank zero owns the
# bounded local input; scatter and collect retain packed m4rst storage.
local = Matrix.identity(field, 7) if world.is_root else None
matrix = DistributedMatrix.distribute(world, root=0, matrix=local, max_bytes=56)
assert int(matrix.get(0, 0)) == 1
assert matrix.extract_bits(0, 0, 7) == 1
assert matrix.set(0, 1, field(1)).flip(0, 1).equals(matrix)
row_ops = matrix.xor_row(1, 0).swap_rows(1, 2).swap_cols(0, 3)
assert int(row_ops.get(2, 1)) == 1
assert int(row_ops.get(2, 3)) == 1
zero = DistributedMatrix.zeros(world, field, 7, 7)
assert (matrix + matrix).equals(zero)
assert (matrix - matrix).equals(zero)
assert matrix.transpose().equals(matrix)
corner = matrix.submatrix(0, 0, 3, 3)
assert corner.equals(DistributedMatrix.identity(world, field, 3))
assert zero.paste(0, 0, corner).rank() == 3
assert (matrix @ matrix).equals(matrix)
solution = matrix.solve_left(matrix)
inverse = matrix.inverse()
assert solution is not None and solution.equals(matrix)
assert inverse is not None and inverse.equals(matrix)
collected = matrix.collect(root=0, max_bytes=56)
if world.is_root:
    assert collected == local
else:
    assert collected is None

random = DistributedMatrix.random(world, field, 7, 7, seed=0x2026)
factorization = random.pluq()
assert factorization.verify(random)
echelon, echelon_rank = random.echelon()
reduced, reduced_rank = random.rref()
assert echelon.shape == random.shape
assert reduced.shape == random.shape
assert echelon_rank == reduced_rank
rank = random.rank()
assert rank == random.ple_rank()
assert random.nullspace().rows == random.cols

if world.is_root:
    print(
        f"Noether distributed F2: processes={world.size}, rank={rank}, MPI={world.library_version}"
    )

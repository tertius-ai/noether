/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Benjamin Antieau and Deven Manam
 */
/*
 * libm4rie oracle shim — for the `validate-libm4rie` benchmark /
 * parity path ONLY.
 *
 * libm4rie is GPL-2; this file is the single point that links it, is
 * compiled only when the `validate-libm4rie` Cargo feature is on, and
 * NEVER ends up in a shipped m4rste build. It exists so the
 * `bench_vs_m4rie` example can time and bit-check m4rste's pure-Rust
 * GF(2^e) kernels against the reference implementation.
 *
 * m4rie's element accessors (mzed_write_elem / mzed_read_elem) are
 * static-inline in the headers, so they cannot be called over a raw
 * Rust FFI boundary — hence this thin C shim, which exposes flat
 * extern-C entry points taking only C primitive types + u16 buffers.
 *
 * Elements are integers in [0, 2^e); `minpoly` is the GF(2^e)
 * defining polynomial as a bit pattern (e.g. GF(256) with
 * x^8+x^4+x^3+x+1 is 0x11B). I/O buffers are row-major u16 arrays.
 */

#include <m4rie/m4rie.h> /* transitively pulls in <m4ri/mzp.h> for mzp_t */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Opaque, preloaded square-matrix fixture for honest kernel timings. The
 * buffer entry points below deliberately remain self-contained parity APIs;
 * benchmark callers use this fixture so field/input marshalling is outside
 * the timed operation.
 */
typedef struct {
  gf2e *ff;
  mzed_t *a;
  mzed_t *b;
} m4rie_oracle_pair;

void *m4rie_oracle_pair_new(uint64_t minpoly, size_t n,
                            const uint16_t *a_data,
                            const uint16_t *b_data) {
  m4rie_oracle_pair *pair = calloc(1, sizeof(*pair));
  if (!pair)
    return NULL;
  pair->ff = gf2e_init(minpoly);
  if (!pair->ff)
    goto fail;
  pair->a = mzed_init(pair->ff, (rci_t)n, (rci_t)n);
  pair->b = mzed_init(pair->ff, (rci_t)n, (rci_t)n);
  if (!pair->a || !pair->b)
    goto fail;
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      mzed_write_elem(pair->a, (rci_t)i, (rci_t)j,
                      (word)a_data[i * n + j]);
      mzed_write_elem(pair->b, (rci_t)i, (rci_t)j,
                      (word)b_data[i * n + j]);
    }
  return pair;

fail:
  if (pair->a)
    mzed_free(pair->a);
  if (pair->b)
    mzed_free(pair->b);
  if (pair->ff)
    gf2e_free(pair->ff);
  free(pair);
  return NULL;
}

void m4rie_oracle_pair_free(void *opaque) {
  m4rie_oracle_pair *pair = opaque;
  if (!pair)
    return;
  mzed_free(pair->a);
  mzed_free(pair->b);
  gf2e_free(pair->ff);
  free(pair);
}

int m4rie_oracle_preloaded_matmul(void *opaque, uint16_t *checksum) {
  m4rie_oracle_pair *pair = opaque;
  if (!pair || !checksum)
    return -1;
  mzed_t *output = mzed_mul(NULL, pair->a, pair->b);
  if (!output)
    return -1;
  *checksum = output->nrows && output->ncols
                  ? (uint16_t)mzed_read_elem(output, 0, 0)
                  : 0;
  mzed_free(output);
  return 0;
}

int m4rie_oracle_preloaded_add(void *opaque, uint16_t *checksum) {
  m4rie_oracle_pair *pair = opaque;
  if (!pair || !checksum)
    return -1;
  mzed_t *output = mzed_add(NULL, pair->a, pair->b);
  if (!output)
    return -1;
  *checksum = output->nrows && output->ncols
                  ? (uint16_t)mzed_read_elem(output, 0, 0)
                  : 0;
  mzed_free(output);
  return 0;
}

size_t m4rie_oracle_preloaded_rank(void *opaque) {
  m4rie_oracle_pair *pair = opaque;
  if (!pair)
    return (size_t)-1;
  mzed_t *work = mzed_copy(NULL, pair->a);
  if (!work)
    return (size_t)-1;
  rci_t rank = mzed_echelonize(work, 0);
  mzed_free(work);
  return (size_t)rank;
}

size_t m4rie_oracle_preloaded_rref(void *opaque, uint16_t *checksum) {
  m4rie_oracle_pair *pair = opaque;
  if (!pair || !checksum)
    return (size_t)-1;
  mzed_t *work = mzed_copy(NULL, pair->a);
  if (!work)
    return (size_t)-1;
  rci_t rank = mzed_echelonize(work, 1);
  *checksum = work->nrows && work->ncols
                  ? (uint16_t)mzed_read_elem(work, 0, 0)
                  : 0;
  mzed_free(work);
  return (size_t)rank;
}

/* C = A * B over GF(2^e). Returns 0 on success, -1 on error. */
int m4rie_oracle_matmul_u16(uint64_t minpoly, size_t nrows_a, size_t ncols_a,
                            const uint16_t *a_data, size_t nrows_b,
                            size_t ncols_b, const uint16_t *b_data,
                            uint16_t *out_data) {
  if (ncols_a != nrows_b)
    return -1;

  gf2e *ff = gf2e_init(minpoly);
  if (!ff)
    return -1;

  mzed_t *A = mzed_init(ff, (rci_t)nrows_a, (rci_t)ncols_a);
  mzed_t *B = mzed_init(ff, (rci_t)nrows_b, (rci_t)ncols_b);
  if (!A || !B) {
    if (A)
      mzed_free(A);
    if (B)
      mzed_free(B);
    gf2e_free(ff);
    return -1;
  }

  for (size_t i = 0; i < nrows_a; ++i)
    for (size_t j = 0; j < ncols_a; ++j)
      mzed_write_elem(A, (rci_t)i, (rci_t)j, (word)a_data[i * ncols_a + j]);

  for (size_t i = 0; i < nrows_b; ++i)
    for (size_t j = 0; j < ncols_b; ++j)
      mzed_write_elem(B, (rci_t)i, (rci_t)j, (word)b_data[i * ncols_b + j]);

  mzed_t *C = mzed_mul(NULL, A, B);
  mzed_free(A);
  mzed_free(B);
  if (!C) {
    gf2e_free(ff);
    return -1;
  }

  for (size_t i = 0; i < nrows_a; ++i)
    for (size_t j = 0; j < ncols_b; ++j)
      out_data[i * ncols_b + j] = (uint16_t)mzed_read_elem(C, (rci_t)i, (rci_t)j);

  mzed_free(C);
  gf2e_free(ff);
  return 0;
}

/* C = A + B over GF(2^e) (= XOR in characteristic 2). */
int m4rie_oracle_add_u16(uint64_t minpoly, size_t nrows, size_t ncols,
                         const uint16_t *a_data, const uint16_t *b_data,
                         uint16_t *out_data) {
  gf2e *ff = gf2e_init(minpoly);
  if (!ff)
    return -1;

  mzed_t *A = mzed_init(ff, (rci_t)nrows, (rci_t)ncols);
  mzed_t *B = mzed_init(ff, (rci_t)nrows, (rci_t)ncols);
  if (!A || !B) {
    if (A)
      mzed_free(A);
    if (B)
      mzed_free(B);
    gf2e_free(ff);
    return -1;
  }

  for (size_t i = 0; i < nrows; ++i)
    for (size_t j = 0; j < ncols; ++j) {
      mzed_write_elem(A, (rci_t)i, (rci_t)j, (word)a_data[i * ncols + j]);
      mzed_write_elem(B, (rci_t)i, (rci_t)j, (word)b_data[i * ncols + j]);
    }

  mzed_add(A, A, B);
  mzed_free(B);

  for (size_t i = 0; i < nrows; ++i)
    for (size_t j = 0; j < ncols; ++j)
      out_data[i * ncols + j] = (uint16_t)mzed_read_elem(A, (rci_t)i, (rci_t)j);

  mzed_free(A);
  gf2e_free(ff);
  return 0;
}

/* Rank over GF(2^e). Returns the rank, or (size_t)-1 on error. */
size_t m4rie_oracle_rank_u16(uint64_t minpoly, size_t nrows, size_t ncols,
                             const uint16_t *data) {
  gf2e *ff = gf2e_init(minpoly);
  if (!ff)
    return (size_t)-1;

  mzed_t *A = mzed_init(ff, (rci_t)nrows, (rci_t)ncols);
  if (!A) {
    gf2e_free(ff);
    return (size_t)-1;
  }

  for (size_t i = 0; i < nrows; ++i)
    for (size_t j = 0; j < ncols; ++j)
      mzed_write_elem(A, (rci_t)i, (rci_t)j, (word)data[i * ncols + j]);

  rci_t rank = mzed_echelonize(A, 0); /* full=0 for speed */
  mzed_free(A);
  gf2e_free(ff);
  return (size_t)rank;
}

/* RREF over GF(2^e). Returns the rank; writes the RREF into out_data. */
size_t m4rie_oracle_rref_u16(uint64_t minpoly, size_t nrows, size_t ncols,
                             const uint16_t *data, uint16_t *out_data) {
  gf2e *ff = gf2e_init(minpoly);
  if (!ff)
    return (size_t)-1;

  mzed_t *A = mzed_init(ff, (rci_t)nrows, (rci_t)ncols);
  if (!A) {
    gf2e_free(ff);
    return (size_t)-1;
  }

  for (size_t i = 0; i < nrows; ++i)
    for (size_t j = 0; j < ncols; ++j)
      mzed_write_elem(A, (rci_t)i, (rci_t)j, (word)data[i * ncols + j]);

  rci_t rank = mzed_echelonize(A, 1); /* full=1 for RREF */

  for (size_t i = 0; i < nrows; ++i)
    for (size_t j = 0; j < ncols; ++j)
      out_data[i * ncols + j] = (uint16_t)mzed_read_elem(A, (rci_t)i, (rci_t)j);

  mzed_free(A);
  gf2e_free(ff);
  return (size_t)rank;
}

/*
 * PLE decomposition rank over GF(2^e), via mzed_ple. Returns the rank,
 * or (size_t)-1 on error.
 *
 * Rank-only, matching m4rst's own m4ri_oracle_pluq_rank precedent
 * (crates/m4rst/oracle/m4ri_oracle.c): a PLUQ/PLE factorization isn't
 * unique across implementations (pivot search order, tie-breaking),
 * so there is no meaningful bit-equality target for P/L/U/Q
 * themselves — only the rank is a well-defined, checkable invariant.
 * m4rste's own `Mzed::pluq` is structurally self-checked instead
 * (P/Q are permutation matrices, L/U are triangular, P·L·U·Q == A —
 * see tests/pluq.rs).
 */
size_t m4rie_oracle_ple_rank_u16(uint64_t minpoly, size_t nrows, size_t ncols,
                                 const uint16_t *data) {
  gf2e *ff = gf2e_init(minpoly);
  if (!ff)
    return (size_t)-1;

  mzed_t *A = mzed_init(ff, (rci_t)nrows, (rci_t)ncols);
  if (!A) {
    gf2e_free(ff);
    return (size_t)-1;
  }

  for (size_t i = 0; i < nrows; ++i)
    for (size_t j = 0; j < ncols; ++j)
      mzed_write_elem(A, (rci_t)i, (rci_t)j, (word)data[i * ncols + j]);

  mzp_t *P = mzp_init((rci_t)nrows);
  mzp_t *Q = mzp_init((rci_t)ncols);
  if (!P || !Q) {
    if (P)
      mzp_free(P);
    if (Q)
      mzp_free(Q);
    mzed_free(A);
    gf2e_free(ff);
    return (size_t)-1;
  }

  rci_t rank = mzed_ple(A, P, Q);

  mzp_free(P);
  mzp_free(Q);
  mzed_free(A);
  gf2e_free(ff);
  return (size_t)rank;
}

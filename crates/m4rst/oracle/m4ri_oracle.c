/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Benjamin Antieau and Deven Manam
 */
/*
 * libm4ri oracle shim — for the `validate-libm4ri` parity path ONLY.
 *
 * libm4ri is GPL-2; this file is the single point that links it, is
 * compiled only when the `validate-libm4ri` Cargo feature is on, and
 * NEVER ends up in a shipped m4rst build. It lets
 * `tests/parity_libm4ri.rs` bit-check m4rst's pure-Rust F_2 kernels
 * against the reference implementation.
 *
 * Why a C shim rather than a Rust-side `#[repr(C)]` mirror of `mzd_t`:
 * m4ri's row + bit accessors (`mzd_row`, `mzd_read_bit`, …) are
 * `static inline` in the headers, so they have no exported symbol a
 * raw Rust FFI boundary could call. The previous parity test instead
 * hand-mirrored the `mzd_t` struct in Rust and re-implemented
 * `mzd_row` against it — which silently broke when libm4ri's layout
 * drifted (the `size_of == 64` assert passed for both the old and new
 * layouts). This file `#include`s the real header, so the layout is
 * always exactly whatever the linked libm4ri uses, and `mzd_row` is
 * the real inline. Flat extern-C entry points take only C primitives
 * and packed u64 buffers.
 *
 * Buffer convention: a matrix is row-major, each row `ceil(ncols/64)`
 * u64 words, bit (i, j) = (data[i*width + j/64] >> (j%64)) & 1. This
 * is m4rst's own `Mzd::as_words()` layout and the universal F_2
 * bit-packing convention; `m4ri_oracle_roundtrip` + the round-trip
 * parity test pin it.
 */

#include <m4ri/m4ri.h>
#include <stddef.h>
#include <stdint.h>

static size_t row_words(size_t ncols) { return (ncols + 63) / 64; }
static void load(mzd_t *M, const uint64_t *data, size_t nrows, size_t ncols);

/* Opaque, preloaded matrices for performance comparisons.  Benchmark code
 * must never mirror m4ri's private mzd_t layout in Rust: that layout has
 * changed while retaining the same sizeof(mzd_t). */
void *m4ri_oracle_matrix_new(size_t nrows, size_t ncols,
                             const uint64_t *data) {
  mzd_t *M = mzd_init((rci_t)nrows, (rci_t)ncols);
  if (!M)
    return NULL;
  load(M, data, nrows, ncols);
  return (void *)M;
}

void m4ri_oracle_matrix_free(void *matrix) {
  if (matrix)
    mzd_free((mzd_t *)matrix);
}

/* Multiply two preloaded matrices and dispose of the result.  `algorithm`
 * selects mzd_mul_m4rm (0) or mzd_mul (1).  The checksum keeps the result
 * observable across the FFI boundary without charging the timed operation
 * for a full packed-buffer copy. */
int m4ri_oracle_preloaded_matmul(const void *left, const void *right,
                                 int algorithm, uint64_t *checksum) {
  const mzd_t *A = (const mzd_t *)left;
  const mzd_t *B = (const mzd_t *)right;
  if (!A || !B || !checksum || A->ncols != B->nrows)
    return -1;
  mzd_t *C = algorithm == 0 ? mzd_mul_m4rm(NULL, A, B, 0)
                            : mzd_mul(NULL, A, B, 0);
  if (!C)
    return -1;
  *checksum = (C->nrows == 0 || C->ncols == 0) ? 0 : (uint64_t)mzd_row(C, 0)[0];
  mzd_free(C);
  return 0;
}

/* Copy and row-reduce one preloaded matrix. This mirrors m4rst's immutable
 * value interface: fixture conversion is outside the timed region, while the
 * result copy/allocation remains part of the operation. */
int m4ri_oracle_preloaded_echelonize(const void *matrix, int full,
                                     uint64_t *checksum, size_t *rank) {
  const mzd_t *A = (const mzd_t *)matrix;
  if (!A || !checksum || !rank)
    return -1;
  mzd_t *R = mzd_copy(NULL, A);
  if (!R)
    return -1;
  *rank = (size_t)mzd_echelonize(R, full);
  *checksum = (R->nrows == 0 || R->ncols == 0) ? 0 : (uint64_t)mzd_row(R, 0)[0];
  mzd_free(R);
  return 0;
}

/* Copy a packed u64 buffer into an mzd_t, row by row via the real
 * (header-inline) `mzd_row`. */
static void load(mzd_t *M, const uint64_t *data, size_t nrows, size_t ncols) {
  size_t w = row_words(ncols);
  for (size_t i = 0; i < nrows; ++i) {
    word *row = mzd_row(M, (rci_t)i);
    for (size_t k = 0; k < w; ++k)
      row[k] = (word)data[i * w + k];
  }
}

/* Read an mzd_t back into a packed u64 buffer. */
static void store(uint64_t *out, const mzd_t *M, size_t nrows, size_t ncols) {
  size_t w = row_words(ncols);
  for (size_t i = 0; i < nrows; ++i) {
    word *row = mzd_row(M, (rci_t)i);
    for (size_t k = 0; k < w; ++k)
      out[i * w + k] = (uint64_t)row[k];
  }
}

/* mzd_set_ui(M, value): value 0 -> zero matrix, value 1 -> identity. */
int m4ri_oracle_set_ui(size_t nrows, size_t ncols, uint32_t value,
                       uint64_t *out) {
  mzd_t *M = mzd_init((rci_t)nrows, (rci_t)ncols);
  if (!M)
    return -1;
  mzd_set_ui(M, (unsigned int)value);
  store(out, M, nrows, ncols);
  mzd_free(M);
  return 0;
}

/* Load a packed buffer into libm4ri and read it straight back —
 * pins the buffer convention against libm4ri's mzd_t layout. */
int m4ri_oracle_roundtrip(size_t nrows, size_t ncols, const uint64_t *data,
                          uint64_t *out) {
  mzd_t *M = mzd_init((rci_t)nrows, (rci_t)ncols);
  if (!M)
    return -1;
  load(M, data, nrows, ncols);
  store(out, M, nrows, ncols);
  mzd_free(M);
  return 0;
}

/* OUT = A * B over F_2. use_strassen: 0 -> mzd_mul_naive (cubic ground
 * truth), 1 -> mzd_mul (Strassen-on-M4RM production dispatcher,
 * cutoff 0 = libm4ri's default). Returns 0 on success, -1 on error. */
int m4ri_oracle_matmul(size_t nra, size_t nca, const uint64_t *a, size_t nrb,
                       size_t ncb, const uint64_t *b, uint64_t *out,
                       int use_strassen) {
  if (nca != nrb)
    return -1;
  mzd_t *A = mzd_init((rci_t)nra, (rci_t)nca);
  mzd_t *B = mzd_init((rci_t)nrb, (rci_t)ncb);
  if (!A || !B) {
    if (A)
      mzd_free(A);
    if (B)
      mzd_free(B);
    return -1;
  }
  load(A, a, nra, nca);
  load(B, b, nrb, ncb);
  mzd_t *C = use_strassen ? mzd_mul(NULL, A, B, 0) : mzd_mul_naive(NULL, A, B);
  mzd_free(A);
  mzd_free(B);
  if (!C)
    return -1;
  store(out, C, nra, ncb);
  mzd_free(C);
  return 0;
}

/* OUT = A^T. OUT is ncols x nrows. */
int m4ri_oracle_transpose(size_t nrows, size_t ncols, const uint64_t *data,
                          uint64_t *out) {
  mzd_t *A = mzd_init((rci_t)nrows, (rci_t)ncols);
  if (!A)
    return -1;
  load(A, data, nrows, ncols);
  mzd_t *T = mzd_transpose(NULL, A);
  mzd_free(A);
  if (!T)
    return -1;
  store(out, T, ncols, nrows);
  mzd_free(T);
  return 0;
}

/* OUT = A + B over F_2 (= elementwise XOR). */
int m4ri_oracle_add(size_t nrows, size_t ncols, const uint64_t *a,
                    const uint64_t *b, uint64_t *out) {
  mzd_t *A = mzd_init((rci_t)nrows, (rci_t)ncols);
  mzd_t *B = mzd_init((rci_t)nrows, (rci_t)ncols);
  if (!A || !B) {
    if (A)
      mzd_free(A);
    if (B)
      mzd_free(B);
    return -1;
  }
  load(A, a, nrows, ncols);
  load(B, b, nrows, ncols);
  mzd_t *C = mzd_add(NULL, A, B);
  mzd_free(A);
  mzd_free(B);
  if (!C)
    return -1;
  store(out, C, nrows, ncols);
  mzd_free(C);
  return 0;
}

/* In-place Gauss-Jordan. full: 1 -> reduced row echelon form,
 * 0 -> row echelon form. Returns the rank (>= 0), or -1 on error;
 * writes the echelon form into out. */
int m4ri_oracle_echelonize(size_t nrows, size_t ncols, const uint64_t *data,
                           int full, uint64_t *out) {
  mzd_t *A = mzd_init((rci_t)nrows, (rci_t)ncols);
  if (!A)
    return -1;
  load(A, data, nrows, ncols);
  rci_t rank = mzd_echelonize(A, full);
  store(out, A, nrows, ncols);
  mzd_free(A);
  return (int)rank;
}

/* Rank via libm4ri's production PLUQ decomposition. Returns the rank
 * (>= 0), or -1 on error. */
int m4ri_oracle_pluq_rank(size_t nrows, size_t ncols, const uint64_t *data) {
  mzd_t *A = mzd_init((rci_t)nrows, (rci_t)ncols);
  if (!A)
    return -1;
  mzp_t *P = mzp_init((rci_t)nrows);
  mzp_t *Q = mzp_init((rci_t)ncols);
  if (!P || !Q) {
    if (P)
      mzp_free(P);
    if (Q)
      mzp_free(Q);
    mzd_free(A);
    return -1;
  }
  load(A, data, nrows, ncols);
  rci_t rank = mzd_pluq(A, P, Q, 0);
  mzp_free(P);
  mzp_free(Q);
  mzd_free(A);
  return (int)rank;
}

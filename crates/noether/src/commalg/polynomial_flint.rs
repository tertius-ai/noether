// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Private persistent FLINT storage and fallback dispatch for polynomials.

#![allow(unsafe_code)]

use core::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::sync::Arc;

use flint_sys::{
    fmpq_poly_clear, fmpq_poly_get_coeff_fmpq, fmpq_poly_init, fmpq_poly_length, fmpq_poly_mullow,
    fmpq_poly_set_coeff_fmpq, fmpz_mod_ctx_clear, fmpz_mod_ctx_init, fmpz_mod_ctx_struct,
    fmpz_mod_poly_clear, fmpz_mod_poly_factor, fmpz_mod_poly_factor_clear,
    fmpz_mod_poly_factor_get_poly, fmpz_mod_poly_factor_init, fmpz_mod_poly_factor_struct,
    fmpz_mod_poly_get_coeff_fmpz, fmpz_mod_poly_init, fmpz_mod_poly_length, fmpz_mod_poly_mullow,
    fmpz_mod_poly_set_coeff_fmpz, fmpz_poly_clear, fmpz_poly_factor, fmpz_poly_factor_clear,
    fmpz_poly_factor_get_fmpz_poly, fmpz_poly_factor_init, fmpz_poly_factor_struct,
    fmpz_poly_get_coeff_fmpz, fmpz_poly_init, fmpz_poly_length, fmpz_poly_mullow,
    fmpz_poly_set_coeff_fmpz, fq_default_poly_clear, fq_default_poly_factor,
    fq_default_poly_factor_clear, fq_default_poly_factor_exp, fq_default_poly_factor_get_poly,
    fq_default_poly_factor_init, fq_default_poly_factor_length, fq_default_poly_factor_struct,
    fq_default_poly_get_coeff, fq_default_poly_init, fq_default_poly_length,
    fq_default_poly_mullow, fq_default_poly_set_coeff, nmod_poly_clear, nmod_poly_factor,
    nmod_poly_factor_clear, nmod_poly_factor_get_poly, nmod_poly_factor_init,
    nmod_poly_factor_struct, nmod_poly_get_coeff_ui, nmod_poly_init, nmod_poly_length,
    nmod_poly_mullow, nmod_poly_set_coeff_ui, ordering_t, ordering_t_ORD_DEGLEX,
    ordering_t_ORD_DEGREVLEX, ordering_t_ORD_LEX, slong,
};

#[cfg(test)]
use flint_sys::{
    fmpq_mpoly_clear, fmpq_mpoly_ctx_clear, fmpq_mpoly_ctx_init, fmpq_mpoly_get_term_coeff_fmpq,
    fmpq_mpoly_get_term_exp_ui, fmpq_mpoly_init, fmpq_mpoly_length, fmpq_mpoly_mul,
    fmpq_mpoly_set_coeff_fmpq_ui, fmpz_mod_mpoly_clear, fmpz_mod_mpoly_ctx_clear,
    fmpz_mod_mpoly_ctx_init, fmpz_mod_mpoly_get_term_coeff_fmpz, fmpz_mod_mpoly_get_term_exp_ui,
    fmpz_mod_mpoly_init, fmpz_mod_mpoly_length, fmpz_mod_mpoly_mul,
    fmpz_mod_mpoly_set_coeff_fmpz_ui, fmpz_mpoly_clear, fmpz_mpoly_ctx_clear, fmpz_mpoly_ctx_init,
    fmpz_mpoly_get_term_coeff_fmpz, fmpz_mpoly_get_term_exp_ui, fmpz_mpoly_init, fmpz_mpoly_length,
    fmpz_mpoly_mul, fmpz_mpoly_set_coeff_fmpz_ui, fq_nmod_mpoly_clear, fq_nmod_mpoly_ctx_clear,
    fq_nmod_mpoly_ctx_init, fq_nmod_mpoly_get_term_coeff_fq_nmod, fq_nmod_mpoly_get_term_exp_ui,
    fq_nmod_mpoly_init, fq_nmod_mpoly_length, fq_nmod_mpoly_mul,
    fq_nmod_mpoly_set_coeff_fq_nmod_ui, nmod_mpoly_clear, nmod_mpoly_ctx_clear,
    nmod_mpoly_ctx_init, nmod_mpoly_get_term_coeff_ui, nmod_mpoly_get_term_exp_ui, nmod_mpoly_init,
    nmod_mpoly_length, nmod_mpoly_mul, nmod_mpoly_set_coeff_ui_ui,
};

use crate::numthy::{
    DefaultExtensionElement, ExtensionContext, ExtensionElement, FiniteField, FiniteFieldElement,
};
use crate::{CommutativeRing, Integer, IntegerRing, Rational, RationalField, Ring};

#[cfg(test)]
use super::PolynomialRing;
use super::{MonomialOrder, ResidueClass, ResidueRing};

// The legacy transient adapters below remain only as a differential harness
// for the persistent representation. Their measured conversion crossovers are
// test data, never public mathematical API or runtime routing policy.
#[cfg(test)]
const FLINT_SCALAR_TERM_PRODUCT_CROSSOVER: usize = 1_024;
#[cfg(test)]
const FLINT_EXTENSION_TERM_PRODUCT_CROSSOVER: usize = 64;
#[cfg(test)]
const FLINT_SERIES_SCALAR_TERM_PRODUCT_CROSSOVER: usize = 16;
#[cfg(test)]
const FLINT_SERIES_MODULAR_TERM_PRODUCT_CROSSOVER: usize = 1;
// Measured with both conversions included: arbitrary moduli cross at 4x4
// terms, while fq_nmod extensions cross at 2x2 terms.
#[cfg(test)]
const FLINT_SERIES_BIG_MODULAR_TERM_PRODUCT_CROSSOVER: usize = 16;
#[cfg(test)]
const FLINT_SERIES_EXTENSION_TERM_PRODUCT_CROSSOVER: usize = 4;

pub(super) enum FlintPolynomialContext {
    Integer {
        raw: flint_sys::fmpz_mpoly_ctx_struct,
        nvars: usize,
        order: MonomialOrder,
    },
    Rational {
        raw: flint_sys::fmpq_mpoly_ctx_struct,
        nvars: usize,
        order: MonomialOrder,
    },
    Nmod {
        raw: flint_sys::nmod_mpoly_ctx_struct,
        nvars: usize,
        order: MonomialOrder,
        modulus: u64,
    },
    FmpzMod {
        raw: flint_sys::fmpz_mod_mpoly_ctx_struct,
        nvars: usize,
        order: MonomialOrder,
        modulus: Integer,
    },
    FqNmod {
        raw: flint_sys::fq_nmod_mpoly_ctx_struct,
        nvars: usize,
        order: MonomialOrder,
        field: Arc<ExtensionContext>,
        characteristic: u64,
        modulus: Vec<u64>,
    },
}

unsafe impl Send for FlintPolynomialContext {}
unsafe impl Sync for FlintPolynomialContext {}

impl FlintPolynomialContext {
    fn nvars(&self) -> usize {
        match self {
            Self::Integer { nvars, .. }
            | Self::Rational { nvars, .. }
            | Self::Nmod { nvars, .. }
            | Self::FmpzMod { nvars, .. }
            | Self::FqNmod { nvars, .. } => *nvars,
        }
    }

    fn order(&self) -> &MonomialOrder {
        match self {
            Self::Integer { order, .. }
            | Self::Rational { order, .. }
            | Self::Nmod { order, .. }
            | Self::FmpzMod { order, .. }
            | Self::FqNmod { order, .. } => order,
        }
    }
}

impl Drop for FlintPolynomialContext {
    fn drop(&mut self) {
        unsafe {
            match self {
                Self::Integer { raw, .. } => flint_sys::fmpz_mpoly_ctx_clear(raw),
                Self::Rational { raw, .. } => flint_sys::fmpq_mpoly_ctx_clear(raw),
                Self::Nmod { raw, .. } => flint_sys::nmod_mpoly_ctx_clear(raw),
                Self::FmpzMod { raw, .. } => flint_sys::fmpz_mod_mpoly_ctx_clear(raw),
                Self::FqNmod { raw, .. } => flint_sys::fq_nmod_mpoly_ctx_clear(raw),
            }
        }
    }
}

pub(super) fn context<R: CommutativeRing>(
    coefficient_ring: &R,
    nvars: usize,
    monomial_order: &MonomialOrder,
) -> Option<Arc<FlintPolynomialContext>> {
    let nvars_word = slong::try_from(nvars).ok()?;
    let ordering = order(monomial_order)?;
    if is_context::<R, IntegerRing, Integer>() {
        let mut raw = MaybeUninit::<flint_sys::fmpz_mpoly_ctx_struct>::uninit();
        unsafe { flint_sys::fmpz_mpoly_ctx_init(raw.as_mut_ptr(), nvars_word, ordering) };
        return Some(Arc::new(FlintPolynomialContext::Integer {
            raw: unsafe { raw.assume_init() },
            nvars,
            order: monomial_order.clone(),
        }));
    }
    if is_context::<R, RationalField, Rational>() {
        let mut raw = MaybeUninit::<flint_sys::fmpq_mpoly_ctx_struct>::uninit();
        unsafe { flint_sys::fmpq_mpoly_ctx_init(raw.as_mut_ptr(), nvars_word, ordering) };
        return Some(Arc::new(FlintPolynomialContext::Rational {
            raw: unsafe { raw.assume_init() },
            nvars,
            order: monomial_order.clone(),
        }));
    }
    if is_context::<R, ResidueRing, ResidueClass>() {
        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
        if let Some(modulus) = ring.modulus().to_u64_checked() {
            let mut raw = MaybeUninit::<flint_sys::nmod_mpoly_ctx_struct>::uninit();
            unsafe {
                flint_sys::nmod_mpoly_ctx_init(raw.as_mut_ptr(), nvars_word, ordering, modulus)
            };
            return Some(Arc::new(FlintPolynomialContext::Nmod {
                raw: unsafe { raw.assume_init() },
                nvars,
                order: monomial_order.clone(),
                modulus,
            }));
        }
        let mut raw = MaybeUninit::<flint_sys::fmpz_mod_mpoly_ctx_struct>::uninit();
        unsafe {
            flint_sys::fmpz_mod_mpoly_ctx_init(
                raw.as_mut_ptr(),
                nvars_word,
                ordering,
                ring.modulus().as_ptr(),
            )
        };
        return Some(Arc::new(FlintPolynomialContext::FmpzMod {
            raw: unsafe { raw.assume_init() },
            nvars,
            order: monomial_order.clone(),
            modulus: ring.modulus().clone(),
        }));
    }
    if is_context::<R, FiniteField, FiniteFieldElement>() {
        let field = cast_ref::<R, FiniteField>(coefficient_ring);
        if field.degree() == 1 {
            if let Some(modulus) = field.word_characteristic() {
                let mut raw = MaybeUninit::<flint_sys::nmod_mpoly_ctx_struct>::uninit();
                unsafe {
                    flint_sys::nmod_mpoly_ctx_init(raw.as_mut_ptr(), nvars_word, ordering, modulus)
                };
                return Some(Arc::new(FlintPolynomialContext::Nmod {
                    raw: unsafe { raw.assume_init() },
                    nvars,
                    order: monomial_order.clone(),
                    modulus,
                }));
            }
            let mut raw = MaybeUninit::<flint_sys::fmpz_mod_mpoly_ctx_struct>::uninit();
            unsafe {
                flint_sys::fmpz_mod_mpoly_ctx_init(
                    raw.as_mut_ptr(),
                    nvars_word,
                    ordering,
                    field.characteristic().as_ptr(),
                )
            };
            return Some(Arc::new(FlintPolynomialContext::FmpzMod {
                raw: unsafe { raw.assume_init() },
                nvars,
                order: monomial_order.clone(),
                modulus: field.characteristic().clone(),
            }));
        }
        let characteristic = field.word_characteristic()?;
        let modulus = field.defining_modulus()?.to_vec();
        let extension = field.extension_context()?;
        let mut raw = MaybeUninit::<flint_sys::fq_nmod_mpoly_ctx_struct>::uninit();
        unsafe {
            flint_sys::fq_nmod_mpoly_ctx_init(
                raw.as_mut_ptr(),
                nvars_word,
                ordering,
                extension.as_ptr(),
            )
        };
        return Some(Arc::new(FlintPolynomialContext::FqNmod {
            raw: unsafe { raw.assume_init() },
            nvars,
            order: monomial_order.clone(),
            field: extension,
            characteristic,
            modulus,
        }));
    }
    None
}

enum FlintPolynomialRaw {
    Integer(flint_sys::fmpz_mpoly_struct),
    Rational(flint_sys::fmpq_mpoly_struct),
    Nmod(flint_sys::nmod_mpoly_struct),
    FmpzMod(flint_sys::fmpz_mod_mpoly_struct),
    FqNmod(flint_sys::fq_nmod_mpoly_struct),
}

pub(super) struct FlintPolynomial {
    context: Arc<FlintPolynomialContext>,
    raw: FlintPolynomialRaw,
}

unsafe impl Send for FlintPolynomial {}
unsafe impl Sync for FlintPolynomial {}

impl Drop for FlintPolynomial {
    fn drop(&mut self) {
        unsafe {
            match (&mut self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(raw),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_clear(raw, ctx),
                (
                    FlintPolynomialRaw::Rational(raw),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_clear(raw, ctx),
                (FlintPolynomialRaw::Nmod(raw), FlintPolynomialContext::Nmod { raw: ctx, .. }) => {
                    flint_sys::nmod_mpoly_clear(raw, ctx)
                }
                (
                    FlintPolynomialRaw::FmpzMod(raw),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_clear(raw, ctx),
                (
                    FlintPolynomialRaw::FqNmod(raw),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_clear(raw, ctx),
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        }
    }
}

impl FlintPolynomial {
    fn zero(context: Arc<FlintPolynomialContext>) -> Self {
        let raw = match context.as_ref() {
            FlintPolynomialContext::Integer { raw: ctx, .. } => {
                let mut raw = MaybeUninit::<flint_sys::fmpz_mpoly_struct>::uninit();
                unsafe { flint_sys::fmpz_mpoly_init(raw.as_mut_ptr(), ctx) };
                FlintPolynomialRaw::Integer(unsafe { raw.assume_init() })
            }
            FlintPolynomialContext::Rational { raw: ctx, .. } => {
                let mut raw = MaybeUninit::<flint_sys::fmpq_mpoly_struct>::uninit();
                unsafe { flint_sys::fmpq_mpoly_init(raw.as_mut_ptr(), ctx) };
                FlintPolynomialRaw::Rational(unsafe { raw.assume_init() })
            }
            FlintPolynomialContext::Nmod { raw: ctx, .. } => {
                let mut raw = MaybeUninit::<flint_sys::nmod_mpoly_struct>::uninit();
                unsafe { flint_sys::nmod_mpoly_init(raw.as_mut_ptr(), ctx) };
                FlintPolynomialRaw::Nmod(unsafe { raw.assume_init() })
            }
            FlintPolynomialContext::FmpzMod { raw: ctx, .. } => {
                let mut raw = MaybeUninit::<flint_sys::fmpz_mod_mpoly_struct>::uninit();
                unsafe { flint_sys::fmpz_mod_mpoly_init(raw.as_mut_ptr(), ctx) };
                FlintPolynomialRaw::FmpzMod(unsafe { raw.assume_init() })
            }
            FlintPolynomialContext::FqNmod { raw: ctx, .. } => {
                let mut raw = MaybeUninit::<flint_sys::fq_nmod_mpoly_struct>::uninit();
                unsafe { flint_sys::fq_nmod_mpoly_init(raw.as_mut_ptr(), ctx) };
                FlintPolynomialRaw::FqNmod(unsafe { raw.assume_init() })
            }
        };
        Self { context, raw }
    }

    fn compatible(&self, other: &Self) -> bool {
        use FlintPolynomialContext::{FmpzMod, FqNmod, Integer as Z, Nmod, Rational as Q};

        match (self.context.as_ref(), other.context.as_ref()) {
            (
                Z {
                    nvars: left_nvars,
                    order: left_order,
                    ..
                },
                Z {
                    nvars: right_nvars,
                    order: right_order,
                    ..
                },
            )
            | (
                Q {
                    nvars: left_nvars,
                    order: left_order,
                    ..
                },
                Q {
                    nvars: right_nvars,
                    order: right_order,
                    ..
                },
            ) => left_nvars == right_nvars && left_order == right_order,
            (
                Nmod {
                    nvars: left_nvars,
                    order: left_order,
                    modulus: left_modulus,
                    ..
                },
                Nmod {
                    nvars: right_nvars,
                    order: right_order,
                    modulus: right_modulus,
                    ..
                },
            ) => {
                left_nvars == right_nvars
                    && left_order == right_order
                    && left_modulus == right_modulus
            }
            (
                FmpzMod {
                    nvars: left_nvars,
                    order: left_order,
                    modulus: left_modulus,
                    ..
                },
                FmpzMod {
                    nvars: right_nvars,
                    order: right_order,
                    modulus: right_modulus,
                    ..
                },
            ) => {
                left_nvars == right_nvars
                    && left_order == right_order
                    && left_modulus == right_modulus
            }
            (
                FqNmod {
                    nvars: left_nvars,
                    order: left_order,
                    characteristic: left_characteristic,
                    modulus: left_modulus,
                    ..
                },
                FqNmod {
                    nvars: right_nvars,
                    order: right_order,
                    characteristic: right_characteristic,
                    modulus: right_modulus,
                    ..
                },
            ) => {
                left_nvars == right_nvars
                    && left_order == right_order
                    && left_characteristic == right_characteristic
                    && left_modulus == right_modulus
            }
            _ => false,
        }
    }

    pub(super) fn len(&self) -> usize {
        unsafe {
            match (&self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(raw),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_length(raw, ctx) as usize,
                (
                    FlintPolynomialRaw::Rational(raw),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_length(raw, ctx) as usize,
                (FlintPolynomialRaw::Nmod(raw), FlintPolynomialContext::Nmod { raw: ctx, .. }) => {
                    flint_sys::nmod_mpoly_length(raw, ctx) as usize
                }
                (
                    FlintPolynomialRaw::FmpzMod(raw),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_length(raw, ctx) as usize,
                (
                    FlintPolynomialRaw::FqNmod(raw),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_length(raw, ctx) as usize,
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        }
    }

    pub(super) fn is_zero(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn pow_u64(&self, exponent: u64) -> Option<Self> {
        let mut output = Self::zero(Arc::clone(&self.context));
        let succeeded = unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(input),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_pow_ui(output, input, exponent, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(input),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_pow_ui(output, input, exponent, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(input),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                ) => flint_sys::nmod_mpoly_pow_ui(output, input, exponent, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(input),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_pow_ui(output, input, exponent, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(input),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_pow_ui(output, input, exponent, ctx),
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        };
        (succeeded != 0).then_some(output)
    }

    pub(super) fn derivative(&self, variable: usize) -> Option<Self> {
        if variable >= self.context.nvars() {
            return None;
        }
        let variable = slong::try_from(variable).ok()?;
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(input),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_derivative(output, input, variable, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(input),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_derivative(output, input, variable, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(input),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                ) => flint_sys::nmod_mpoly_derivative(output, input, variable, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(input),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_derivative(output, input, variable, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(input),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_derivative(output, input, variable, ctx),
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        }
        Some(output)
    }

    pub(super) fn evaluate<R: CommutativeRing>(
        &self,
        coefficient_ring: &R,
        values: &[R::Element],
    ) -> Option<R::Element> {
        if values.len() != self.context.nvars() {
            return None;
        }
        unsafe {
            match (&self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(input),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => {
                    let values = values
                        .iter()
                        .map(|value| cast_ref::<_, Integer>(value).as_ptr().cast_mut())
                        .collect::<Vec<_>>();
                    let mut output = Integer::zero();
                    (flint_sys::fmpz_mpoly_evaluate_all_fmpz(
                        output.as_mut_ptr(),
                        input,
                        values.as_ptr(),
                        ctx,
                    ) != 0)
                        .then(|| cast_owned(output))
                }
                (
                    FlintPolynomialRaw::Rational(input),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => {
                    let values = values
                        .iter()
                        .map(|value| cast_ref::<_, Rational>(value).as_ptr().cast_mut())
                        .collect::<Vec<_>>();
                    let mut output = Rational::zero();
                    (flint_sys::fmpq_mpoly_evaluate_all_fmpq(
                        output.as_mut_ptr(),
                        input,
                        values.as_ptr(),
                        ctx,
                    ) != 0)
                        .then(|| cast_owned(output))
                }
                (
                    FlintPolynomialRaw::Nmod(input),
                    FlintPolynomialContext::Nmod {
                        raw: ctx, modulus, ..
                    },
                ) => {
                    let values = values
                        .iter()
                        .map(|value| {
                            if is_context::<R, ResidueRing, ResidueClass>() {
                                cast_ref::<_, ResidueClass>(value).representative()
                            } else {
                                cast_ref::<_, FiniteFieldElement>(value).representative()
                            }
                            .modulo_u64(*modulus)
                        })
                        .collect::<Vec<_>>();
                    let output = flint_sys::nmod_mpoly_evaluate_all_ui(input, values.as_ptr(), ctx);
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        Some(cast_owned(ring.element(output)))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        Some(cast_owned(field.element(output)))
                    }
                }
                (
                    FlintPolynomialRaw::FmpzMod(input),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => {
                    let values = values
                        .iter()
                        .map(|value| {
                            if is_context::<R, ResidueRing, ResidueClass>() {
                                cast_ref::<_, ResidueClass>(value).representative()
                            } else {
                                cast_ref::<_, FiniteFieldElement>(value).representative()
                            }
                            .as_ptr()
                            .cast_mut()
                        })
                        .collect::<Vec<_>>();
                    let mut output = Integer::zero();
                    flint_sys::fmpz_mod_mpoly_evaluate_all_fmpz(
                        output.as_mut_ptr(),
                        input,
                        values.as_ptr(),
                        ctx,
                    );
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        Some(cast_owned(ring.element(output)))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        Some(cast_owned(field.element(output)))
                    }
                }
                (
                    FlintPolynomialRaw::FqNmod(input),
                    FlintPolynomialContext::FqNmod {
                        raw: ctx,
                        field: extension,
                        ..
                    },
                ) => {
                    let values = values
                        .iter()
                        .map(|value| {
                            cast_ref::<_, FiniteFieldElement>(value)
                                .extension_value()
                                .map(|value| {
                                    (&value.raw as *const flint_sys::fq_nmod_struct).cast_mut()
                                })
                        })
                        .collect::<Option<Vec<_>>>()?;
                    let mut output = ExtensionElement::initialized(Arc::clone(extension));
                    flint_sys::fq_nmod_mpoly_evaluate_all_fq_nmod(
                        &mut output.raw,
                        input,
                        values.as_ptr(),
                        ctx,
                    );
                    let field = cast_ref::<R, FiniteField>(coefficient_ring);
                    Some(cast_owned(field.wrap_extension(output)))
                }
                _ => None,
            }
        }
    }

    pub(super) fn compose(&self, images: &[&Self]) -> Option<Self> {
        if images.len() != self.context.nvars()
            || images.iter().any(|image| !self.compatible(image))
        {
            return None;
        }
        let mut output = Self::zero(Arc::clone(&self.context));
        let succeeded = unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(input),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => {
                    let images = images
                        .iter()
                        .map(|image| match &image.raw {
                            FlintPolynomialRaw::Integer(raw) => {
                                Some((raw as *const flint_sys::fmpz_mpoly_struct).cast_mut())
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    flint_sys::fmpz_mpoly_compose_fmpz_mpoly(
                        output,
                        input,
                        images.as_ptr(),
                        ctx,
                        ctx,
                    )
                }
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(input),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => {
                    let images = images
                        .iter()
                        .map(|image| match &image.raw {
                            FlintPolynomialRaw::Rational(raw) => {
                                Some((raw as *const flint_sys::fmpq_mpoly_struct).cast_mut())
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    flint_sys::fmpq_mpoly_compose_fmpq_mpoly(
                        output,
                        input,
                        images.as_ptr(),
                        ctx,
                        ctx,
                    )
                }
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(input),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                ) => {
                    let images = images
                        .iter()
                        .map(|image| match &image.raw {
                            FlintPolynomialRaw::Nmod(raw) => {
                                Some((raw as *const flint_sys::nmod_mpoly_struct).cast_mut())
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    flint_sys::nmod_mpoly_compose_nmod_mpoly(
                        output,
                        input,
                        images.as_ptr(),
                        ctx,
                        ctx,
                    )
                }
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(input),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => {
                    let images = images
                        .iter()
                        .map(|image| match &image.raw {
                            FlintPolynomialRaw::FmpzMod(raw) => {
                                Some((raw as *const flint_sys::fmpz_mod_mpoly_struct).cast_mut())
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    flint_sys::fmpz_mod_mpoly_compose_fmpz_mod_mpoly(
                        output,
                        input,
                        images.as_ptr(),
                        ctx,
                        ctx,
                    )
                }
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(input),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => {
                    let images = images
                        .iter()
                        .map(|image| match &image.raw {
                            FlintPolynomialRaw::FqNmod(raw) => {
                                Some((raw as *const flint_sys::fq_nmod_mpoly_struct).cast_mut())
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    flint_sys::fq_nmod_mpoly_compose_fq_nmod_mpoly(
                        output,
                        input,
                        images.as_ptr(),
                        ctx,
                        ctx,
                    )
                }
                _ => return None,
            }
        };
        (succeeded != 0).then_some(output)
    }

    /// Return whether `divisor` divides `self`, together with FLINT's quotient
    /// storage. The quotient is meaningful only when the flag is true.
    pub(super) fn divides(&self, divisor: &Self) -> Option<(bool, Self)> {
        if !self.compatible(divisor) {
            return None;
        }
        let mut output = Self::zero(Arc::clone(&self.context));
        let succeeded = unsafe {
            match (
                &mut output.raw,
                &self.raw,
                &divisor.raw,
                self.context.as_ref(),
            ) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(dividend),
                    FlintPolynomialRaw::Integer(divisor),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_divides(output, dividend, divisor, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(dividend),
                    FlintPolynomialRaw::Rational(divisor),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_divides(output, dividend, divisor, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(dividend),
                    FlintPolynomialRaw::Nmod(divisor),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                ) => flint_sys::nmod_mpoly_divides(output, dividend, divisor, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(dividend),
                    FlintPolynomialRaw::FmpzMod(divisor),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_divides(output, dividend, divisor, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(dividend),
                    FlintPolynomialRaw::FqNmod(divisor),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_divides(output, dividend, divisor, ctx),
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        };
        Some((succeeded != 0, output))
    }

    pub(super) fn gcd(&self, other: &Self) -> Option<Self> {
        if !self.compatible(other) {
            return None;
        }
        let mut output = Self::zero(Arc::clone(&self.context));
        let succeeded = unsafe {
            match (
                &mut output.raw,
                &self.raw,
                &other.raw,
                self.context.as_ref(),
            ) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(left),
                    FlintPolynomialRaw::Integer(right),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_gcd(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(left),
                    FlintPolynomialRaw::Rational(right),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_gcd(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(left),
                    FlintPolynomialRaw::Nmod(right),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                ) => flint_sys::nmod_mpoly_gcd(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(left),
                    FlintPolynomialRaw::FmpzMod(right),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_gcd(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(left),
                    FlintPolynomialRaw::FqNmod(right),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_gcd(output, left, right, ctx),
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        };
        (succeeded != 0).then_some(output)
    }

    pub(super) fn equal(&self, other: &Self) -> Option<bool> {
        if !self.compatible(other) {
            return None;
        }
        Some(unsafe {
            match (&self.raw, &other.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(left),
                    FlintPolynomialRaw::Integer(right),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_equal(left, right, ctx) != 0,
                (
                    FlintPolynomialRaw::Rational(left),
                    FlintPolynomialRaw::Rational(right),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_equal(left, right, ctx) != 0,
                (
                    FlintPolynomialRaw::Nmod(left),
                    FlintPolynomialRaw::Nmod(right),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                ) => flint_sys::nmod_mpoly_equal(left, right, ctx) != 0,
                (
                    FlintPolynomialRaw::FmpzMod(left),
                    FlintPolynomialRaw::FmpzMod(right),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_equal(left, right, ctx) != 0,
                (
                    FlintPolynomialRaw::FqNmod(left),
                    FlintPolynomialRaw::FqNmod(right),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_equal(left, right, ctx) != 0,
                _ => unreachable!("compatible FLINT polynomial storage has matching variants"),
            }
        })
    }

    pub(super) fn coefficient<R: CommutativeRing>(
        &self,
        coefficient_ring: &R,
        exponent: &[Integer],
    ) -> R::Element {
        let Some(exponent) = exponent
            .iter()
            .map(Integer::to_u64_checked)
            .collect::<Option<Vec<_>>>()
        else {
            return coefficient_ring.zero();
        };
        unsafe {
            match (&self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(raw),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => {
                    let mut coefficient = Integer::zero();
                    flint_sys::fmpz_mpoly_get_coeff_fmpz_ui(
                        coefficient.as_mut_ptr(),
                        raw,
                        exponent.as_ptr(),
                        ctx,
                    );
                    cast_owned(coefficient)
                }
                (
                    FlintPolynomialRaw::Rational(raw),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => {
                    let mut coefficient = Rational::zero();
                    flint_sys::fmpq_mpoly_get_coeff_fmpq_ui(
                        coefficient.as_mut_ptr(),
                        raw,
                        exponent.as_ptr(),
                        ctx,
                    );
                    cast_owned(coefficient)
                }
                (FlintPolynomialRaw::Nmod(raw), FlintPolynomialContext::Nmod { raw: ctx, .. }) => {
                    let coefficient =
                        flint_sys::nmod_mpoly_get_coeff_ui_ui(raw, exponent.as_ptr(), ctx);
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        cast_owned(ring.element(coefficient))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        cast_owned(field.element(coefficient))
                    }
                }
                (
                    FlintPolynomialRaw::FmpzMod(raw),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => {
                    let mut coefficient = Integer::zero();
                    flint_sys::fmpz_mod_mpoly_get_coeff_fmpz_ui(
                        coefficient.as_mut_ptr(),
                        raw,
                        exponent.as_ptr(),
                        ctx,
                    );
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        cast_owned(ring.element(coefficient))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        cast_owned(field.element(coefficient))
                    }
                }
                (
                    FlintPolynomialRaw::FqNmod(raw),
                    FlintPolynomialContext::FqNmod {
                        raw: ctx, field, ..
                    },
                ) => {
                    let mut coefficient = ExtensionElement::initialized(Arc::clone(field));
                    flint_sys::fq_nmod_mpoly_get_coeff_fq_nmod_ui(
                        &mut coefficient.raw,
                        raw,
                        exponent.as_ptr(),
                        ctx,
                    );
                    let coefficient_ring = cast_ref::<R, FiniteField>(coefficient_ring);
                    cast_owned(coefficient_ring.wrap_extension(coefficient))
                }
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        }
    }

    pub(super) fn add(&self, other: &Self) -> Option<Self> {
        self.binary(other, NativeBinaryOperation::Add)
    }

    pub(super) fn sub(&self, other: &Self) -> Option<Self> {
        self.binary(other, NativeBinaryOperation::Sub)
    }

    pub(super) fn mul(&self, other: &Self) -> Option<Self> {
        if !self.compatible(other) || !self.product_exponents_fit(other) {
            return None;
        }
        self.binary(other, NativeBinaryOperation::Mul)
    }

    pub(super) fn sum_products(products: &[(&Self, &Self)]) -> Option<Self> {
        let (first_left, first_right) = products.first().copied()?;
        if products.iter().any(|&(left, right)| {
            !first_left.compatible(left)
                || !first_left.compatible(right)
                || !left.product_exponents_fit(right)
        }) {
            return None;
        }

        let mut sum = Self::zero(Arc::clone(&first_left.context));
        sum.assign_binary(first_left, first_right, NativeBinaryOperation::Mul);
        if products.len() == 1 {
            return Some(sum);
        }

        let mut product = Self::zero(Arc::clone(&first_left.context));
        let mut next_sum = Self::zero(Arc::clone(&first_left.context));
        for &(left, right) in &products[1..] {
            product.assign_binary(left, right, NativeBinaryOperation::Mul);
            next_sum.assign_binary(&sum, &product, NativeBinaryOperation::Add);
            std::mem::swap(&mut sum, &mut next_sum);
        }
        Some(sum)
    }

    fn binary(&self, other: &Self, operation: NativeBinaryOperation) -> Option<Self> {
        if !self.compatible(other) {
            return None;
        }
        let mut output = Self::zero(Arc::clone(&self.context));
        output.assign_binary(self, other, operation);
        Some(output)
    }

    fn assign_binary(&mut self, left: &Self, right: &Self, operation: NativeBinaryOperation) {
        debug_assert!(self.compatible(left));
        debug_assert!(self.compatible(right));
        unsafe {
            match (
                &mut self.raw,
                &left.raw,
                &right.raw,
                self.context.as_ref(),
                operation,
            ) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(left),
                    FlintPolynomialRaw::Integer(right),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                    NativeBinaryOperation::Add,
                ) => flint_sys::fmpz_mpoly_add(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(left),
                    FlintPolynomialRaw::Integer(right),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                    NativeBinaryOperation::Sub,
                ) => flint_sys::fmpz_mpoly_sub(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(left),
                    FlintPolynomialRaw::Integer(right),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                    NativeBinaryOperation::Mul,
                ) => flint_sys::fmpz_mpoly_mul(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(left),
                    FlintPolynomialRaw::Rational(right),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                    NativeBinaryOperation::Add,
                ) => flint_sys::fmpq_mpoly_add(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(left),
                    FlintPolynomialRaw::Rational(right),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                    NativeBinaryOperation::Sub,
                ) => flint_sys::fmpq_mpoly_sub(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(left),
                    FlintPolynomialRaw::Rational(right),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                    NativeBinaryOperation::Mul,
                ) => flint_sys::fmpq_mpoly_mul(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(left),
                    FlintPolynomialRaw::Nmod(right),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                    NativeBinaryOperation::Add,
                ) => flint_sys::nmod_mpoly_add(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(left),
                    FlintPolynomialRaw::Nmod(right),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                    NativeBinaryOperation::Sub,
                ) => flint_sys::nmod_mpoly_sub(output, left, right, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(left),
                    FlintPolynomialRaw::Nmod(right),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                    NativeBinaryOperation::Mul,
                ) => flint_sys::nmod_mpoly_mul(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(left),
                    FlintPolynomialRaw::FmpzMod(right),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                    NativeBinaryOperation::Add,
                ) => flint_sys::fmpz_mod_mpoly_add(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(left),
                    FlintPolynomialRaw::FmpzMod(right),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                    NativeBinaryOperation::Sub,
                ) => flint_sys::fmpz_mod_mpoly_sub(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(left),
                    FlintPolynomialRaw::FmpzMod(right),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                    NativeBinaryOperation::Mul,
                ) => flint_sys::fmpz_mod_mpoly_mul(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(left),
                    FlintPolynomialRaw::FqNmod(right),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                    NativeBinaryOperation::Add,
                ) => flint_sys::fq_nmod_mpoly_add(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(left),
                    FlintPolynomialRaw::FqNmod(right),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                    NativeBinaryOperation::Sub,
                ) => flint_sys::fq_nmod_mpoly_sub(output, left, right, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(left),
                    FlintPolynomialRaw::FqNmod(right),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                    NativeBinaryOperation::Mul,
                ) => flint_sys::fq_nmod_mpoly_mul(output, left, right, ctx),
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        }
    }

    pub(super) fn neg(&self) -> Option<Self> {
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(input),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_neg(output, input, ctx),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(input),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_neg(output, input, ctx),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(input),
                    FlintPolynomialContext::Nmod { raw: ctx, .. },
                ) => flint_sys::nmod_mpoly_neg(output, input, ctx),
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(input),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_mpoly_neg(output, input, ctx),
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(input),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => flint_sys::fq_nmod_mpoly_neg(output, input, ctx),
                _ => return None,
            }
        }
        Some(output)
    }

    fn visit_exponents(&self, mut visitor: impl FnMut(&[u64])) {
        let nvars = self.context.nvars();
        let mut exponent = vec![0_u64; nvars];
        for term in 0..self.len() {
            unsafe {
                match (&self.raw, self.context.as_ref()) {
                    (
                        FlintPolynomialRaw::Integer(raw),
                        FlintPolynomialContext::Integer { raw: ctx, .. },
                    ) => flint_sys::fmpz_mpoly_get_term_exp_ui(
                        exponent.as_mut_ptr(),
                        raw,
                        term as slong,
                        ctx,
                    ),
                    (
                        FlintPolynomialRaw::Rational(raw),
                        FlintPolynomialContext::Rational { raw: ctx, .. },
                    ) => flint_sys::fmpq_mpoly_get_term_exp_ui(
                        exponent.as_mut_ptr(),
                        raw,
                        term as slong,
                        ctx,
                    ),
                    (
                        FlintPolynomialRaw::Nmod(raw),
                        FlintPolynomialContext::Nmod { raw: ctx, .. },
                    ) => flint_sys::nmod_mpoly_get_term_exp_ui(
                        exponent.as_mut_ptr(),
                        raw,
                        term as slong,
                        ctx,
                    ),
                    (
                        FlintPolynomialRaw::FmpzMod(raw),
                        FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                    ) => flint_sys::fmpz_mod_mpoly_get_term_exp_ui(
                        exponent.as_mut_ptr(),
                        raw,
                        term as slong,
                        ctx,
                    ),
                    (
                        FlintPolynomialRaw::FqNmod(raw),
                        FlintPolynomialContext::FqNmod { raw: ctx, .. },
                    ) => flint_sys::fq_nmod_mpoly_get_term_exp_ui(
                        exponent.as_mut_ptr(),
                        raw,
                        term as slong,
                        ctx,
                    ),
                    _ => unreachable!("FLINT polynomial/context storage invariant"),
                }
            }
            visitor(&exponent);
        }
    }

    fn exponent_vectors(&self) -> Vec<Vec<u64>> {
        let mut exponents = Vec::with_capacity(self.len());
        self.visit_exponents(|exponent| exponents.push(exponent.to_vec()));
        exponents
    }

    pub(super) fn visit_residue_terms(
        &self,
        mut visitor: impl FnMut(&[u64], Integer),
    ) -> Option<()> {
        let nvars = self.context.nvars();
        let mut exponent = vec![0_u64; nvars];
        for term in 0..self.len() {
            let coefficient = unsafe {
                match (&self.raw, self.context.as_ref()) {
                    (
                        FlintPolynomialRaw::Nmod(raw),
                        FlintPolynomialContext::Nmod { raw: ctx, .. },
                    ) => {
                        flint_sys::nmod_mpoly_get_term_exp_ui(
                            exponent.as_mut_ptr(),
                            raw,
                            term as slong,
                            ctx,
                        );
                        Integer::from(flint_sys::nmod_mpoly_get_term_coeff_ui(
                            raw,
                            term as slong,
                            ctx,
                        ))
                    }
                    (
                        FlintPolynomialRaw::FmpzMod(raw),
                        FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                    ) => {
                        flint_sys::fmpz_mod_mpoly_get_term_exp_ui(
                            exponent.as_mut_ptr(),
                            raw,
                            term as slong,
                            ctx,
                        );
                        let mut coefficient = Integer::zero();
                        flint_sys::fmpz_mod_mpoly_get_term_coeff_fmpz(
                            coefficient.as_mut_ptr(),
                            raw,
                            term as slong,
                            ctx,
                        );
                        coefficient
                    }
                    _ => return None,
                }
            };
            visitor(&exponent, coefficient);
        }
        Some(())
    }

    fn product_exponents_fit(&self, other: &Self) -> bool {
        if self.is_zero() || other.is_zero() {
            return true;
        }
        let bounds = |polynomial: &Self| {
            let mut coordinate_maxima = vec![0_u64; polynomial.context.nvars()];
            let mut maximum_total = Some(0_u64);
            polynomial.visit_exponents(|exponent| {
                for (maximum, value) in coordinate_maxima.iter_mut().zip(exponent) {
                    *maximum = (*maximum).max(*value);
                }
                let total = exponent
                    .iter()
                    .try_fold(0_u64, |sum, value| sum.checked_add(*value));
                maximum_total = maximum_total
                    .zip(total)
                    .map(|(maximum, total)| maximum.max(total));
            });
            (coordinate_maxima, maximum_total)
        };
        let (left_maxima, left_total) = bounds(self);
        let (right_maxima, right_total) = bounds(other);
        for variable in 0..self.context.nvars() {
            if left_maxima[variable]
                .checked_add(right_maxima[variable])
                .is_none()
            {
                return false;
            }
        }
        if self.context.order() == &MonomialOrder::Lexicographic {
            return true;
        }
        left_total
            .zip(right_total)
            .is_some_and(|(left, right)| left.checked_add(right).is_some())
    }
}

#[derive(Clone, Copy)]
enum NativeBinaryOperation {
    Add,
    Sub,
    Mul,
}

pub(super) fn from_terms<R: CommutativeRing>(
    context: Arc<FlintPolynomialContext>,
    _coefficient_ring: &R,
    terms: &BTreeMap<Vec<Integer>, R::Element>,
) -> Option<FlintPolynomial> {
    let terms = machine_terms(terms)?;
    from_machine_terms::<R>(context, &terms)
}

pub(super) fn from_canonical_u64_terms<R: CommutativeRing>(
    context: Arc<FlintPolynomialContext>,
    _coefficient_ring: &R,
    terms: &BTreeMap<Vec<u64>, R::Element>,
) -> Option<FlintPolynomial> {
    if matches!(context.as_ref(), FlintPolynomialContext::FqNmod { .. }) {
        return None;
    }
    let terms = terms
        .iter()
        .map(|(exponent, coefficient)| (exponent.clone(), coefficient))
        .collect::<Vec<_>>();
    from_machine_terms::<R>(context, &terms)
}

/// Whether machine-word exponents should bypass the arbitrary-exponent map.
pub(super) fn supports_direct_u64_construction(context: &FlintPolynomialContext) -> bool {
    !matches!(context, FlintPolynomialContext::FqNmod { .. })
}

pub(super) fn from_canonical_u32_terms<R: CommutativeRing>(
    context: Arc<FlintPolynomialContext>,
    _coefficient_ring: &R,
    terms: &[(Vec<u32>, R::Element)],
) -> Option<FlintPolynomial> {
    let mut polynomial = FlintPolynomial::zero(context);
    let mut exponent = Vec::with_capacity(polynomial.context.nvars());
    unsafe {
        match (&mut polynomial.raw, polynomial.context.as_ref()) {
            (
                FlintPolynomialRaw::Integer(raw),
                FlintPolynomialContext::Integer { raw: ctx, .. },
            ) => {
                for (powers, coefficient) in terms {
                    copy_u32_exponent(powers, &mut exponent);
                    flint_sys::fmpz_mpoly_push_term_fmpz_ui(
                        raw,
                        cast_ref::<_, Integer>(coefficient).as_ptr(),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
            }
            (
                FlintPolynomialRaw::Rational(raw),
                FlintPolynomialContext::Rational { raw: ctx, .. },
            ) => {
                for (powers, coefficient) in terms {
                    copy_u32_exponent(powers, &mut exponent);
                    let coefficient = cast_ref::<_, Rational>(coefficient);
                    flint_sys::fmpq_mpoly_push_term_fmpq_ui(
                        raw,
                        coefficient.as_ptr(),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
                if !terms.is_empty() {
                    flint_sys::fmpq_mpoly_reduce(raw, ctx);
                }
            }
            (
                FlintPolynomialRaw::Nmod(raw),
                FlintPolynomialContext::Nmod {
                    raw: ctx, modulus, ..
                },
            ) => {
                for (powers, coefficient) in terms {
                    copy_u32_exponent(powers, &mut exponent);
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(coefficient).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(coefficient).representative()
                    };
                    flint_sys::nmod_mpoly_push_term_ui_ui(
                        raw,
                        representative.modulo_u64(*modulus),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
            }
            (
                FlintPolynomialRaw::FmpzMod(raw),
                FlintPolynomialContext::FmpzMod { raw: ctx, .. },
            ) => {
                for (powers, coefficient) in terms {
                    copy_u32_exponent(powers, &mut exponent);
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(coefficient).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(coefficient).representative()
                    };
                    flint_sys::fmpz_mod_mpoly_push_term_fmpz_ui(
                        raw,
                        representative.as_ptr(),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
            }
            (FlintPolynomialRaw::FqNmod(raw), FlintPolynomialContext::FqNmod { raw: ctx, .. }) => {
                for (powers, coefficient) in terms {
                    copy_u32_exponent(powers, &mut exponent);
                    let coefficient = cast_ref::<_, FiniteFieldElement>(coefficient)
                        .extension_value()
                        .expect("extension-field polynomial coefficient invariant");
                    flint_sys::fq_nmod_mpoly_push_term_fq_nmod_ui(
                        raw,
                        &coefficient.raw,
                        exponent.as_ptr(),
                        ctx,
                    );
                }
            }
            _ => return None,
        }
    }
    Some(polynomial)
}

fn copy_u32_exponent(source: &[u32], target: &mut Vec<u64>) {
    target.clear();
    target.extend(source.iter().copied().map(u64::from));
}

fn from_machine_terms<R: CommutativeRing>(
    context: Arc<FlintPolynomialContext>,
    terms: &[(Vec<u64>, &R::Element)],
) -> Option<FlintPolynomial> {
    let mut polynomial = FlintPolynomial::zero(context);
    unsafe {
        match (&mut polynomial.raw, polynomial.context.as_ref()) {
            (
                FlintPolynomialRaw::Integer(raw),
                FlintPolynomialContext::Integer { raw: ctx, .. },
            ) => {
                for (exponent, coefficient) in terms {
                    let coefficient = cast_ref::<_, Integer>(*coefficient);
                    flint_sys::fmpz_mpoly_push_term_fmpz_ui(
                        raw,
                        coefficient.as_ptr(),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
                flint_sys::fmpz_mpoly_sort_terms(raw, ctx);
            }
            (
                FlintPolynomialRaw::Rational(raw),
                FlintPolynomialContext::Rational { raw: ctx, .. },
            ) => {
                for (exponent, coefficient) in terms {
                    let coefficient = cast_ref::<_, Rational>(*coefficient);
                    flint_sys::fmpq_mpoly_push_term_fmpq_ui(
                        raw,
                        coefficient.as_ptr(),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
                flint_sys::fmpq_mpoly_sort_terms(raw, ctx);
                if !terms.is_empty() {
                    flint_sys::fmpq_mpoly_reduce(raw, ctx);
                }
            }
            (
                FlintPolynomialRaw::Nmod(raw),
                FlintPolynomialContext::Nmod {
                    raw: ctx, modulus, ..
                },
            ) => {
                for (exponent, coefficient) in terms {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(*coefficient).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(*coefficient).representative()
                    };
                    flint_sys::nmod_mpoly_push_term_ui_ui(
                        raw,
                        representative.modulo_u64(*modulus),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
                flint_sys::nmod_mpoly_sort_terms(raw, ctx);
            }
            (
                FlintPolynomialRaw::FmpzMod(raw),
                FlintPolynomialContext::FmpzMod { raw: ctx, .. },
            ) => {
                for (exponent, coefficient) in terms {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(*coefficient).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(*coefficient).representative()
                    };
                    flint_sys::fmpz_mod_mpoly_push_term_fmpz_ui(
                        raw,
                        representative.as_ptr(),
                        exponent.as_ptr(),
                        ctx,
                    );
                }
                flint_sys::fmpz_mod_mpoly_sort_terms(raw, ctx);
            }
            (FlintPolynomialRaw::FqNmod(raw), FlintPolynomialContext::FqNmod { raw: ctx, .. }) => {
                for (exponent, coefficient) in terms {
                    let coefficient = cast_ref::<_, FiniteFieldElement>(*coefficient)
                        .extension_value()
                        .expect("extension-field polynomial coefficient invariant");
                    flint_sys::fq_nmod_mpoly_set_coeff_fq_nmod_ui(
                        raw,
                        &coefficient.raw,
                        exponent.as_ptr(),
                        ctx,
                    );
                }
            }
            _ => return None,
        }
    }
    Some(polynomial)
}

impl FlintPolynomial {
    pub(super) fn leading_term<R: CommutativeRing>(
        &self,
        coefficient_ring: &R,
    ) -> Option<(Vec<Integer>, R::Element)> {
        if self.is_zero() {
            return None;
        }
        let mut exponent = vec![0_u64; self.context.nvars()];
        let coefficient = unsafe {
            match (&self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(raw),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => {
                    flint_sys::fmpz_mpoly_get_term_exp_ui(exponent.as_mut_ptr(), raw, 0, ctx);
                    let mut coefficient = Integer::zero();
                    flint_sys::fmpz_mpoly_get_term_coeff_fmpz(
                        coefficient.as_mut_ptr(),
                        raw,
                        0,
                        ctx,
                    );
                    cast_owned(coefficient)
                }
                (
                    FlintPolynomialRaw::Rational(raw),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => {
                    flint_sys::fmpq_mpoly_get_term_exp_ui(exponent.as_mut_ptr(), raw, 0, ctx);
                    let mut coefficient = Rational::zero();
                    flint_sys::fmpq_mpoly_get_term_coeff_fmpq(
                        coefficient.as_mut_ptr(),
                        raw,
                        0,
                        ctx,
                    );
                    cast_owned(coefficient)
                }
                (FlintPolynomialRaw::Nmod(raw), FlintPolynomialContext::Nmod { raw: ctx, .. }) => {
                    flint_sys::nmod_mpoly_get_term_exp_ui(exponent.as_mut_ptr(), raw, 0, ctx);
                    let coefficient = flint_sys::nmod_mpoly_get_term_coeff_ui(raw, 0, ctx);
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        cast_owned(ring.element(coefficient))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        cast_owned(field.element(coefficient))
                    }
                }
                (
                    FlintPolynomialRaw::FmpzMod(raw),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => {
                    flint_sys::fmpz_mod_mpoly_get_term_exp_ui(exponent.as_mut_ptr(), raw, 0, ctx);
                    let mut coefficient = Integer::zero();
                    flint_sys::fmpz_mod_mpoly_get_term_coeff_fmpz(
                        coefficient.as_mut_ptr(),
                        raw,
                        0,
                        ctx,
                    );
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        cast_owned(ring.element(coefficient))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        cast_owned(field.element(coefficient))
                    }
                }
                (
                    FlintPolynomialRaw::FqNmod(raw),
                    FlintPolynomialContext::FqNmod {
                        raw: ctx,
                        field: extension,
                        ..
                    },
                ) => {
                    flint_sys::fq_nmod_mpoly_get_term_exp_ui(exponent.as_mut_ptr(), raw, 0, ctx);
                    let mut coefficient = ExtensionElement::initialized(Arc::clone(extension));
                    flint_sys::fq_nmod_mpoly_get_term_coeff_fq_nmod(
                        &mut coefficient.raw,
                        raw,
                        0,
                        ctx,
                    );
                    let field = cast_ref::<R, FiniteField>(coefficient_ring);
                    cast_owned(field.wrap_extension(coefficient))
                }
                _ => unreachable!("FLINT polynomial/context storage invariant"),
            }
        };
        Some((
            exponent.into_iter().map(Integer::from).collect(),
            coefficient,
        ))
    }

    pub(super) fn to_terms<R: CommutativeRing>(
        &self,
        coefficient_ring: &R,
    ) -> BTreeMap<Vec<Integer>, R::Element> {
        let exponents = self.exponent_vectors();
        exponents
            .into_iter()
            .enumerate()
            .map(|(term, exponent)| {
                let coefficient = unsafe {
                    match (&self.raw, self.context.as_ref()) {
                        (
                            FlintPolynomialRaw::Integer(raw),
                            FlintPolynomialContext::Integer { raw: ctx, .. },
                        ) => {
                            let mut coefficient = Integer::zero();
                            flint_sys::fmpz_mpoly_get_term_coeff_fmpz(
                                coefficient.as_mut_ptr(),
                                raw,
                                term as slong,
                                ctx,
                            );
                            cast_owned(coefficient)
                        }
                        (
                            FlintPolynomialRaw::Rational(raw),
                            FlintPolynomialContext::Rational { raw: ctx, .. },
                        ) => {
                            let mut coefficient = Rational::zero();
                            flint_sys::fmpq_mpoly_get_term_coeff_fmpq(
                                coefficient.as_mut_ptr(),
                                raw,
                                term as slong,
                                ctx,
                            );
                            cast_owned(coefficient)
                        }
                        (
                            FlintPolynomialRaw::Nmod(raw),
                            FlintPolynomialContext::Nmod { raw: ctx, .. },
                        ) => {
                            let coefficient =
                                flint_sys::nmod_mpoly_get_term_coeff_ui(raw, term as slong, ctx);
                            if is_context::<R, ResidueRing, ResidueClass>() {
                                let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                                cast_owned(ring.element(coefficient))
                            } else {
                                let field = cast_ref::<R, FiniteField>(coefficient_ring);
                                cast_owned(field.element(coefficient))
                            }
                        }
                        (
                            FlintPolynomialRaw::FmpzMod(raw),
                            FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                        ) => {
                            let mut coefficient = Integer::zero();
                            flint_sys::fmpz_mod_mpoly_get_term_coeff_fmpz(
                                coefficient.as_mut_ptr(),
                                raw,
                                term as slong,
                                ctx,
                            );
                            if is_context::<R, ResidueRing, ResidueClass>() {
                                let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                                cast_owned(ring.element(coefficient))
                            } else {
                                let field = cast_ref::<R, FiniteField>(coefficient_ring);
                                cast_owned(field.element(coefficient))
                            }
                        }
                        (
                            FlintPolynomialRaw::FqNmod(raw),
                            FlintPolynomialContext::FqNmod {
                                raw: ctx,
                                field: extension,
                                ..
                            },
                        ) => {
                            let mut coefficient =
                                ExtensionElement::initialized(Arc::clone(extension));
                            flint_sys::fq_nmod_mpoly_get_term_coeff_fq_nmod(
                                &mut coefficient.raw,
                                raw,
                                term as slong,
                                ctx,
                            );
                            let field = cast_ref::<R, FiniteField>(coefficient_ring);
                            cast_owned(field.wrap_extension(coefficient))
                        }
                        _ => unreachable!("FLINT polynomial/context storage invariant"),
                    }
                };
                (
                    exponent.into_iter().map(Integer::from).collect(),
                    coefficient,
                )
            })
            .collect()
    }

    pub(super) fn scalar_mul<R: CommutativeRing>(
        &self,
        _coefficient_ring: &R,
        scalar: &R::Element,
    ) -> Option<Self> {
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (
                    FlintPolynomialRaw::Integer(output),
                    FlintPolynomialRaw::Integer(input),
                    FlintPolynomialContext::Integer { raw: ctx, .. },
                ) => flint_sys::fmpz_mpoly_scalar_mul_fmpz(
                    output,
                    input,
                    cast_ref::<_, Integer>(scalar).as_ptr(),
                    ctx,
                ),
                (
                    FlintPolynomialRaw::Rational(output),
                    FlintPolynomialRaw::Rational(input),
                    FlintPolynomialContext::Rational { raw: ctx, .. },
                ) => flint_sys::fmpq_mpoly_scalar_mul_fmpq(
                    output,
                    input,
                    cast_ref::<_, Rational>(scalar).as_ptr(),
                    ctx,
                ),
                (
                    FlintPolynomialRaw::Nmod(output),
                    FlintPolynomialRaw::Nmod(input),
                    FlintPolynomialContext::Nmod {
                        raw: ctx, modulus, ..
                    },
                ) => {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(scalar).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(scalar).representative()
                    };
                    flint_sys::nmod_mpoly_scalar_mul_ui(
                        output,
                        input,
                        representative.modulo_u64(*modulus),
                        ctx,
                    );
                }
                (
                    FlintPolynomialRaw::FmpzMod(output),
                    FlintPolynomialRaw::FmpzMod(input),
                    FlintPolynomialContext::FmpzMod { raw: ctx, .. },
                ) => {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(scalar).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(scalar).representative()
                    };
                    flint_sys::fmpz_mod_mpoly_scalar_mul_fmpz(
                        output,
                        input,
                        representative.as_ptr(),
                        ctx,
                    );
                }
                (
                    FlintPolynomialRaw::FqNmod(output),
                    FlintPolynomialRaw::FqNmod(input),
                    FlintPolynomialContext::FqNmod { raw: ctx, .. },
                ) => {
                    let coefficient = cast_ref::<_, FiniteFieldElement>(scalar)
                        .extension_value()
                        .expect("extension-field polynomial coefficient invariant");
                    flint_sys::fq_nmod_mpoly_scalar_mul_fq_nmod(
                        output,
                        input,
                        &coefficient.raw,
                        ctx,
                    );
                }
                _ => return None,
            }
        }
        Some(output)
    }
}

/// FLINT context selected for a fixed-precision univariate series ring.
///
/// Integer and rational polynomials do not need a FLINT context object, but
/// the precision still belongs here so every stored value shares one exact
/// allocation policy. Modular and extension polynomials additionally keep
/// their coefficient context alive for the full lifetime of every value.
pub(super) enum FlintSeriesContext {
    Integer {
        precision: slong,
    },
    Rational {
        precision: slong,
    },
    Nmod {
        precision: slong,
        modulus: u64,
    },
    FmpzMod {
        precision: slong,
        raw: fmpz_mod_ctx_struct,
        modulus: Integer,
    },
    FqNmod {
        precision: slong,
        field: Arc<ExtensionContext>,
        characteristic: u64,
        modulus: Vec<u64>,
    },
}

unsafe impl Send for FlintSeriesContext {}
unsafe impl Sync for FlintSeriesContext {}

impl Drop for FlintSeriesContext {
    fn drop(&mut self) {
        if let Self::FmpzMod { raw, .. } = self {
            unsafe { fmpz_mod_ctx_clear(raw) };
        }
    }
}

pub(super) fn series_context<R: CommutativeRing>(
    coefficient_ring: &R,
    precision: u64,
) -> Option<Arc<FlintSeriesContext>> {
    let precision = slong::try_from(precision).ok()?;
    if is_context::<R, IntegerRing, Integer>() {
        return Some(Arc::new(FlintSeriesContext::Integer { precision }));
    }
    if is_context::<R, RationalField, Rational>() {
        return Some(Arc::new(FlintSeriesContext::Rational { precision }));
    }
    if is_context::<R, ResidueRing, ResidueClass>() {
        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
        if let Some(modulus) = ring.modulus().to_u64_checked() {
            return Some(Arc::new(FlintSeriesContext::Nmod { precision, modulus }));
        }
        let mut raw = MaybeUninit::<fmpz_mod_ctx_struct>::uninit();
        unsafe { fmpz_mod_ctx_init(raw.as_mut_ptr(), ring.modulus().as_ptr()) };
        return Some(Arc::new(FlintSeriesContext::FmpzMod {
            precision,
            raw: unsafe { raw.assume_init() },
            modulus: ring.modulus().clone(),
        }));
    }
    if is_context::<R, FiniteField, FiniteFieldElement>() {
        let field = cast_ref::<R, FiniteField>(coefficient_ring);
        if field.degree() == 1 {
            if let Some(modulus) = field.word_characteristic() {
                return Some(Arc::new(FlintSeriesContext::Nmod { precision, modulus }));
            }
            let mut raw = MaybeUninit::<fmpz_mod_ctx_struct>::uninit();
            unsafe { fmpz_mod_ctx_init(raw.as_mut_ptr(), field.characteristic().as_ptr()) };
            return Some(Arc::new(FlintSeriesContext::FmpzMod {
                precision,
                raw: unsafe { raw.assume_init() },
                modulus: field.characteristic().clone(),
            }));
        }
        let characteristic = field.word_characteristic()?;
        let modulus = field.defining_modulus()?.to_vec();
        return Some(Arc::new(FlintSeriesContext::FqNmod {
            precision,
            field: field.extension_context()?,
            characteristic,
            modulus,
        }));
    }
    None
}

enum FlintSeriesRaw {
    Integer(flint_sys::fmpz_poly_struct),
    Rational(flint_sys::fmpq_poly_struct),
    Nmod(flint_sys::nmod_poly_struct),
    FmpzMod(flint_sys::fmpz_mod_poly_struct),
    FqDefault(flint_sys::fq_default_poly_struct),
}

pub(super) struct FlintSeries {
    context: Arc<FlintSeriesContext>,
    raw: FlintSeriesRaw,
}

unsafe impl Send for FlintSeries {}
unsafe impl Sync for FlintSeries {}

impl Drop for FlintSeries {
    fn drop(&mut self) {
        unsafe {
            match (&mut self.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(raw), FlintSeriesContext::Integer { .. }) => {
                    fmpz_poly_clear(raw)
                }
                (FlintSeriesRaw::Rational(raw), FlintSeriesContext::Rational { .. }) => {
                    fmpq_poly_clear(raw)
                }
                (FlintSeriesRaw::Nmod(raw), FlintSeriesContext::Nmod { .. }) => {
                    nmod_poly_clear(raw)
                }
                (FlintSeriesRaw::FmpzMod(raw), FlintSeriesContext::FmpzMod { raw: ctx, .. }) => {
                    flint_sys::fmpz_mod_poly_clear(raw, ctx)
                }
                (FlintSeriesRaw::FqDefault(raw), FlintSeriesContext::FqNmod { field, .. }) => {
                    fq_default_poly_clear(raw, field.default_as_ptr())
                }
                _ => unreachable!("FLINT series/context storage invariant"),
            }
        }
    }
}

impl FlintSeries {
    fn zero(context: Arc<FlintSeriesContext>) -> Self {
        let raw = match context.as_ref() {
            FlintSeriesContext::Integer { .. } => {
                let mut raw = MaybeUninit::<flint_sys::fmpz_poly_struct>::uninit();
                unsafe { fmpz_poly_init(raw.as_mut_ptr()) };
                FlintSeriesRaw::Integer(unsafe { raw.assume_init() })
            }
            FlintSeriesContext::Rational { .. } => {
                let mut raw = MaybeUninit::<flint_sys::fmpq_poly_struct>::uninit();
                unsafe { fmpq_poly_init(raw.as_mut_ptr()) };
                FlintSeriesRaw::Rational(unsafe { raw.assume_init() })
            }
            FlintSeriesContext::Nmod { modulus, .. } => {
                let mut raw = MaybeUninit::<flint_sys::nmod_poly_struct>::uninit();
                unsafe { nmod_poly_init(raw.as_mut_ptr(), *modulus) };
                FlintSeriesRaw::Nmod(unsafe { raw.assume_init() })
            }
            FlintSeriesContext::FmpzMod { raw: ctx, .. } => {
                let mut raw = MaybeUninit::<flint_sys::fmpz_mod_poly_struct>::uninit();
                unsafe { fmpz_mod_poly_init(raw.as_mut_ptr(), ctx) };
                FlintSeriesRaw::FmpzMod(unsafe { raw.assume_init() })
            }
            FlintSeriesContext::FqNmod { field, .. } => {
                let mut raw = MaybeUninit::<flint_sys::fq_default_poly_struct>::uninit();
                unsafe { fq_default_poly_init(raw.as_mut_ptr(), field.default_as_ptr()) };
                FlintSeriesRaw::FqDefault(unsafe { raw.assume_init() })
            }
        };
        Self { context, raw }
    }

    fn compatible(&self, other: &Self) -> bool {
        use FlintSeriesContext::{FmpzMod, FqNmod, Integer as Z, Nmod, Rational as Q};

        match (self.context.as_ref(), other.context.as_ref()) {
            (Z { precision: left }, Z { precision: right })
            | (Q { precision: left }, Q { precision: right }) => left == right,
            (
                Nmod {
                    precision: left_precision,
                    modulus: left_modulus,
                },
                Nmod {
                    precision: right_precision,
                    modulus: right_modulus,
                },
            ) => left_precision == right_precision && left_modulus == right_modulus,
            (
                FmpzMod {
                    precision: left_precision,
                    modulus: left_modulus,
                    ..
                },
                FmpzMod {
                    precision: right_precision,
                    modulus: right_modulus,
                    ..
                },
            ) => left_precision == right_precision && left_modulus == right_modulus,
            (
                FqNmod {
                    precision: left_precision,
                    characteristic: left_characteristic,
                    modulus: left_modulus,
                    ..
                },
                FqNmod {
                    precision: right_precision,
                    characteristic: right_characteristic,
                    modulus: right_modulus,
                    ..
                },
            ) => {
                left_precision == right_precision
                    && left_characteristic == right_characteristic
                    && left_modulus == right_modulus
            }
            _ => false,
        }
    }

    pub(super) fn equal(&self, other: &Self) -> Option<bool> {
        if !self.compatible(other) {
            return None;
        }
        let equal = unsafe {
            match (&self.raw, &other.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(left), FlintSeriesRaw::Integer(right), _) => {
                    flint_sys::fmpz_poly_equal(left, right)
                }
                (FlintSeriesRaw::Rational(left), FlintSeriesRaw::Rational(right), _) => {
                    flint_sys::fmpq_poly_equal(left, right)
                }
                (FlintSeriesRaw::Nmod(left), FlintSeriesRaw::Nmod(right), _) => {
                    flint_sys::nmod_poly_equal(left, right)
                }
                (
                    FlintSeriesRaw::FmpzMod(left),
                    FlintSeriesRaw::FmpzMod(right),
                    FlintSeriesContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_poly_equal(left, right, ctx),
                (
                    FlintSeriesRaw::FqDefault(left),
                    FlintSeriesRaw::FqDefault(right),
                    FlintSeriesContext::FqNmod { field, .. },
                ) => flint_sys::fq_default_poly_equal(left, right, field.default_as_ptr()),
                _ => return None,
            }
        };
        Some(equal != 0)
    }

    pub(super) fn len(&self) -> usize {
        unsafe {
            match (&self.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(raw), FlintSeriesContext::Integer { .. }) => {
                    fmpz_poly_length(raw) as usize
                }
                (FlintSeriesRaw::Rational(raw), FlintSeriesContext::Rational { .. }) => {
                    fmpq_poly_length(raw) as usize
                }
                (FlintSeriesRaw::Nmod(raw), FlintSeriesContext::Nmod { .. }) => {
                    nmod_poly_length(raw) as usize
                }
                (FlintSeriesRaw::FmpzMod(raw), FlintSeriesContext::FmpzMod { raw: ctx, .. }) => {
                    fmpz_mod_poly_length(raw, ctx) as usize
                }
                (FlintSeriesRaw::FqDefault(raw), FlintSeriesContext::FqNmod { field, .. }) => {
                    fq_default_poly_length(raw, field.default_as_ptr()) as usize
                }
                _ => unreachable!("FLINT series/context storage invariant"),
            }
        }
    }

    pub(super) fn nonzero_count(&self) -> usize {
        let count = unsafe {
            match (&self.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(raw), FlintSeriesContext::Integer { .. }) => {
                    flint_sys::_fmpz_poly_hamming_weight(raw.coeffs, raw.length)
                }
                (FlintSeriesRaw::Rational(raw), FlintSeriesContext::Rational { .. }) => {
                    // fmpq_poly has one common denominator, so its nonzero
                    // coefficients are exactly its nonzero fmpz numerators.
                    flint_sys::_fmpz_poly_hamming_weight(raw.coeffs, raw.length)
                }
                (FlintSeriesRaw::Nmod(raw), FlintSeriesContext::Nmod { .. }) => {
                    flint_sys::nmod_poly_hamming_weight(raw)
                }
                (FlintSeriesRaw::FmpzMod(raw), FlintSeriesContext::FmpzMod { raw: ctx, .. }) => {
                    flint_sys::fmpz_mod_poly_hamming_weight(raw, ctx)
                }
                (FlintSeriesRaw::FqDefault(raw), FlintSeriesContext::FqNmod { field, .. }) => {
                    flint_sys::fq_default_poly_hamming_weight(raw, field.default_as_ptr())
                }
                _ => unreachable!("FLINT series/context storage invariant"),
            }
        };
        usize::try_from(count).expect("FLINT polynomial Hamming weight is nonnegative")
    }

    pub(super) fn is_zero(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn valuation<R: CommutativeRing>(&self, coefficient_ring: &R) -> Option<u64> {
        (0..self.len()).find_map(|exponent| {
            (self.coefficient(coefficient_ring, exponent as u64) != coefficient_ring.zero())
                .then_some(exponent as u64)
        })
    }

    pub(super) fn shift_right(&self, amount: u64) -> Option<Self> {
        let amount = slong::try_from(amount).ok()?;
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(output), FlintSeriesRaw::Integer(input), _) => {
                    flint_sys::fmpz_poly_shift_right(output, input, amount)
                }
                (FlintSeriesRaw::Rational(output), FlintSeriesRaw::Rational(input), _) => {
                    flint_sys::fmpq_poly_shift_right(output, input, amount)
                }
                (FlintSeriesRaw::Nmod(output), FlintSeriesRaw::Nmod(input), _) => {
                    flint_sys::nmod_poly_shift_right(output, input, amount)
                }
                (
                    FlintSeriesRaw::FmpzMod(output),
                    FlintSeriesRaw::FmpzMod(input),
                    FlintSeriesContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_poly_shift_right(output, input, amount, ctx),
                (
                    FlintSeriesRaw::FqDefault(output),
                    FlintSeriesRaw::FqDefault(input),
                    FlintSeriesContext::FqNmod { field, .. },
                ) => flint_sys::fq_default_poly_shift_right(
                    output,
                    input,
                    amount,
                    field.default_as_ptr(),
                ),
                _ => return None,
            }
        }
        Some(output)
    }

    pub(super) fn coefficient<R: CommutativeRing>(
        &self,
        coefficient_ring: &R,
        exponent: u64,
    ) -> R::Element {
        let exponent = exponent as slong;
        unsafe {
            match (&self.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(raw), FlintSeriesContext::Integer { .. }) => {
                    let mut coefficient = Integer::zero();
                    fmpz_poly_get_coeff_fmpz(coefficient.as_mut_ptr(), raw, exponent);
                    cast_owned(coefficient)
                }
                (FlintSeriesRaw::Rational(raw), FlintSeriesContext::Rational { .. }) => {
                    let mut coefficient = Rational::zero();
                    fmpq_poly_get_coeff_fmpq(coefficient.as_mut_ptr(), raw, exponent);
                    cast_owned(coefficient)
                }
                (FlintSeriesRaw::Nmod(raw), FlintSeriesContext::Nmod { .. }) => {
                    let coefficient = nmod_poly_get_coeff_ui(raw, exponent);
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        cast_owned(ring.element(coefficient))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        cast_owned(field.element(coefficient))
                    }
                }
                (FlintSeriesRaw::FmpzMod(raw), FlintSeriesContext::FmpzMod { raw: ctx, .. }) => {
                    let mut coefficient = Integer::zero();
                    fmpz_mod_poly_get_coeff_fmpz(coefficient.as_mut_ptr(), raw, exponent, ctx);
                    if is_context::<R, ResidueRing, ResidueClass>() {
                        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
                        cast_owned(ring.element(coefficient))
                    } else {
                        let field = cast_ref::<R, FiniteField>(coefficient_ring);
                        cast_owned(field.element(coefficient))
                    }
                }
                (FlintSeriesRaw::FqDefault(raw), FlintSeriesContext::FqNmod { field, .. }) => {
                    let mut coefficient = DefaultExtensionElement::initialized(Arc::clone(field));
                    fq_default_poly_get_coeff(
                        &mut coefficient.raw,
                        raw,
                        exponent,
                        field.default_as_ptr(),
                    );
                    let coefficient_ring = cast_ref::<R, FiniteField>(coefficient_ring);
                    cast_owned(coefficient_ring.wrap_extension(coefficient.to_extension()))
                }
                _ => unreachable!("FLINT series/context storage invariant"),
            }
        }
    }

    pub(super) fn to_terms<R: CommutativeRing>(
        &self,
        coefficient_ring: &R,
    ) -> BTreeMap<Vec<u64>, R::Element> {
        (0..self.len())
            .filter_map(|exponent| {
                let coefficient = self.coefficient(coefficient_ring, exponent as u64);
                (coefficient != coefficient_ring.zero())
                    .then_some((vec![exponent as u64], coefficient))
            })
            .collect()
    }

    pub(super) fn add(&self, other: &Self) -> Option<Self> {
        self.binary(other, NativeBinaryOperation::Add)
    }

    pub(super) fn sub(&self, other: &Self) -> Option<Self> {
        self.binary(other, NativeBinaryOperation::Sub)
    }

    pub(super) fn mul(&self, other: &Self) -> Option<Self> {
        self.binary(other, NativeBinaryOperation::Mul)
    }

    pub(super) fn derivative(&self, target: Arc<FlintSeriesContext>) -> Option<Self> {
        let mut output = Self::zero(target);
        unsafe {
            match (&mut output.raw, &self.raw, output.context.as_ref()) {
                (FlintSeriesRaw::Integer(output), FlintSeriesRaw::Integer(input), _) => {
                    flint_sys::fmpz_poly_derivative(output, input)
                }
                (FlintSeriesRaw::Rational(output), FlintSeriesRaw::Rational(input), _) => {
                    flint_sys::fmpq_poly_derivative(output, input)
                }
                (FlintSeriesRaw::Nmod(output), FlintSeriesRaw::Nmod(input), _) => {
                    flint_sys::nmod_poly_derivative(output, input)
                }
                (
                    FlintSeriesRaw::FmpzMod(output),
                    FlintSeriesRaw::FmpzMod(input),
                    FlintSeriesContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_poly_derivative(output, input, ctx),
                (
                    FlintSeriesRaw::FqDefault(output),
                    FlintSeriesRaw::FqDefault(input),
                    FlintSeriesContext::FqNmod { field, .. },
                ) => flint_sys::fq_default_poly_derivative(output, input, field.default_as_ptr()),
                _ => return None,
            }
        }
        Some(output)
    }

    pub(super) fn inverse(&self) -> Self {
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (
                    FlintSeriesRaw::Integer(output),
                    FlintSeriesRaw::Integer(input),
                    FlintSeriesContext::Integer { precision },
                ) => flint_sys::fmpz_poly_inv_series(output, input, *precision),
                (
                    FlintSeriesRaw::Rational(output),
                    FlintSeriesRaw::Rational(input),
                    FlintSeriesContext::Rational { precision },
                ) => flint_sys::fmpq_poly_inv_series(output, input, *precision),
                (
                    FlintSeriesRaw::Nmod(output),
                    FlintSeriesRaw::Nmod(input),
                    FlintSeriesContext::Nmod { precision, .. },
                ) => flint_sys::nmod_poly_inv_series(output, input, *precision),
                (
                    FlintSeriesRaw::FmpzMod(output),
                    FlintSeriesRaw::FmpzMod(input),
                    FlintSeriesContext::FmpzMod {
                        precision,
                        raw: ctx,
                        ..
                    },
                ) => flint_sys::fmpz_mod_poly_inv_series(output, input, *precision, ctx),
                (
                    FlintSeriesRaw::FqDefault(output),
                    FlintSeriesRaw::FqDefault(input),
                    FlintSeriesContext::FqNmod {
                        precision, field, ..
                    },
                ) => flint_sys::fq_default_poly_inv_series(
                    output,
                    input,
                    *precision,
                    field.default_as_ptr(),
                ),
                _ => unreachable!("FLINT series/context storage invariant"),
            }
        }
        output
    }

    pub(super) fn compose(&self, image: &Self) -> Option<Self> {
        if !self.compatible(image) {
            return None;
        }
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (
                &mut output.raw,
                &self.raw,
                &image.raw,
                self.context.as_ref(),
            ) {
                (
                    FlintSeriesRaw::Integer(output),
                    FlintSeriesRaw::Integer(input),
                    FlintSeriesRaw::Integer(image),
                    FlintSeriesContext::Integer { precision },
                ) => flint_sys::fmpz_poly_compose_series(output, input, image, *precision),
                (
                    FlintSeriesRaw::Rational(output),
                    FlintSeriesRaw::Rational(input),
                    FlintSeriesRaw::Rational(image),
                    FlintSeriesContext::Rational { precision },
                ) => flint_sys::fmpq_poly_compose_series(output, input, image, *precision),
                (
                    FlintSeriesRaw::Nmod(output),
                    FlintSeriesRaw::Nmod(input),
                    FlintSeriesRaw::Nmod(image),
                    FlintSeriesContext::Nmod { precision, .. },
                ) => flint_sys::nmod_poly_compose_series(output, input, image, *precision),
                (
                    FlintSeriesRaw::FmpzMod(output),
                    FlintSeriesRaw::FmpzMod(input),
                    FlintSeriesRaw::FmpzMod(image),
                    FlintSeriesContext::FmpzMod {
                        precision,
                        raw: ctx,
                        ..
                    },
                ) => {
                    // FLINT has no fmpz_mod compose-series entry point. Reduction
                    // modulo x^precision is the exact truncated composition and
                    // prevents discarded coefficients from entering the result.
                    let mut modulus = Self::zero(Arc::clone(&self.context));
                    let FlintSeriesRaw::FmpzMod(modulus) = &mut modulus.raw else {
                        unreachable!("FLINT series/context storage invariant")
                    };
                    let one = Integer::one();
                    flint_sys::fmpz_mod_poly_set_coeff_fmpz(modulus, *precision, one.as_ptr(), ctx);
                    flint_sys::fmpz_mod_poly_compose_mod(output, input, image, modulus, ctx);
                }
                (
                    FlintSeriesRaw::FqDefault(output),
                    FlintSeriesRaw::FqDefault(input),
                    FlintSeriesRaw::FqDefault(image),
                    FlintSeriesContext::FqNmod {
                        precision, field, ..
                    },
                ) => {
                    // fq_default likewise exposes modular rather than explicitly
                    // truncated composition. The internal modulus is control
                    // data, not a stored coefficient of the resulting series.
                    let mut modulus = Self::zero(Arc::clone(&self.context));
                    let FlintSeriesRaw::FqDefault(modulus) = &mut modulus.raw else {
                        unreachable!("FLINT series/context storage invariant")
                    };
                    let mut one = DefaultExtensionElement::initialized(Arc::clone(field));
                    flint_sys::fq_default_one(&mut one.raw, field.default_as_ptr());
                    flint_sys::fq_default_poly_set_coeff(
                        modulus,
                        *precision,
                        &one.raw,
                        field.default_as_ptr(),
                    );
                    flint_sys::fq_default_poly_compose_mod(
                        output,
                        input,
                        image,
                        modulus,
                        field.default_as_ptr(),
                    );
                }
                _ => return None,
            }
        }
        Some(output)
    }

    fn binary(&self, other: &Self, operation: NativeBinaryOperation) -> Option<Self> {
        if !self.compatible(other) {
            return None;
        }
        let mut output = Self::zero(Arc::clone(&self.context));
        output.assign_binary(self, other, operation);
        Some(output)
    }

    pub(super) fn sum_products(products: &[(&Self, &Self)]) -> Option<Self> {
        let (first_left, first_right) = products.first().copied()?;
        if products
            .iter()
            .any(|&(left, right)| !first_left.compatible(left) || !first_left.compatible(right))
        {
            return None;
        }

        let mut sum = Self::zero(Arc::clone(&first_left.context));
        sum.assign_binary(first_left, first_right, NativeBinaryOperation::Mul);
        if products.len() == 1 {
            return Some(sum);
        }

        let mut product = Self::zero(Arc::clone(&first_left.context));
        let mut next_sum = Self::zero(Arc::clone(&first_left.context));
        for &(left, right) in &products[1..] {
            product.assign_binary(left, right, NativeBinaryOperation::Mul);
            next_sum.assign_binary(&sum, &product, NativeBinaryOperation::Add);
            std::mem::swap(&mut sum, &mut next_sum);
        }
        Some(sum)
    }

    fn assign_binary(&mut self, left: &Self, right: &Self, operation: NativeBinaryOperation) {
        debug_assert!(self.compatible(left));
        debug_assert!(self.compatible(right));
        unsafe {
            match (
                &mut self.raw,
                &left.raw,
                &right.raw,
                self.context.as_ref(),
                operation,
            ) {
                (
                    FlintSeriesRaw::Integer(output),
                    FlintSeriesRaw::Integer(left),
                    FlintSeriesRaw::Integer(right),
                    FlintSeriesContext::Integer { precision },
                    operation,
                ) => match operation {
                    NativeBinaryOperation::Add => flint_sys::fmpz_poly_add(output, left, right),
                    NativeBinaryOperation::Sub => flint_sys::fmpz_poly_sub(output, left, right),
                    NativeBinaryOperation::Mul => fmpz_poly_mullow(output, left, right, *precision),
                },
                (
                    FlintSeriesRaw::Rational(output),
                    FlintSeriesRaw::Rational(left),
                    FlintSeriesRaw::Rational(right),
                    FlintSeriesContext::Rational { precision },
                    operation,
                ) => match operation {
                    NativeBinaryOperation::Add => flint_sys::fmpq_poly_add(output, left, right),
                    NativeBinaryOperation::Sub => flint_sys::fmpq_poly_sub(output, left, right),
                    NativeBinaryOperation::Mul => fmpq_poly_mullow(output, left, right, *precision),
                },
                (
                    FlintSeriesRaw::Nmod(output),
                    FlintSeriesRaw::Nmod(left),
                    FlintSeriesRaw::Nmod(right),
                    FlintSeriesContext::Nmod { precision, .. },
                    operation,
                ) => match operation {
                    NativeBinaryOperation::Add => flint_sys::nmod_poly_add(output, left, right),
                    NativeBinaryOperation::Sub => flint_sys::nmod_poly_sub(output, left, right),
                    NativeBinaryOperation::Mul => nmod_poly_mullow(output, left, right, *precision),
                },
                (
                    FlintSeriesRaw::FmpzMod(output),
                    FlintSeriesRaw::FmpzMod(left),
                    FlintSeriesRaw::FmpzMod(right),
                    FlintSeriesContext::FmpzMod {
                        precision,
                        raw: ctx,
                        ..
                    },
                    operation,
                ) => match operation {
                    NativeBinaryOperation::Add => {
                        flint_sys::fmpz_mod_poly_add(output, left, right, ctx)
                    }
                    NativeBinaryOperation::Sub => {
                        flint_sys::fmpz_mod_poly_sub(output, left, right, ctx)
                    }
                    NativeBinaryOperation::Mul => {
                        fmpz_mod_poly_mullow(output, left, right, *precision, ctx)
                    }
                },
                (
                    FlintSeriesRaw::FqDefault(output),
                    FlintSeriesRaw::FqDefault(left),
                    FlintSeriesRaw::FqDefault(right),
                    FlintSeriesContext::FqNmod {
                        precision, field, ..
                    },
                    operation,
                ) => match operation {
                    NativeBinaryOperation::Add => {
                        flint_sys::fq_default_poly_add(output, left, right, field.default_as_ptr())
                    }
                    NativeBinaryOperation::Sub => {
                        flint_sys::fq_default_poly_sub(output, left, right, field.default_as_ptr())
                    }
                    NativeBinaryOperation::Mul => fq_default_poly_mullow(
                        output,
                        left,
                        right,
                        *precision,
                        field.default_as_ptr(),
                    ),
                },
                _ => unreachable!("FLINT series/context storage invariant"),
            }
        }
    }

    pub(super) fn neg(&self) -> Self {
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(output), FlintSeriesRaw::Integer(input), _) => {
                    flint_sys::fmpz_poly_neg(output, input)
                }
                (FlintSeriesRaw::Rational(output), FlintSeriesRaw::Rational(input), _) => {
                    flint_sys::fmpq_poly_neg(output, input)
                }
                (FlintSeriesRaw::Nmod(output), FlintSeriesRaw::Nmod(input), _) => {
                    flint_sys::nmod_poly_neg(output, input)
                }
                (
                    FlintSeriesRaw::FmpzMod(output),
                    FlintSeriesRaw::FmpzMod(input),
                    FlintSeriesContext::FmpzMod { raw: ctx, .. },
                ) => flint_sys::fmpz_mod_poly_neg(output, input, ctx),
                (
                    FlintSeriesRaw::FqDefault(output),
                    FlintSeriesRaw::FqDefault(input),
                    FlintSeriesContext::FqNmod { field, .. },
                ) => flint_sys::fq_default_poly_neg(output, input, field.default_as_ptr()),
                _ => unreachable!("FLINT series/context storage invariant"),
            }
        }
        output
    }

    pub(super) fn scalar_mul<R: CommutativeRing>(&self, scalar: &R::Element) -> Option<Self> {
        let mut output = Self::zero(Arc::clone(&self.context));
        unsafe {
            match (&mut output.raw, &self.raw, self.context.as_ref()) {
                (FlintSeriesRaw::Integer(output), FlintSeriesRaw::Integer(input), _) => {
                    flint_sys::fmpz_poly_scalar_mul_fmpz(
                        output,
                        input,
                        cast_ref::<_, Integer>(scalar).as_ptr(),
                    )
                }
                (FlintSeriesRaw::Rational(output), FlintSeriesRaw::Rational(input), _) => {
                    flint_sys::fmpq_poly_scalar_mul_fmpq(
                        output,
                        input,
                        cast_ref::<_, Rational>(scalar).as_ptr(),
                    )
                }
                (
                    FlintSeriesRaw::Nmod(output),
                    FlintSeriesRaw::Nmod(input),
                    FlintSeriesContext::Nmod { modulus, .. },
                ) => {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(scalar).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(scalar).representative()
                    };
                    flint_sys::nmod_poly_scalar_mul_nmod(
                        output,
                        input,
                        representative.modulo_u64(*modulus),
                    );
                }
                (
                    FlintSeriesRaw::FmpzMod(output),
                    FlintSeriesRaw::FmpzMod(input),
                    FlintSeriesContext::FmpzMod { raw: ctx, .. },
                ) => {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(scalar).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(scalar).representative()
                    };
                    flint_sys::fmpz_mod_poly_scalar_mul_fmpz(
                        output,
                        input,
                        representative.as_ptr(),
                        ctx,
                    );
                }
                (
                    FlintSeriesRaw::FqDefault(output),
                    FlintSeriesRaw::FqDefault(input),
                    FlintSeriesContext::FqNmod { field, .. },
                ) => {
                    let coefficient = cast_ref::<_, FiniteFieldElement>(scalar)
                        .extension_value()
                        .expect("extension-field series coefficient invariant")
                        .to_default();
                    flint_sys::fq_default_poly_scalar_mul_fq_default(
                        output,
                        input,
                        &coefficient.raw,
                        field.default_as_ptr(),
                    );
                }
                _ => return None,
            }
        }
        Some(output)
    }
}

pub(super) fn series_from_terms<R: CommutativeRing>(
    context: Arc<FlintSeriesContext>,
    terms: &BTreeMap<Vec<u64>, R::Element>,
) -> Option<FlintSeries> {
    let mut series = FlintSeries::zero(context);
    unsafe {
        match (&mut series.raw, series.context.as_ref()) {
            (FlintSeriesRaw::Integer(raw), FlintSeriesContext::Integer { .. }) => {
                for (exponent, coefficient) in terms {
                    fmpz_poly_set_coeff_fmpz(
                        raw,
                        exponent[0] as slong,
                        cast_ref::<_, Integer>(coefficient).as_ptr(),
                    );
                }
            }
            (FlintSeriesRaw::Rational(raw), FlintSeriesContext::Rational { .. }) => {
                for (exponent, coefficient) in terms {
                    fmpq_poly_set_coeff_fmpq(
                        raw,
                        exponent[0] as slong,
                        cast_ref::<_, Rational>(coefficient).as_ptr(),
                    );
                }
            }
            (FlintSeriesRaw::Nmod(raw), FlintSeriesContext::Nmod { modulus, .. }) => {
                for (exponent, coefficient) in terms {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(coefficient).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(coefficient).representative()
                    };
                    nmod_poly_set_coeff_ui(
                        raw,
                        exponent[0] as slong,
                        representative.modulo_u64(*modulus),
                    );
                }
            }
            (FlintSeriesRaw::FmpzMod(raw), FlintSeriesContext::FmpzMod { raw: ctx, .. }) => {
                for (exponent, coefficient) in terms {
                    let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                        cast_ref::<_, ResidueClass>(coefficient).representative()
                    } else {
                        cast_ref::<_, FiniteFieldElement>(coefficient).representative()
                    };
                    fmpz_mod_poly_set_coeff_fmpz(
                        raw,
                        exponent[0] as slong,
                        representative.as_ptr(),
                        ctx,
                    );
                }
            }
            (FlintSeriesRaw::FqDefault(raw), FlintSeriesContext::FqNmod { field, .. }) => {
                for (exponent, coefficient) in terms {
                    let coefficient = cast_ref::<_, FiniteFieldElement>(coefficient)
                        .extension_value()
                        .expect("extension-field series coefficient invariant")
                        .to_default();
                    fq_default_poly_set_coeff(
                        raw,
                        exponent[0] as slong,
                        &coefficient.raw,
                        field.default_as_ptr(),
                    );
                }
            }
            _ => return None,
        }
    }
    Some(series)
}

#[cfg(test)]
fn multiply_with_crossover<R: CommutativeRing>(
    ring: &PolynomialRing<R>,
    left: &BTreeMap<Vec<Integer>, R::Element>,
    right: &BTreeMap<Vec<Integer>, R::Element>,
    enforce_crossover: bool,
) -> Option<BTreeMap<Vec<Integer>, R::Element>> {
    if left.is_empty() || right.is_empty() {
        return Some(BTreeMap::new());
    }
    let work = left.len().saturating_mul(right.len());
    let crossover = if is_context::<R, IntegerRing, Integer>()
        || is_context::<R, RationalField, Rational>()
        || is_context::<R, ResidueRing, ResidueClass>()
    {
        FLINT_SCALAR_TERM_PRODUCT_CROSSOVER
    } else if is_context::<R, FiniteField, FiniteFieldElement>() {
        let field = cast_ref::<R, FiniteField>(ring.coefficient_ring());
        if field.degree() == 1 {
            FLINT_SCALAR_TERM_PRODUCT_CROSSOVER
        } else {
            FLINT_EXTENSION_TERM_PRODUCT_CROSSOVER
        }
    } else {
        return None;
    };
    if enforce_crossover && work < crossover {
        return None;
    }
    let nvars = slong::try_from(ring.variables().len()).ok()?;
    let left = machine_terms(left)?;
    let right = machine_terms(right)?;
    if !product_exponents_fit(&left, &right, ring.term_order()) {
        return None;
    }
    let order = order(&ring.term_order())?;

    if is_context::<R, IntegerRing, Integer>() {
        let left = typed_terms::<_, Integer>(&left);
        let right = typed_terms::<_, Integer>(&right);
        return Some(cast_terms(multiply_fmpz(nvars, order, &left, &right)));
    }
    if is_context::<R, RationalField, Rational>() {
        let left = typed_terms::<_, Rational>(&left);
        let right = typed_terms::<_, Rational>(&right);
        return Some(cast_terms(multiply_fmpq(nvars, order, &left, &right)));
    }
    if is_context::<R, ResidueRing, ResidueClass>() {
        let residues = cast_ref::<R, ResidueRing>(ring.coefficient_ring());
        let typed_left = typed_terms::<_, ResidueClass>(&left);
        let typed_right = typed_terms::<_, ResidueClass>(&right);
        let output = if let Some(modulus) = residues.modulus().to_u64_checked() {
            let left = typed_left
                .iter()
                .map(|(exponent, coefficient)| {
                    (
                        exponent.clone(),
                        coefficient.representative().modulo_u64(modulus),
                    )
                })
                .collect::<Vec<_>>();
            let right = typed_right
                .iter()
                .map(|(exponent, coefficient)| {
                    (
                        exponent.clone(),
                        coefficient.representative().modulo_u64(modulus),
                    )
                })
                .collect::<Vec<_>>();
            multiply_nmod(nvars, order, modulus, &left, &right)
                .into_iter()
                .map(|(exponent, coefficient)| (exponent, residues.element(coefficient)))
                .collect()
        } else {
            let left = typed_left
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient.representative()))
                .collect::<Vec<_>>();
            let right = typed_right
                .iter()
                .map(|(exponent, coefficient)| (exponent.clone(), coefficient.representative()))
                .collect::<Vec<_>>();
            multiply_fmpz_mod(nvars, order, residues.modulus(), &left, &right)
                .into_iter()
                .map(|(exponent, coefficient)| (exponent, residues.element(coefficient)))
                .collect()
        };
        return Some(cast_terms(output));
    }
    if is_context::<R, FiniteField, FiniteFieldElement>() {
        let field = cast_ref::<R, FiniteField>(ring.coefficient_ring());
        let typed_left = typed_terms::<_, FiniteFieldElement>(&left);
        let typed_right = typed_terms::<_, FiniteFieldElement>(&right);
        let output = if field.degree() == 1 {
            if let Some(modulus) = field.word_characteristic() {
                let left = typed_left
                    .iter()
                    .map(|(exponent, coefficient)| {
                        (
                            exponent.clone(),
                            coefficient.representative().modulo_u64(modulus),
                        )
                    })
                    .collect::<Vec<_>>();
                let right = typed_right
                    .iter()
                    .map(|(exponent, coefficient)| {
                        (
                            exponent.clone(),
                            coefficient.representative().modulo_u64(modulus),
                        )
                    })
                    .collect::<Vec<_>>();
                multiply_nmod(nvars, order, modulus, &left, &right)
                    .into_iter()
                    .map(|(exponent, coefficient)| (exponent, field.element(coefficient)))
                    .collect()
            } else {
                let left = typed_left
                    .iter()
                    .map(|(exponent, coefficient)| (exponent.clone(), coefficient.representative()))
                    .collect::<Vec<_>>();
                let right = typed_right
                    .iter()
                    .map(|(exponent, coefficient)| (exponent.clone(), coefficient.representative()))
                    .collect::<Vec<_>>();
                multiply_fmpz_mod(nvars, order, field.characteristic(), &left, &right)
                    .into_iter()
                    .map(|(exponent, coefficient)| (exponent, field.element(coefficient)))
                    .collect()
            }
        } else {
            multiply_fq_nmod(nvars, order, field, &typed_left, &typed_right)
        };
        return Some(cast_terms(output));
    }
    None
}

#[cfg(test)]
pub(super) fn multiply_univariate_series<R: CommutativeRing>(
    coefficient_ring: &R,
    left: &BTreeMap<Vec<u64>, R::Element>,
    right: &BTreeMap<Vec<u64>, R::Element>,
    precision: u64,
) -> Option<BTreeMap<Vec<u64>, R::Element>> {
    multiply_univariate_series_with_crossover(coefficient_ring, left, right, precision, true)
}

#[cfg(test)]
fn multiply_univariate_series_with_crossover<R: CommutativeRing>(
    coefficient_ring: &R,
    left: &BTreeMap<Vec<u64>, R::Element>,
    right: &BTreeMap<Vec<u64>, R::Element>,
    precision: u64,
    enforce_crossover: bool,
) -> Option<BTreeMap<Vec<u64>, R::Element>> {
    if left.is_empty() || right.is_empty() {
        return Some(BTreeMap::new());
    }
    let work = left.len().saturating_mul(right.len());
    let precision = slong::try_from(precision).ok()?;
    if is_context::<R, IntegerRing, Integer>() {
        if enforce_crossover && work < FLINT_SERIES_SCALAR_TERM_PRODUCT_CROSSOVER {
            return None;
        }
        let left = left
            .iter()
            .map(|(exponent, coefficient)| (exponent[0], cast_ref::<_, Integer>(coefficient)))
            .collect::<Vec<_>>();
        let right = right
            .iter()
            .map(|(exponent, coefficient)| (exponent[0], cast_ref::<_, Integer>(coefficient)))
            .collect::<Vec<_>>();
        return Some(cast_series_terms(multiply_fmpz_series(
            &left, &right, precision,
        )));
    }
    if is_context::<R, RationalField, Rational>() {
        if enforce_crossover && work < FLINT_SERIES_SCALAR_TERM_PRODUCT_CROSSOVER {
            return None;
        }
        let left = left
            .iter()
            .map(|(exponent, coefficient)| (exponent[0], cast_ref::<_, Rational>(coefficient)))
            .collect::<Vec<_>>();
        let right = right
            .iter()
            .map(|(exponent, coefficient)| (exponent[0], cast_ref::<_, Rational>(coefficient)))
            .collect::<Vec<_>>();
        return Some(cast_series_terms(multiply_fmpq_series(
            &left, &right, precision,
        )));
    }

    if is_context::<R, ResidueRing, ResidueClass>() {
        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
        if ring.modulus().to_u64_checked().is_none() {
            if enforce_crossover && work < FLINT_SERIES_BIG_MODULAR_TERM_PRODUCT_CROSSOVER {
                return None;
            }
            let left = left
                .iter()
                .map(|(exponent, coefficient)| {
                    (
                        exponent[0],
                        cast_ref::<_, ResidueClass>(coefficient).representative(),
                    )
                })
                .collect::<Vec<_>>();
            let right = right
                .iter()
                .map(|(exponent, coefficient)| {
                    (
                        exponent[0],
                        cast_ref::<_, ResidueClass>(coefficient).representative(),
                    )
                })
                .collect::<Vec<_>>();
            return Some(cast_series_terms(
                multiply_fmpz_mod_series(ring.modulus(), &left, &right, precision)
                    .into_iter()
                    .map(|(exponent, coefficient)| (exponent, ring.element(coefficient)))
                    .collect(),
            ));
        }
    }
    if is_context::<R, FiniteField, FiniteFieldElement>() {
        let field = cast_ref::<R, FiniteField>(coefficient_ring);
        if field.degree() > 1 {
            if enforce_crossover && work < FLINT_SERIES_EXTENSION_TERM_PRODUCT_CROSSOVER {
                return None;
            }
            let left = left
                .iter()
                .map(|(exponent, coefficient)| {
                    (exponent[0], cast_ref::<_, FiniteFieldElement>(coefficient))
                })
                .collect::<Vec<_>>();
            let right = right
                .iter()
                .map(|(exponent, coefficient)| {
                    (exponent[0], cast_ref::<_, FiniteFieldElement>(coefficient))
                })
                .collect::<Vec<_>>();
            return Some(cast_series_terms(multiply_fq_nmod_series(
                field, &left, &right, precision,
            )));
        }
    }

    let modulus = if is_context::<R, ResidueRing, ResidueClass>() {
        cast_ref::<R, ResidueRing>(coefficient_ring)
            .modulus()
            .to_u64_checked()
    } else if is_context::<R, FiniteField, FiniteFieldElement>() {
        let field = cast_ref::<R, FiniteField>(coefficient_ring);
        (field.degree() == 1)
            .then(|| field.word_characteristic())
            .flatten()
    } else {
        None
    }?;
    if enforce_crossover && work < FLINT_SERIES_MODULAR_TERM_PRODUCT_CROSSOVER {
        return None;
    }
    let left = left
        .iter()
        .map(|(exponent, coefficient)| {
            let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                cast_ref::<_, ResidueClass>(coefficient).representative()
            } else {
                cast_ref::<_, FiniteFieldElement>(coefficient).representative()
            };
            (exponent[0], representative.modulo_u64(modulus))
        })
        .collect::<Vec<_>>();
    let right = right
        .iter()
        .map(|(exponent, coefficient)| {
            let representative = if is_context::<R, ResidueRing, ResidueClass>() {
                cast_ref::<_, ResidueClass>(coefficient).representative()
            } else {
                cast_ref::<_, FiniteFieldElement>(coefficient).representative()
            };
            (exponent[0], representative.modulo_u64(modulus))
        })
        .collect::<Vec<_>>();
    let values = multiply_nmod_series(modulus, &left, &right, precision);
    if is_context::<R, ResidueRing, ResidueClass>() {
        let ring = cast_ref::<R, ResidueRing>(coefficient_ring);
        Some(cast_series_terms(
            values
                .into_iter()
                .map(|(exponent, coefficient)| (exponent, ring.element(coefficient)))
                .collect(),
        ))
    } else {
        let field = cast_ref::<R, FiniteField>(coefficient_ring);
        Some(cast_series_terms(
            values
                .into_iter()
                .map(|(exponent, coefficient)| (exponent, field.element(coefficient)))
                .collect(),
        ))
    }
}

#[cfg(test)]
fn cast_series_terms<T: Any, E: Any>(terms: Vec<(u64, T)>) -> BTreeMap<Vec<u64>, E> {
    terms
        .into_iter()
        .map(|(exponent, coefficient)| (vec![exponent], cast_owned(coefficient)))
        .collect()
}

pub(super) struct DenseFactorization<E> {
    pub(super) coefficient: E,
    pub(super) factors: Vec<(Vec<E>, u64)>,
}

pub(super) struct NativeFactorization<E> {
    pub(super) coefficient: E,
    pub(super) factors: Vec<(FlintPolynomial, u64)>,
}

impl FlintPolynomial {
    pub(super) fn factor_integer_multivariate(
        &self,
    ) -> Result<NativeFactorization<Integer>, &'static str> {
        let (input, context) = match (&self.raw, self.context.as_ref()) {
            (
                FlintPolynomialRaw::Integer(input),
                FlintPolynomialContext::Integer { raw: context, .. },
            ) => (input, context),
            _ => return Err("the native polynomial is not integral"),
        };
        let mut factorization = OwnedMpolyFactor::new(
            context,
            flint_sys::fmpz_mpoly_factor_init,
            flint_sys::fmpz_mpoly_factor_clear,
        );
        if unsafe { flint_sys::fmpz_mpoly_factor(factorization.as_mut_ptr(), input, context) } == 0
        {
            return Err("FLINT failed to factor an integral multivariate polynomial");
        }
        let mut coefficient = Integer::zero();
        unsafe {
            flint_sys::fmpz_mpoly_factor_get_constant_fmpz(
                coefficient.as_mut_ptr(),
                factorization.as_ptr(),
                context,
            )
        };
        let count = unsafe { flint_sys::fmpz_mpoly_factor_length(factorization.as_ptr(), context) };
        let count = usize::try_from(count).map_err(|_| "FLINT returned a negative factor count")?;
        let mut factors = Vec::with_capacity(count);
        for index in 0..count {
            let index =
                slong::try_from(index).map_err(|_| "the factor index does not fit slong")?;
            let mut factor = Self::zero(Arc::clone(&self.context));
            let FlintPolynomialRaw::Integer(output) = &mut factor.raw else {
                unreachable!("factor storage inherits its integral context")
            };
            unsafe {
                flint_sys::fmpz_mpoly_factor_get_base(
                    output,
                    factorization.as_ptr(),
                    index,
                    context,
                )
            };
            let exponent = unsafe {
                flint_sys::fmpz_mpoly_factor_get_exp_si(factorization.as_mut_ptr(), index, context)
            };
            factors.push((factor, positive_factor_exponent(exponent)?));
        }
        Ok(NativeFactorization {
            coefficient,
            factors,
        })
    }

    pub(super) fn factor_rational_multivariate(
        &self,
    ) -> Result<NativeFactorization<Rational>, &'static str> {
        let (input, context) = match (&self.raw, self.context.as_ref()) {
            (
                FlintPolynomialRaw::Rational(input),
                FlintPolynomialContext::Rational { raw: context, .. },
            ) => (input, context),
            _ => return Err("the native polynomial is not rational"),
        };
        let mut factorization = OwnedMpolyFactor::new(
            context,
            flint_sys::fmpq_mpoly_factor_init,
            flint_sys::fmpq_mpoly_factor_clear,
        );
        if unsafe { flint_sys::fmpq_mpoly_factor(factorization.as_mut_ptr(), input, context) } == 0
        {
            return Err("FLINT failed to factor a rational multivariate polynomial");
        }
        let mut coefficient = Rational::zero();
        unsafe {
            flint_sys::fmpq_mpoly_factor_get_constant_fmpq(
                coefficient.as_mut_ptr(),
                factorization.as_ptr(),
                context,
            )
        };
        let count = unsafe { flint_sys::fmpq_mpoly_factor_length(factorization.as_ptr(), context) };
        let count = usize::try_from(count).map_err(|_| "FLINT returned a negative factor count")?;
        let mut factors = Vec::with_capacity(count);
        for index in 0..count {
            let index =
                slong::try_from(index).map_err(|_| "the factor index does not fit slong")?;
            let mut factor = Self::zero(Arc::clone(&self.context));
            let FlintPolynomialRaw::Rational(output) = &mut factor.raw else {
                unreachable!("factor storage inherits its rational context")
            };
            unsafe {
                flint_sys::fmpq_mpoly_factor_get_base(
                    output,
                    factorization.as_ptr(),
                    index,
                    context,
                )
            };
            let exponent = unsafe {
                flint_sys::fmpq_mpoly_factor_get_exp_si(factorization.as_mut_ptr(), index, context)
            };
            factors.push((factor, positive_factor_exponent(exponent)?));
        }
        Ok(NativeFactorization {
            coefficient,
            factors,
        })
    }

    pub(super) fn factor_finite_field_multivariate(
        &self,
        field: &FiniteField,
    ) -> Result<NativeFactorization<FiniteFieldElement>, &'static str> {
        match (&self.raw, self.context.as_ref()) {
            (
                FlintPolynomialRaw::Nmod(input),
                FlintPolynomialContext::Nmod { raw: context, .. },
            ) => {
                let mut factorization = OwnedMpolyFactor::new(
                    context,
                    flint_sys::nmod_mpoly_factor_init,
                    flint_sys::nmod_mpoly_factor_clear,
                );
                if unsafe {
                    flint_sys::nmod_mpoly_factor(factorization.as_mut_ptr(), input, context)
                } == 0
                {
                    return Err("FLINT failed to factor a word-prime multivariate polynomial");
                }
                let coefficient = field.element(unsafe {
                    flint_sys::nmod_mpoly_factor_get_constant_ui(factorization.as_ptr(), context)
                });
                let count =
                    unsafe { flint_sys::nmod_mpoly_factor_length(factorization.as_ptr(), context) };
                let count =
                    usize::try_from(count).map_err(|_| "FLINT returned a negative factor count")?;
                let mut factors = Vec::with_capacity(count);
                for index in 0..count {
                    let index = slong::try_from(index)
                        .map_err(|_| "the factor index does not fit slong")?;
                    let mut factor = Self::zero(Arc::clone(&self.context));
                    let FlintPolynomialRaw::Nmod(output) = &mut factor.raw else {
                        unreachable!("factor storage inherits its word-prime context")
                    };
                    unsafe {
                        flint_sys::nmod_mpoly_factor_get_base(
                            output,
                            factorization.as_ptr(),
                            index,
                            context,
                        )
                    };
                    let exponent = unsafe {
                        flint_sys::nmod_mpoly_factor_get_exp_si(
                            factorization.as_mut_ptr(),
                            index,
                            context,
                        )
                    };
                    factors.push((factor, positive_factor_exponent(exponent)?));
                }
                Ok(NativeFactorization {
                    coefficient,
                    factors,
                })
            }
            (
                FlintPolynomialRaw::FmpzMod(input),
                FlintPolynomialContext::FmpzMod { raw: context, .. },
            ) => {
                let mut factorization = OwnedMpolyFactor::new(
                    context,
                    flint_sys::fmpz_mod_mpoly_factor_init,
                    flint_sys::fmpz_mod_mpoly_factor_clear,
                );
                if unsafe {
                    flint_sys::fmpz_mod_mpoly_factor(factorization.as_mut_ptr(), input, context)
                } == 0
                {
                    return Err("FLINT failed to factor a large-prime multivariate polynomial");
                }
                let mut representative = Integer::zero();
                unsafe {
                    flint_sys::fmpz_mod_mpoly_factor_get_constant_fmpz(
                        representative.as_mut_ptr(),
                        factorization.as_ptr(),
                        context,
                    )
                };
                let coefficient = field.element(representative);
                let count = unsafe {
                    flint_sys::fmpz_mod_mpoly_factor_length(factorization.as_ptr(), context)
                };
                let count =
                    usize::try_from(count).map_err(|_| "FLINT returned a negative factor count")?;
                let mut factors = Vec::with_capacity(count);
                for index in 0..count {
                    let index = slong::try_from(index)
                        .map_err(|_| "the factor index does not fit slong")?;
                    let mut factor = Self::zero(Arc::clone(&self.context));
                    let FlintPolynomialRaw::FmpzMod(output) = &mut factor.raw else {
                        unreachable!("factor storage inherits its large-prime context")
                    };
                    unsafe {
                        flint_sys::fmpz_mod_mpoly_factor_get_base(
                            output,
                            factorization.as_ptr(),
                            index,
                            context,
                        )
                    };
                    let exponent = unsafe {
                        flint_sys::fmpz_mod_mpoly_factor_get_exp_si(
                            factorization.as_mut_ptr(),
                            index,
                            context,
                        )
                    };
                    factors.push((factor, positive_factor_exponent(exponent)?));
                }
                Ok(NativeFactorization {
                    coefficient,
                    factors,
                })
            }
            (
                FlintPolynomialRaw::FqNmod(input),
                FlintPolynomialContext::FqNmod {
                    raw: context,
                    field: extension,
                    ..
                },
            ) => {
                let mut factorization = OwnedMpolyFactor::new(
                    context,
                    flint_sys::fq_nmod_mpoly_factor_init,
                    flint_sys::fq_nmod_mpoly_factor_clear,
                );
                if unsafe {
                    flint_sys::fq_nmod_mpoly_factor(factorization.as_mut_ptr(), input, context)
                } == 0
                {
                    return Err(
                        "FLINT failed to factor an extension-field multivariate polynomial",
                    );
                }
                let mut constant = ExtensionElement::initialized(Arc::clone(extension));
                unsafe {
                    flint_sys::fq_nmod_mpoly_factor_get_constant_fq_nmod(
                        &mut constant.raw,
                        factorization.as_ptr(),
                        context,
                    )
                };
                let coefficient = field.wrap_extension(constant);
                let count = unsafe {
                    flint_sys::fq_nmod_mpoly_factor_length(factorization.as_ptr(), context)
                };
                let count =
                    usize::try_from(count).map_err(|_| "FLINT returned a negative factor count")?;
                let mut factors = Vec::with_capacity(count);
                for index in 0..count {
                    let index = slong::try_from(index)
                        .map_err(|_| "the factor index does not fit slong")?;
                    let mut factor = Self::zero(Arc::clone(&self.context));
                    let FlintPolynomialRaw::FqNmod(output) = &mut factor.raw else {
                        unreachable!("factor storage inherits its extension-field context")
                    };
                    unsafe {
                        flint_sys::fq_nmod_mpoly_factor_get_base(
                            output,
                            factorization.as_ptr(),
                            index,
                            context,
                        )
                    };
                    let exponent = unsafe {
                        flint_sys::fq_nmod_mpoly_factor_get_exp_si(
                            factorization.as_mut_ptr(),
                            index,
                            context,
                        )
                    };
                    factors.push((factor, positive_factor_exponent(exponent)?));
                }
                Ok(NativeFactorization {
                    coefficient,
                    factors,
                })
            }
            _ => Err("the native polynomial has no finite-field factorization context"),
        }
    }
}

fn positive_factor_exponent(exponent: slong) -> Result<u64, &'static str> {
    u64::try_from(exponent)
        .ok()
        .filter(|value| *value > 0)
        .ok_or("FLINT returned a nonpositive factor exponent")
}

pub(super) fn factor_integer_dense(coefficients: &[Integer]) -> DenseFactorization<Integer> {
    let mut input = OwnedUnivariate::new(|raw| unsafe { fmpz_poly_init(raw) }, fmpz_poly_clear);
    for (exponent, coefficient) in coefficients.iter().enumerate() {
        unsafe {
            fmpz_poly_set_coeff_fmpz(input.as_mut_ptr(), exponent as slong, coefficient.as_ptr())
        };
    }
    let mut factorization = OwnedFmpzPolynomialFactor::new();
    unsafe { fmpz_poly_factor(factorization.as_mut_ptr(), input.as_ptr()) };
    let coefficient = unsafe { Integer::clone_from_ptr(&factorization.raw.c) };
    let count = usize::try_from(factorization.raw.num)
        .expect("FLINT polynomial factor count is nonnegative");
    let mut factors = Vec::with_capacity(count);
    for index in 0..count {
        let mut factor =
            OwnedUnivariate::new(|raw| unsafe { fmpz_poly_init(raw) }, fmpz_poly_clear);
        unsafe {
            fmpz_poly_factor_get_fmpz_poly(
                factor.as_mut_ptr(),
                factorization.as_ptr(),
                index as slong,
            )
        };
        factors.push((dense_fmpz_polynomial(factor.as_ptr()), unsafe {
            *factorization.raw.exp.add(index) as u64
        }));
    }
    DenseFactorization {
        coefficient,
        factors,
    }
}

pub(super) fn factor_finite_field_dense(
    field: &FiniteField,
    coefficients: &[FiniteFieldElement],
) -> DenseFactorization<FiniteFieldElement> {
    if let Some(context) = field.extension_context() {
        return factor_extension_field_dense(field, context, coefficients);
    }
    if let Some(modulus) = field.word_characteristic() {
        return factor_word_prime_field_dense(field, modulus, coefficients);
    }
    factor_big_prime_field_dense(field, coefficients)
}

fn factor_word_prime_field_dense(
    field: &FiniteField,
    modulus: u64,
    coefficients: &[FiniteFieldElement],
) -> DenseFactorization<FiniteFieldElement> {
    let mut input = OwnedUnivariate::new(
        |raw| unsafe { nmod_poly_init(raw, modulus) },
        nmod_poly_clear,
    );
    for (exponent, coefficient) in coefficients.iter().enumerate() {
        unsafe {
            nmod_poly_set_coeff_ui(
                input.as_mut_ptr(),
                exponent as slong,
                coefficient.representative().modulo_u64(modulus),
            )
        };
    }
    let mut factorization = OwnedNmodPolynomialFactor::new();
    let leading = unsafe { nmod_poly_factor(factorization.as_mut_ptr(), input.as_ptr()) };
    let count = usize::try_from(factorization.raw.num)
        .expect("FLINT polynomial factor count is nonnegative");
    let mut factors = Vec::with_capacity(count);
    for index in 0..count {
        let mut factor = OwnedUnivariate::new(
            |raw| unsafe { nmod_poly_init(raw, modulus) },
            nmod_poly_clear,
        );
        unsafe {
            nmod_poly_factor_get_poly(factor.as_mut_ptr(), factorization.as_ptr(), index as slong)
        };
        factors.push((dense_nmod_polynomial(field, factor.as_ptr()), unsafe {
            *factorization.raw.exp.add(index) as u64
        }));
    }
    DenseFactorization {
        coefficient: field.element(leading),
        factors,
    }
}

fn factor_big_prime_field_dense(
    field: &FiniteField,
    coefficients: &[FiniteFieldElement],
) -> DenseFactorization<FiniteFieldElement> {
    let context = OwnedContext::<fmpz_mod_ctx_struct>::new(
        |raw| unsafe { fmpz_mod_ctx_init(raw, field.characteristic().as_ptr()) },
        fmpz_mod_ctx_clear,
    );
    let mut input = OwnedPolynomial::new(&context, fmpz_mod_poly_init, fmpz_mod_poly_clear);
    for (exponent, coefficient) in coefficients.iter().enumerate() {
        unsafe {
            fmpz_mod_poly_set_coeff_fmpz(
                input.as_mut_ptr(),
                exponent as slong,
                coefficient.representative().as_ptr(),
                context.as_ptr(),
            )
        };
    }
    let mut factorization = OwnedFmpzModPolynomialFactor::new(&context);
    unsafe { fmpz_mod_poly_factor(factorization.as_mut_ptr(), input.as_ptr(), context.as_ptr()) };
    let count = usize::try_from(factorization.raw.num)
        .expect("FLINT polynomial factor count is nonnegative");
    let mut factors = Vec::with_capacity(count);
    for index in 0..count {
        let mut factor = OwnedPolynomial::new(&context, fmpz_mod_poly_init, fmpz_mod_poly_clear);
        unsafe {
            fmpz_mod_poly_factor_get_poly(
                factor.as_mut_ptr(),
                factorization.as_ptr(),
                index as slong,
                context.as_ptr(),
            )
        };
        factors.push((
            dense_fmpz_mod_polynomial(field, &context, factor.as_ptr()),
            unsafe { *factorization.raw.exp.add(index) as u64 },
        ));
    }
    DenseFactorization {
        coefficient: coefficients
            .last()
            .expect("a factored polynomial is nonzero")
            .clone(),
        factors,
    }
}

fn factor_extension_field_dense(
    field: &FiniteField,
    context: Arc<ExtensionContext>,
    coefficients: &[FiniteFieldElement],
) -> DenseFactorization<FiniteFieldElement> {
    let mut input = OwnedFqDefaultPolynomial::new(Arc::clone(&context));
    for (exponent, coefficient) in coefficients.iter().enumerate() {
        let coefficient = coefficient
            .extension_value()
            .expect("extension-field coefficient belongs to the selected context")
            .to_default();
        unsafe {
            fq_default_poly_set_coeff(
                input.as_mut_ptr(),
                exponent as slong,
                &coefficient.raw,
                context.default_as_ptr(),
            )
        };
    }
    let mut factorization = OwnedFqDefaultPolynomialFactor::new(Arc::clone(&context));
    let mut leading = DefaultExtensionElement::initialized(Arc::clone(&context));
    unsafe {
        fq_default_poly_factor(
            factorization.as_mut_ptr(),
            &mut leading.raw,
            input.as_ptr(),
            context.default_as_ptr(),
        )
    };
    let count = usize::try_from(unsafe {
        fq_default_poly_factor_length(factorization.as_mut_ptr(), context.default_as_ptr())
    })
    .expect("FLINT polynomial factor count is nonnegative");
    let mut factors = Vec::with_capacity(count);
    for index in 0..count {
        let mut factor = OwnedFqDefaultPolynomial::new(Arc::clone(&context));
        unsafe {
            fq_default_poly_factor_get_poly(
                factor.as_mut_ptr(),
                factorization.as_ptr(),
                index as slong,
                context.default_as_ptr(),
            )
        };
        factors.push((
            dense_fq_default_polynomial(field, &context, factor.as_ptr()),
            unsafe {
                fq_default_poly_factor_exp(
                    factorization.as_mut_ptr(),
                    index as slong,
                    context.default_as_ptr(),
                ) as u64
            },
        ));
    }
    DenseFactorization {
        coefficient: field.wrap_extension(leading.to_extension()),
        factors,
    }
}

fn dense_fmpz_polynomial(raw: *const flint_sys::fmpz_poly_struct) -> Vec<Integer> {
    let length = unsafe { fmpz_poly_length(raw) } as usize;
    (0..length)
        .map(|exponent| {
            let mut coefficient = Integer::zero();
            unsafe { fmpz_poly_get_coeff_fmpz(coefficient.as_mut_ptr(), raw, exponent as slong) };
            coefficient
        })
        .collect()
}

fn dense_nmod_polynomial(
    field: &FiniteField,
    raw: *const flint_sys::nmod_poly_struct,
) -> Vec<FiniteFieldElement> {
    let length = unsafe { nmod_poly_length(raw) } as usize;
    (0..length)
        .map(|exponent| field.element(unsafe { nmod_poly_get_coeff_ui(raw, exponent as slong) }))
        .collect()
}

fn dense_fmpz_mod_polynomial(
    field: &FiniteField,
    context: &OwnedContext<fmpz_mod_ctx_struct>,
    raw: *const flint_sys::fmpz_mod_poly_struct,
) -> Vec<FiniteFieldElement> {
    let length = unsafe { fmpz_mod_poly_length(raw, context.as_ptr()) } as usize;
    (0..length)
        .map(|exponent| {
            let mut coefficient = Integer::zero();
            unsafe {
                fmpz_mod_poly_get_coeff_fmpz(
                    coefficient.as_mut_ptr(),
                    raw,
                    exponent as slong,
                    context.as_ptr(),
                )
            };
            field.element(coefficient)
        })
        .collect()
}

fn dense_fq_default_polynomial(
    field: &FiniteField,
    context: &Arc<ExtensionContext>,
    raw: *const flint_sys::fq_default_poly_struct,
) -> Vec<FiniteFieldElement> {
    let length = unsafe { fq_default_poly_length(raw, context.default_as_ptr()) } as usize;
    (0..length)
        .map(|exponent| {
            let mut coefficient = DefaultExtensionElement::initialized(Arc::clone(context));
            unsafe {
                fq_default_poly_get_coeff(
                    &mut coefficient.raw,
                    raw,
                    exponent as slong,
                    context.default_as_ptr(),
                )
            };
            field.wrap_extension(coefficient.to_extension())
        })
        .collect()
}

fn order(order: &MonomialOrder) -> Option<ordering_t> {
    match order.standard_name()? {
        "lex" => Some(ordering_t_ORD_LEX),
        "grlex" => Some(ordering_t_ORD_DEGLEX),
        "grevlex" => Some(ordering_t_ORD_DEGREVLEX),
        _ => None,
    }
}

fn machine_terms<E>(terms: &BTreeMap<Vec<Integer>, E>) -> Option<Vec<(Vec<u64>, &E)>> {
    terms
        .iter()
        .map(|(exponent, coefficient)| {
            exponent
                .iter()
                .map(Integer::to_u64_checked)
                .collect::<Option<Vec<_>>>()
                .map(|exponent| (exponent, coefficient))
        })
        .collect()
}

#[cfg(test)]
fn product_exponents_fit<E>(
    left: &[(Vec<u64>, &E)],
    right: &[(Vec<u64>, &E)],
    order: MonomialOrder,
) -> bool {
    let arity = left[0].0.len();
    for variable in 0..arity {
        let left_max = left
            .iter()
            .map(|(exponent, _)| exponent[variable])
            .max()
            .unwrap_or(0);
        let right_max = right
            .iter()
            .map(|(exponent, _)| exponent[variable])
            .max()
            .unwrap_or(0);
        if left_max.checked_add(right_max).is_none() {
            return false;
        }
    }
    if order == MonomialOrder::Lexicographic {
        return true;
    }
    let max_total = |terms: &[(Vec<u64>, &E)]| {
        terms
            .iter()
            .map(|(exponent, _)| {
                exponent
                    .iter()
                    .try_fold(0_u64, |total, value| total.checked_add(*value))
            })
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .max()
    };
    max_total(left)
        .zip(max_total(right))
        .is_some_and(|(left, right)| left.checked_add(right).is_some())
}

#[cfg(test)]
fn typed_terms<'a, E: Any, T: Any>(terms: &'a [(Vec<u64>, &'a E)]) -> Vec<(Vec<u64>, &'a T)> {
    terms
        .iter()
        .map(|(exponent, coefficient)| (exponent.clone(), cast_ref::<E, T>(*coefficient)))
        .collect()
}

#[cfg(test)]
fn cast_terms<T: Any, E: Any>(terms: Vec<(Vec<u64>, T)>) -> BTreeMap<Vec<Integer>, E> {
    terms
        .into_iter()
        .map(|(exponent, coefficient)| {
            (
                exponent.into_iter().map(Integer::from).collect(),
                cast_owned(coefficient),
            )
        })
        .collect()
}

fn is_context<R: Ring, C: Any, E: Any>() -> bool {
    TypeId::of::<R>() == TypeId::of::<C>() && TypeId::of::<R::Element>() == TypeId::of::<E>()
}

fn cast_ref<T: Any, U: Any>(value: &T) -> &U {
    (value as &dyn Any)
        .downcast_ref::<U>()
        .expect("private polynomial dispatch type invariant")
}

fn cast_owned<T: Any, U: Any>(value: T) -> U {
    try_cast_owned(value).unwrap_or_else(|_| panic!("private polynomial dispatch type invariant"))
}

pub(super) fn try_cast_owned<T: Any, U: Any>(value: T) -> Result<U, T> {
    if TypeId::of::<T>() != TypeId::of::<U>() {
        return Err(value);
    }
    let value = std::mem::ManuallyDrop::new(value);
    // SAFETY: equal `TypeId`s mean that `T` and `U` are the same concrete
    // `'static` type. Reading through the retyped pointer moves that value;
    // `ManuallyDrop` prevents the original binding from being dropped twice.
    Ok(unsafe { std::ptr::read((&*value as *const T).cast::<U>()) })
}

struct OwnedUnivariate<P> {
    raw: P,
    clear: unsafe extern "C" fn(*mut P),
}

impl<P> OwnedUnivariate<P> {
    fn new(initialize: impl FnOnce(*mut P), clear: unsafe extern "C" fn(*mut P)) -> Self {
        let mut raw = MaybeUninit::<P>::uninit();
        initialize(raw.as_mut_ptr());
        Self {
            raw: unsafe { raw.assume_init() },
            clear,
        }
    }

    fn as_ptr(&self) -> *const P {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut P {
        &mut self.raw
    }
}

impl<P> Drop for OwnedUnivariate<P> {
    fn drop(&mut self) {
        unsafe { (self.clear)(&mut self.raw) };
    }
}

struct OwnedContext<C> {
    raw: C,
    clear: unsafe extern "C" fn(*mut C),
}

impl<C> OwnedContext<C> {
    fn new(initialize: impl FnOnce(*mut C), clear: unsafe extern "C" fn(*mut C)) -> Self {
        let mut raw = MaybeUninit::<C>::uninit();
        initialize(raw.as_mut_ptr());
        Self {
            raw: unsafe { raw.assume_init() },
            clear,
        }
    }

    fn as_ptr(&self) -> *const C {
        &self.raw
    }
}

impl<C> Drop for OwnedContext<C> {
    fn drop(&mut self) {
        unsafe { (self.clear)(&mut self.raw) };
    }
}

struct OwnedPolynomial<'a, C, P> {
    raw: P,
    context: &'a OwnedContext<C>,
    clear: unsafe extern "C" fn(*mut P, *const C),
}

impl<'a, C, P> OwnedPolynomial<'a, C, P> {
    fn new(
        context: &'a OwnedContext<C>,
        initialize: unsafe extern "C" fn(*mut P, *const C),
        clear: unsafe extern "C" fn(*mut P, *const C),
    ) -> Self {
        let mut raw = MaybeUninit::<P>::uninit();
        unsafe { initialize(raw.as_mut_ptr(), context.as_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
            clear,
        }
    }

    fn as_ptr(&self) -> *const P {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut P {
        &mut self.raw
    }
}

impl<C, P> Drop for OwnedPolynomial<'_, C, P> {
    fn drop(&mut self) {
        unsafe { (self.clear)(&mut self.raw, self.context.as_ptr()) };
    }
}

struct OwnedMpolyFactor<'a, C, F> {
    raw: F,
    context: &'a C,
    clear: unsafe extern "C" fn(*mut F, *const C),
}

impl<'a, C, F> OwnedMpolyFactor<'a, C, F> {
    fn new(
        context: &'a C,
        initialize: unsafe extern "C" fn(*mut F, *const C),
        clear: unsafe extern "C" fn(*mut F, *const C),
    ) -> Self {
        let mut raw = MaybeUninit::<F>::uninit();
        unsafe { initialize(raw.as_mut_ptr(), context) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
            clear,
        }
    }

    fn as_ptr(&self) -> *const F {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut F {
        &mut self.raw
    }
}

impl<C, F> Drop for OwnedMpolyFactor<'_, C, F> {
    fn drop(&mut self) {
        unsafe { (self.clear)(&mut self.raw, self.context) };
    }
}

struct OwnedFmpzPolynomialFactor {
    raw: fmpz_poly_factor_struct,
}

impl OwnedFmpzPolynomialFactor {
    fn new() -> Self {
        let mut raw = MaybeUninit::<fmpz_poly_factor_struct>::uninit();
        unsafe { fmpz_poly_factor_init(raw.as_mut_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
        }
    }

    fn as_ptr(&self) -> *const fmpz_poly_factor_struct {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut fmpz_poly_factor_struct {
        &mut self.raw
    }
}

impl Drop for OwnedFmpzPolynomialFactor {
    fn drop(&mut self) {
        unsafe { fmpz_poly_factor_clear(&mut self.raw) };
    }
}

struct OwnedNmodPolynomialFactor {
    raw: nmod_poly_factor_struct,
}

impl OwnedNmodPolynomialFactor {
    fn new() -> Self {
        let mut raw = MaybeUninit::<nmod_poly_factor_struct>::uninit();
        unsafe { nmod_poly_factor_init(raw.as_mut_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
        }
    }

    fn as_ptr(&self) -> *const nmod_poly_factor_struct {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut nmod_poly_factor_struct {
        &mut self.raw
    }
}

impl Drop for OwnedNmodPolynomialFactor {
    fn drop(&mut self) {
        unsafe { nmod_poly_factor_clear(&mut self.raw) };
    }
}

struct OwnedFmpzModPolynomialFactor<'a> {
    raw: fmpz_mod_poly_factor_struct,
    context: &'a OwnedContext<fmpz_mod_ctx_struct>,
}

impl<'a> OwnedFmpzModPolynomialFactor<'a> {
    fn new(context: &'a OwnedContext<fmpz_mod_ctx_struct>) -> Self {
        let mut raw = MaybeUninit::<fmpz_mod_poly_factor_struct>::uninit();
        unsafe { fmpz_mod_poly_factor_init(raw.as_mut_ptr(), context.as_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    fn as_ptr(&self) -> *const fmpz_mod_poly_factor_struct {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut fmpz_mod_poly_factor_struct {
        &mut self.raw
    }
}

impl Drop for OwnedFmpzModPolynomialFactor<'_> {
    fn drop(&mut self) {
        unsafe { fmpz_mod_poly_factor_clear(&mut self.raw, self.context.as_ptr()) };
    }
}

#[cfg(test)]
fn multiply_fmpz_series(
    left: &[(u64, &Integer)],
    right: &[(u64, &Integer)],
    precision: slong,
) -> Vec<(u64, Integer)> {
    let mut left_raw =
        OwnedUnivariate::new(|output| unsafe { fmpz_poly_init(output) }, fmpz_poly_clear);
    let mut right_raw =
        OwnedUnivariate::new(|output| unsafe { fmpz_poly_init(output) }, fmpz_poly_clear);
    for (exponent, coefficient) in left {
        unsafe {
            fmpz_poly_set_coeff_fmpz(
                left_raw.as_mut_ptr(),
                *exponent as slong,
                coefficient.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        unsafe {
            fmpz_poly_set_coeff_fmpz(
                right_raw.as_mut_ptr(),
                *exponent as slong,
                coefficient.as_ptr(),
            )
        };
    }
    let mut output = OwnedUnivariate::new(|raw| unsafe { fmpz_poly_init(raw) }, fmpz_poly_clear);
    unsafe {
        fmpz_poly_mullow(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            precision,
        )
    };
    let length = unsafe { fmpz_poly_length(output.as_ptr()) } as usize;
    (0..length)
        .filter_map(|exponent| {
            let mut coefficient = Integer::zero();
            unsafe {
                fmpz_poly_get_coeff_fmpz(
                    coefficient.as_mut_ptr(),
                    output.as_ptr(),
                    exponent as slong,
                )
            };
            (coefficient != Integer::zero()).then_some((exponent as u64, coefficient))
        })
        .collect()
}

#[cfg(test)]
fn multiply_fmpq_series(
    left: &[(u64, &Rational)],
    right: &[(u64, &Rational)],
    precision: slong,
) -> Vec<(u64, Rational)> {
    let mut left_raw =
        OwnedUnivariate::new(|output| unsafe { fmpq_poly_init(output) }, fmpq_poly_clear);
    let mut right_raw =
        OwnedUnivariate::new(|output| unsafe { fmpq_poly_init(output) }, fmpq_poly_clear);
    for (exponent, coefficient) in left {
        unsafe {
            fmpq_poly_set_coeff_fmpq(
                left_raw.as_mut_ptr(),
                *exponent as slong,
                coefficient.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        unsafe {
            fmpq_poly_set_coeff_fmpq(
                right_raw.as_mut_ptr(),
                *exponent as slong,
                coefficient.as_ptr(),
            )
        };
    }
    let mut output = OwnedUnivariate::new(|raw| unsafe { fmpq_poly_init(raw) }, fmpq_poly_clear);
    unsafe {
        fmpq_poly_mullow(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            precision,
        )
    };
    let length = unsafe { fmpq_poly_length(output.as_ptr()) } as usize;
    (0..length)
        .filter_map(|exponent| {
            let mut coefficient = Rational::zero();
            unsafe {
                fmpq_poly_get_coeff_fmpq(
                    coefficient.as_mut_ptr(),
                    output.as_ptr(),
                    exponent as slong,
                )
            };
            (coefficient != Rational::zero()).then_some((exponent as u64, coefficient))
        })
        .collect()
}

#[cfg(test)]
fn multiply_nmod_series(
    modulus: u64,
    left: &[(u64, u64)],
    right: &[(u64, u64)],
    precision: slong,
) -> Vec<(u64, u64)> {
    let mut left_raw = OwnedUnivariate::new(
        |output| unsafe { nmod_poly_init(output, modulus) },
        nmod_poly_clear,
    );
    let mut right_raw = OwnedUnivariate::new(
        |output| unsafe { nmod_poly_init(output, modulus) },
        nmod_poly_clear,
    );
    for (exponent, coefficient) in left {
        unsafe { nmod_poly_set_coeff_ui(left_raw.as_mut_ptr(), *exponent as slong, *coefficient) };
    }
    for (exponent, coefficient) in right {
        unsafe { nmod_poly_set_coeff_ui(right_raw.as_mut_ptr(), *exponent as slong, *coefficient) };
    }
    let mut output = OwnedUnivariate::new(
        |raw| unsafe { nmod_poly_init(raw, modulus) },
        nmod_poly_clear,
    );
    unsafe {
        nmod_poly_mullow(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            precision,
        )
    };
    let length = unsafe { nmod_poly_length(output.as_ptr()) } as usize;
    (0..length)
        .filter_map(|exponent| {
            let coefficient = unsafe { nmod_poly_get_coeff_ui(output.as_ptr(), exponent as slong) };
            (coefficient != 0).then_some((exponent as u64, coefficient))
        })
        .collect()
}

#[cfg(test)]
fn multiply_fmpz_mod_series(
    modulus: &Integer,
    left: &[(u64, &Integer)],
    right: &[(u64, &Integer)],
    precision: slong,
) -> Vec<(u64, Integer)> {
    let context = OwnedContext::<fmpz_mod_ctx_struct>::new(
        |output| unsafe { fmpz_mod_ctx_init(output, modulus.as_ptr()) },
        fmpz_mod_ctx_clear,
    );
    let mut left_raw = OwnedPolynomial::new(&context, fmpz_mod_poly_init, fmpz_mod_poly_clear);
    let mut right_raw = OwnedPolynomial::new(&context, fmpz_mod_poly_init, fmpz_mod_poly_clear);
    for (exponent, coefficient) in left {
        unsafe {
            fmpz_mod_poly_set_coeff_fmpz(
                left_raw.as_mut_ptr(),
                *exponent as slong,
                coefficient.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        unsafe {
            fmpz_mod_poly_set_coeff_fmpz(
                right_raw.as_mut_ptr(),
                *exponent as slong,
                coefficient.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    let mut output = OwnedPolynomial::new(&context, fmpz_mod_poly_init, fmpz_mod_poly_clear);
    unsafe {
        fmpz_mod_poly_mullow(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            precision,
            context.as_ptr(),
        )
    };
    let length = unsafe { fmpz_mod_poly_length(output.as_ptr(), context.as_ptr()) } as usize;
    (0..length)
        .filter_map(|exponent| {
            let mut coefficient = Integer::zero();
            unsafe {
                fmpz_mod_poly_get_coeff_fmpz(
                    coefficient.as_mut_ptr(),
                    output.as_ptr(),
                    exponent as slong,
                    context.as_ptr(),
                )
            };
            (coefficient != Integer::zero()).then_some((exponent as u64, coefficient))
        })
        .collect()
}

struct OwnedFqDefaultPolynomial {
    raw: flint_sys::fq_default_poly_struct,
    context: std::sync::Arc<ExtensionContext>,
}

impl OwnedFqDefaultPolynomial {
    fn new(context: std::sync::Arc<ExtensionContext>) -> Self {
        let mut raw = MaybeUninit::<flint_sys::fq_default_poly_struct>::uninit();
        unsafe { fq_default_poly_init(raw.as_mut_ptr(), context.default_as_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    fn as_ptr(&self) -> *const flint_sys::fq_default_poly_struct {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut flint_sys::fq_default_poly_struct {
        &mut self.raw
    }
}

impl Drop for OwnedFqDefaultPolynomial {
    fn drop(&mut self) {
        unsafe { fq_default_poly_clear(&mut self.raw, self.context.default_as_ptr()) };
    }
}

struct OwnedFqDefaultPolynomialFactor {
    raw: fq_default_poly_factor_struct,
    context: Arc<ExtensionContext>,
}

impl OwnedFqDefaultPolynomialFactor {
    fn new(context: Arc<ExtensionContext>) -> Self {
        let mut raw = MaybeUninit::<fq_default_poly_factor_struct>::uninit();
        unsafe { fq_default_poly_factor_init(raw.as_mut_ptr(), context.default_as_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    fn as_ptr(&self) -> *const fq_default_poly_factor_struct {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut fq_default_poly_factor_struct {
        &mut self.raw
    }
}

impl Drop for OwnedFqDefaultPolynomialFactor {
    fn drop(&mut self) {
        unsafe { fq_default_poly_factor_clear(&mut self.raw, self.context.default_as_ptr()) };
    }
}

#[cfg(test)]
fn multiply_fq_nmod_series(
    field: &FiniteField,
    left: &[(u64, &FiniteFieldElement)],
    right: &[(u64, &FiniteFieldElement)],
    precision: slong,
) -> Vec<(u64, FiniteFieldElement)> {
    let context = field
        .extension_context()
        .expect("extension-field series dispatch requires an fq_nmod context");
    let mut left_raw = OwnedFqDefaultPolynomial::new(std::sync::Arc::clone(&context));
    let mut right_raw = OwnedFqDefaultPolynomial::new(std::sync::Arc::clone(&context));
    for (exponent, coefficient) in left {
        let coefficient = coefficient
            .extension_value()
            .expect("extension-field coefficient belongs to the selected context")
            .to_default();
        unsafe {
            fq_default_poly_set_coeff(
                left_raw.as_mut_ptr(),
                *exponent as slong,
                &coefficient.raw,
                context.default_as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        let coefficient = coefficient
            .extension_value()
            .expect("extension-field coefficient belongs to the selected context")
            .to_default();
        unsafe {
            fq_default_poly_set_coeff(
                right_raw.as_mut_ptr(),
                *exponent as slong,
                &coefficient.raw,
                context.default_as_ptr(),
            )
        };
    }
    let mut output = OwnedFqDefaultPolynomial::new(std::sync::Arc::clone(&context));
    unsafe {
        fq_default_poly_mullow(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            precision,
            context.default_as_ptr(),
        )
    };
    let length =
        unsafe { fq_default_poly_length(output.as_ptr(), context.default_as_ptr()) } as usize;
    (0..length)
        .filter_map(|exponent| {
            let mut coefficient =
                DefaultExtensionElement::initialized(std::sync::Arc::clone(&context));
            unsafe {
                fq_default_poly_get_coeff(
                    &mut coefficient.raw,
                    output.as_ptr(),
                    exponent as slong,
                    context.default_as_ptr(),
                )
            };
            let coefficient = field.wrap_extension(coefficient.to_extension());
            (!coefficient.is_zero()).then_some((exponent as u64, coefficient))
        })
        .collect()
}

#[cfg(test)]
fn multiply_fmpz(
    nvars: slong,
    order: ordering_t,
    left: &[(Vec<u64>, &Integer)],
    right: &[(Vec<u64>, &Integer)],
) -> Vec<(Vec<u64>, Integer)> {
    let context = OwnedContext::new(
        |output| unsafe { fmpz_mpoly_ctx_init(output, nvars, order) },
        fmpz_mpoly_ctx_clear,
    );
    let mut left_raw = OwnedPolynomial::new(&context, fmpz_mpoly_init, fmpz_mpoly_clear);
    let mut right_raw = OwnedPolynomial::new(&context, fmpz_mpoly_init, fmpz_mpoly_clear);
    for (exponent, coefficient) in left {
        unsafe {
            fmpz_mpoly_set_coeff_fmpz_ui(
                left_raw.as_mut_ptr(),
                coefficient.as_ptr(),
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        unsafe {
            fmpz_mpoly_set_coeff_fmpz_ui(
                right_raw.as_mut_ptr(),
                coefficient.as_ptr(),
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    let mut output = OwnedPolynomial::new(&context, fmpz_mpoly_init, fmpz_mpoly_clear);
    unsafe {
        fmpz_mpoly_mul(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            context.as_ptr(),
        )
    };
    let length = unsafe { fmpz_mpoly_length(output.as_ptr(), context.as_ptr()) } as usize;
    (0..length)
        .map(|term| {
            let mut coefficient = Integer::zero();
            let mut exponent = vec![0; nvars as usize];
            unsafe {
                fmpz_mpoly_get_term_coeff_fmpz(
                    coefficient.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
                fmpz_mpoly_get_term_exp_ui(
                    exponent.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
            }
            (exponent, coefficient)
        })
        .collect()
}

#[cfg(test)]
fn multiply_fmpq(
    nvars: slong,
    order: ordering_t,
    left: &[(Vec<u64>, &Rational)],
    right: &[(Vec<u64>, &Rational)],
) -> Vec<(Vec<u64>, Rational)> {
    let context = OwnedContext::new(
        |output| unsafe { fmpq_mpoly_ctx_init(output, nvars, order) },
        fmpq_mpoly_ctx_clear,
    );
    let mut left_raw = OwnedPolynomial::new(&context, fmpq_mpoly_init, fmpq_mpoly_clear);
    let mut right_raw = OwnedPolynomial::new(&context, fmpq_mpoly_init, fmpq_mpoly_clear);
    for (exponent, coefficient) in left {
        unsafe {
            fmpq_mpoly_set_coeff_fmpq_ui(
                left_raw.as_mut_ptr(),
                coefficient.as_ptr(),
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        unsafe {
            fmpq_mpoly_set_coeff_fmpq_ui(
                right_raw.as_mut_ptr(),
                coefficient.as_ptr(),
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    let mut output = OwnedPolynomial::new(&context, fmpq_mpoly_init, fmpq_mpoly_clear);
    unsafe {
        fmpq_mpoly_mul(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            context.as_ptr(),
        )
    };
    let length = unsafe { fmpq_mpoly_length(output.as_ptr(), context.as_ptr()) } as usize;
    (0..length)
        .map(|term| {
            let mut coefficient = Rational::zero();
            let mut exponent = vec![0; nvars as usize];
            unsafe {
                fmpq_mpoly_get_term_coeff_fmpq(
                    coefficient.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
                fmpq_mpoly_get_term_exp_ui(
                    exponent.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
            }
            (exponent, coefficient)
        })
        .collect()
}

#[cfg(test)]
fn multiply_nmod(
    nvars: slong,
    order: ordering_t,
    modulus: u64,
    left: &[(Vec<u64>, u64)],
    right: &[(Vec<u64>, u64)],
) -> Vec<(Vec<u64>, u64)> {
    let context = OwnedContext::new(
        |output| unsafe { nmod_mpoly_ctx_init(output, nvars, order, modulus) },
        nmod_mpoly_ctx_clear,
    );
    let mut left_raw = OwnedPolynomial::new(&context, nmod_mpoly_init, nmod_mpoly_clear);
    let mut right_raw = OwnedPolynomial::new(&context, nmod_mpoly_init, nmod_mpoly_clear);
    for (exponent, coefficient) in left {
        unsafe {
            nmod_mpoly_set_coeff_ui_ui(
                left_raw.as_mut_ptr(),
                *coefficient,
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        unsafe {
            nmod_mpoly_set_coeff_ui_ui(
                right_raw.as_mut_ptr(),
                *coefficient,
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    let mut output = OwnedPolynomial::new(&context, nmod_mpoly_init, nmod_mpoly_clear);
    unsafe {
        nmod_mpoly_mul(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            context.as_ptr(),
        )
    };
    let length = unsafe { nmod_mpoly_length(output.as_ptr(), context.as_ptr()) } as usize;
    (0..length)
        .map(|term| {
            let coefficient = unsafe {
                nmod_mpoly_get_term_coeff_ui(output.as_ptr(), term as slong, context.as_ptr())
            };
            let mut exponent = vec![0; nvars as usize];
            unsafe {
                nmod_mpoly_get_term_exp_ui(
                    exponent.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                )
            };
            (exponent, coefficient)
        })
        .collect()
}

#[cfg(test)]
fn multiply_fmpz_mod(
    nvars: slong,
    order: ordering_t,
    modulus: &Integer,
    left: &[(Vec<u64>, &Integer)],
    right: &[(Vec<u64>, &Integer)],
) -> Vec<(Vec<u64>, Integer)> {
    let context = OwnedContext::new(
        |output| unsafe { fmpz_mod_mpoly_ctx_init(output, nvars, order, modulus.as_ptr()) },
        fmpz_mod_mpoly_ctx_clear,
    );
    let mut left_raw = OwnedPolynomial::new(&context, fmpz_mod_mpoly_init, fmpz_mod_mpoly_clear);
    let mut right_raw = OwnedPolynomial::new(&context, fmpz_mod_mpoly_init, fmpz_mod_mpoly_clear);
    for (exponent, coefficient) in left {
        unsafe {
            fmpz_mod_mpoly_set_coeff_fmpz_ui(
                left_raw.as_mut_ptr(),
                coefficient.as_ptr(),
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        unsafe {
            fmpz_mod_mpoly_set_coeff_fmpz_ui(
                right_raw.as_mut_ptr(),
                coefficient.as_ptr(),
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    let mut output = OwnedPolynomial::new(&context, fmpz_mod_mpoly_init, fmpz_mod_mpoly_clear);
    unsafe {
        fmpz_mod_mpoly_mul(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            context.as_ptr(),
        )
    };
    let length = unsafe { fmpz_mod_mpoly_length(output.as_ptr(), context.as_ptr()) } as usize;
    (0..length)
        .map(|term| {
            let mut coefficient = Integer::zero();
            let mut exponent = vec![0; nvars as usize];
            unsafe {
                fmpz_mod_mpoly_get_term_coeff_fmpz(
                    coefficient.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
                fmpz_mod_mpoly_get_term_exp_ui(
                    exponent.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
            }
            (exponent, coefficient)
        })
        .collect()
}

#[cfg(test)]
fn multiply_fq_nmod(
    nvars: slong,
    order: ordering_t,
    field: &FiniteField,
    left: &[(Vec<u64>, &FiniteFieldElement)],
    right: &[(Vec<u64>, &FiniteFieldElement)],
) -> Vec<(Vec<u64>, FiniteFieldElement)> {
    let field_context = field
        .extension_context()
        .expect("extension-field polynomial dispatch invariant");
    let context = OwnedContext::new(
        |output| unsafe { fq_nmod_mpoly_ctx_init(output, nvars, order, field_context.as_ptr()) },
        fq_nmod_mpoly_ctx_clear,
    );
    let mut left_raw = OwnedPolynomial::new(&context, fq_nmod_mpoly_init, fq_nmod_mpoly_clear);
    let mut right_raw = OwnedPolynomial::new(&context, fq_nmod_mpoly_init, fq_nmod_mpoly_clear);
    for (exponent, coefficient) in left {
        let coefficient = coefficient
            .extension_value()
            .expect("extension-field coefficient invariant");
        unsafe {
            fq_nmod_mpoly_set_coeff_fq_nmod_ui(
                left_raw.as_mut_ptr(),
                &coefficient.raw,
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    for (exponent, coefficient) in right {
        let coefficient = coefficient
            .extension_value()
            .expect("extension-field coefficient invariant");
        unsafe {
            fq_nmod_mpoly_set_coeff_fq_nmod_ui(
                right_raw.as_mut_ptr(),
                &coefficient.raw,
                exponent.as_ptr(),
                context.as_ptr(),
            )
        };
    }
    let mut output = OwnedPolynomial::new(&context, fq_nmod_mpoly_init, fq_nmod_mpoly_clear);
    unsafe {
        fq_nmod_mpoly_mul(
            output.as_mut_ptr(),
            left_raw.as_ptr(),
            right_raw.as_ptr(),
            context.as_ptr(),
        )
    };
    let length = unsafe { fq_nmod_mpoly_length(output.as_ptr(), context.as_ptr()) } as usize;
    (0..length)
        .map(|term| {
            let mut coefficient = ExtensionElement::initialized(field_context.clone());
            let mut exponent = vec![0; nvars as usize];
            unsafe {
                fq_nmod_mpoly_get_term_coeff_fq_nmod(
                    &mut coefficient.raw,
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
                fq_nmod_mpoly_get_term_exp_ui(
                    exponent.as_mut_ptr(),
                    output.as_ptr(),
                    term as slong,
                    context.as_ptr(),
                );
            }
            (exponent, field.wrap_extension(coefficient))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Q, Z};

    fn generic_product<R: CommutativeRing>(
        ring: &PolynomialRing<R>,
        left: &BTreeMap<Vec<Integer>, R::Element>,
        right: &BTreeMap<Vec<Integer>, R::Element>,
    ) -> BTreeMap<Vec<Integer>, R::Element> {
        let mut output = BTreeMap::new();
        for (left_exponent, left_coefficient) in left {
            for (right_exponent, right_coefficient) in right {
                let exponent = left_exponent
                    .iter()
                    .zip(right_exponent)
                    .map(|(left, right)| left + right)
                    .collect::<Vec<_>>();
                let product = ring
                    .coefficient_ring()
                    .mul(left_coefficient, right_coefficient)
                    .unwrap();
                let coefficient = if let Some(previous) = output.remove(&exponent) {
                    ring.coefficient_ring().add(&previous, &product).unwrap()
                } else {
                    product
                };
                if coefficient != ring.coefficient_ring().zero() {
                    output.insert(exponent, coefficient);
                }
            }
        }
        output
    }

    fn assert_dispatch_matches_generic<R: CommutativeRing>(
        ring: PolynomialRing<R>,
        left: BTreeMap<Vec<Integer>, R::Element>,
        right: BTreeMap<Vec<Integer>, R::Element>,
    ) {
        let actual = multiply_with_crossover(&ring, &left, &right, false)
            .expect("context should select FLINT");
        let expected = generic_product(&ring, &left, &right);
        assert!(actual == expected);
    }

    fn exponents(values: [u64; 2]) -> Vec<Integer> {
        values.into_iter().map(Integer::from).collect()
    }

    #[test]
    fn exact_coefficient_families_match_the_generic_product() {
        let integers =
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedLexicographic).unwrap();
        assert_dispatch_matches_generic(
            integers,
            BTreeMap::from([
                (exponents([3, 0]), Integer::from(7)),
                (exponents([0, 2]), Integer::from(-5)),
            ]),
            BTreeMap::from([
                (exponents([1, 1]), Integer::from(11)),
                (exponents([0, 0]), Integer::from(13)),
            ]),
        );

        let rationals = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        assert_dispatch_matches_generic(
            rationals,
            BTreeMap::from([
                (exponents([3, 0]), "7/3".parse::<Rational>().unwrap()),
                (exponents([0, 2]), "-5/2".parse::<Rational>().unwrap()),
            ]),
            BTreeMap::from([
                (exponents([1, 1]), "11/5".parse::<Rational>().unwrap()),
                (exponents([0, 0]), "13/7".parse::<Rational>().unwrap()),
            ]),
        );

        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            let ring = PolynomialRing::new(
                residues.clone(),
                ["x", "y"],
                MonomialOrder::GradedReverseLexicographic,
            )
            .unwrap();
            assert_dispatch_matches_generic(
                ring,
                BTreeMap::from([
                    (exponents([3, 0]), residues.element(7_u8)),
                    (exponents([0, 2]), residues.element(20_u8)),
                ]),
                BTreeMap::from([
                    (exponents([1, 1]), residues.element(11_u8)),
                    (exponents([0, 0]), residues.element(13_u8)),
                ]),
            );
        }

        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(3, 2).unwrap(),
        ] {
            let ring = PolynomialRing::new(
                field.clone(),
                ["x", "y"],
                MonomialOrder::GradedReverseLexicographic,
            )
            .unwrap();
            assert_dispatch_matches_generic(
                ring,
                BTreeMap::from([
                    (exponents([3, 0]), field.element(2_u8)),
                    (exponents([0, 2]), field.generator()),
                ]),
                BTreeMap::from([
                    (exponents([1, 1]), field.element(3_u8)),
                    (exponents([0, 0]), field.generator()),
                ]),
            );
        }
    }

    #[test]
    fn arbitrary_exponents_fail_over_to_the_exact_generic_path() {
        let ring =
            PolynomialRing::new(Z, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let left = BTreeMap::from([(vec![Integer::from(u64::MAX)], Integer::one())]);
        let right = BTreeMap::from([(vec![Integer::one()], Integer::one())]);
        assert!(multiply_with_crossover(&ring, &left, &right, false).is_none());
    }

    #[test]
    fn univariate_series_dispatch_truncates_and_matches_the_ring_floor() {
        fn check<R: CommutativeRing>(
            ring: R,
            left: BTreeMap<Vec<u64>, R::Element>,
            right: BTreeMap<Vec<u64>, R::Element>,
        ) {
            let actual =
                multiply_univariate_series_with_crossover(&ring, &left, &right, 5, false).unwrap();
            let mut expected = BTreeMap::new();
            for (left_exponent, left_coefficient) in &left {
                for (right_exponent, right_coefficient) in &right {
                    let exponent = left_exponent[0] + right_exponent[0];
                    if exponent >= 5 {
                        continue;
                    }
                    let product = ring.mul(left_coefficient, right_coefficient).unwrap();
                    let coefficient = expected.remove(&vec![exponent]).map_or_else(
                        || product.clone(),
                        |previous| ring.add(&previous, &product).unwrap(),
                    );
                    if coefficient != ring.zero() {
                        expected.insert(vec![exponent], coefficient);
                    }
                }
            }
            assert!(actual == expected);
        }

        check(
            Z,
            BTreeMap::from([(vec![0], Integer::from(2)), (vec![3], Integer::from(5))]),
            BTreeMap::from([(vec![1], Integer::from(7)), (vec![4], Integer::from(11))]),
        );
        check(
            Q,
            BTreeMap::from([
                (vec![0], "2/3".parse::<Rational>().unwrap()),
                (vec![3], "5/7".parse().unwrap()),
            ]),
            BTreeMap::from([
                (vec![1], "7/11".parse::<Rational>().unwrap()),
                (vec![4], "11/13".parse().unwrap()),
            ]),
        );
        let residues = ResidueRing::new(25_u8).unwrap();
        check(
            residues.clone(),
            BTreeMap::from([
                (vec![0], residues.element(2)),
                (vec![3], residues.element(5)),
            ]),
            BTreeMap::from([
                (vec![1], residues.element(7)),
                (vec![4], residues.element(11)),
            ]),
        );
        let field = FiniteField::prime(7_u8).unwrap();
        check(
            field.clone(),
            BTreeMap::from([(vec![0], field.element(2)), (vec![3], field.element(5))]),
            BTreeMap::from([(vec![1], field.element(7)), (vec![4], field.element(11))]),
        );

        let extension = FiniteField::extension(3, 2).unwrap();
        check(
            extension.clone(),
            BTreeMap::from([(vec![0], extension.generator()), (vec![3], extension.one())]),
            BTreeMap::from([(vec![1], extension.generator()), (vec![4], extension.one())]),
        );
        assert!(multiply_univariate_series(
            &extension,
            &BTreeMap::from([(vec![0], extension.generator()), (vec![1], extension.one())]),
            &BTreeMap::from([(vec![0], extension.one()), (vec![1], extension.generator())]),
            5,
        )
        .is_some());

        let big_residues =
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap();
        check(
            big_residues.clone(),
            BTreeMap::from([
                (vec![0], big_residues.element(2)),
                (vec![3], big_residues.element(5)),
            ]),
            BTreeMap::from([
                (vec![1], big_residues.element(7)),
                (vec![4], big_residues.element(11)),
            ]),
        );
        let dense = (0..4)
            .map(|exponent| {
                (
                    vec![exponent],
                    big_residues.element(Integer::from(exponent + 1)),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert!(multiply_univariate_series(&big_residues, &dense, &dense, 8).is_some());
    }
}

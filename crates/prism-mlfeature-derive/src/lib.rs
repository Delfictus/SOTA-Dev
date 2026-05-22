//! # PRISM-4D Pillar 5 firewall (proc-macro)
//!
//! Compile-time enforcement that `FeatureRole::ReportingOnly` fields cannot
//! enter ML training tensors.
//!
//! ## Usage — named-field struct
//!
//! ```ignore
//! use prism_mlfeature_derive::MLFeature;
//! use prism_nhs::feature_role::FeatureRole;
//!
//! #[derive(MLFeature)]
//! pub struct CleanFeatures {
//!     #[role(Mechanistic)]
//!     pub kcc_score: f64,
//!     #[role(CausalInformation)]
//!     pub transfer_entropy: f64,
//! }
//! ```
//!
//! ## Usage — tuple struct (newtype)
//!
//! ```ignore
//! #[derive(MLFeature)]
//! pub struct CausalDriverView(
//!     #[role(Mechanistic)] pub ManifoldViewData,
//! );
//! ```
//!
//! Tuple-struct fields are referenced by synthetic name `field_<N>` where
//! `<N>` is the zero-based positional index. The diagnostic for a tuple
//! `ReportingOnly` violation reads:
//! `field field_0 (of CausalDriverView) has FeatureRole::ReportingOnly which
//! is forbidden ...`.
//!
//! ## Stable-Rust translation note
//!
//! The PRISM-4D Entangled Transform Blueprint specifies the firewall as a
//! `MLFeatureRole<const R: FeatureRole>` parameterised marker trait. Custom
//! enum types as const-generic parameters require the `adt_const_params`
//! nightly feature. To keep the firewall on stable Rust (per Part 2
//! sub-lane 6 constraint "must work on stable Rust"), this implementation
//! preserves the SEMANTIC of "every field carries a compile-time role tag"
//! by inspecting `#[role(<Variant>)]` attribute annotations at macro
//! expansion time. The generated `MLFeatureRole` impl carries each field's
//! role as a `const FIELD_ROLES: &'static [(&'static str, FeatureRole)]`,
//! which is observable at compile time and at runtime, and serves the same
//! firewall purpose. Operator-authorized translation (memory:
//! `project_pillar5_firewall_stable_rust.md`).

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Ident, LitStr};

const KNOWN_ROLES: &[&str] = &[
    "Localization",
    "Mechanistic",
    "CausalInformation",
    "Thermodynamic",
    "StabilityConsensus",
    "QualityControl",
    "ReportingOnly",
];

#[proc_macro_derive(MLFeature, attributes(role))]
pub fn derive_mlfeature(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_ident = &input.ident;
    let struct_name = struct_ident.to_string();
    let (impl_generics, type_generics, where_clause) =
        input.generics.split_for_impl();

    // Accept named-field structs and tuple structs. Reject unit structs
    // (no fields ⇒ trivially passes the firewall but provides no value)
    // and enums/unions.
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            Fields::Unnamed(unnamed) => &unnamed.unnamed,
            Fields::Unit => {
                return syn::Error::new(
                    input.span(),
                    "MLFeature cannot be derived on unit structs (no fields to firewall)",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new(
                input.span(),
                "MLFeature can only be derived on structs (named-field or tuple)",
            )
            .to_compile_error()
            .into();
        }
    };

    // Per-field accumulator. `field_name_str` is the canonical identifier
    // used in both diagnostics and the generated FIELD_ROLES table:
    //   - named fields:  the literal field name (e.g. `kcc_score`)
    //   - tuple fields:  synthetic `field_<N>` (e.g. `field_0`)
    let mut field_role_pairs: Vec<(String, String, Span)> = Vec::new();
    let mut errors: Vec<syn::Error> = Vec::new();

    for (idx, field) in fields.iter().enumerate() {
        let field_name_str = match &field.ident {
            Some(ident) => ident.to_string(),
            None => format!("field_{}", idx),
        };

        let role_attrs: Vec<&syn::Attribute> = field
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("role"))
            .collect();

        let role_attr = match role_attrs.as_slice() {
            [single] => *single,
            [] => {
                errors.push(syn::Error::new(
                    field.span(),
                    format!(
                        "field `{}` (of `{}`) is missing #[role(...)] annotation \
                         (Pillar 5 firewall requires every field to declare its FeatureRole)",
                        field_name_str, struct_name,
                    ),
                ));
                continue;
            }
            _ => {
                errors.push(syn::Error::new(
                    field.span(),
                    format!(
                        "field `{}` (of `{}`) has multiple #[role(...)] attributes; \
                         exactly one is required",
                        field_name_str, struct_name,
                    ),
                ));
                continue;
            }
        };

        let role_ident: Ident = match role_attr.parse_args() {
            Ok(id) => id,
            Err(e) => {
                errors.push(syn::Error::new(
                    role_attr.span(),
                    format!("expected `#[role(<Variant>)]`: {}", e),
                ));
                continue;
            }
        };

        let role_str = role_ident.to_string();
        if !KNOWN_ROLES.contains(&role_str.as_str()) {
            errors.push(syn::Error::new(
                role_ident.span(),
                format!(
                    "unknown FeatureRole `{}`; expected one of: {}",
                    role_str,
                    KNOWN_ROLES.join(", ")
                ),
            ));
            continue;
        }

        if role_str == "ReportingOnly" {
            errors.push(syn::Error::new(
                role_attr.span(),
                format!(
                    "field `{}` (of `{}`) has FeatureRole::ReportingOnly which is forbidden \
                     from ML training tensors (Pillar 5 firewall: §5 of the PRISM-4D \
                     Entangled Transform Blueprint)",
                    field_name_str, struct_name,
                ),
            ));
            continue;
        }

        field_role_pairs.push((field_name_str, role_str, field.span()));
    }

    if !errors.is_empty() {
        let mut combined = errors.remove(0);
        for e in errors {
            combined.combine(e);
        }
        return combined.to_compile_error().into();
    }

    let role_entries = field_role_pairs.iter().map(|(name_str, role, span)| {
        let name_lit = LitStr::new(name_str, *span);
        let role_ident = Ident::new(role, *span);
        quote_spanned! { *span =>
            (#name_lit, ::prism_nhs::feature_role::FeatureRole::#role_ident)
        }
    });

    let n = field_role_pairs.len();
    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics ::prism_nhs::feature_role::MLFeatureRole
            for #struct_ident #type_generics
            #where_clause
        {
            const FIELD_ROLES: &'static [(
                &'static str,
                ::prism_nhs::feature_role::FeatureRole,
            )] = &[
                #(#role_entries),*
            ];
        }

        impl #impl_generics #struct_ident #type_generics #where_clause {
            #[doc(hidden)]
            #[allow(dead_code)]
            const __ML_FEATURE_FIELD_COUNT: usize = #n;
        }
    };

    expanded.into()
}

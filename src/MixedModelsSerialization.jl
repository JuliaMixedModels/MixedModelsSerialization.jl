module MixedModelsSerialization

using GLM
using LinearAlgebra
using MixedModels
using SparseArrays
using StatsAPI
using StatsBase
using StatsFuns
using StatsModels

using JLD2
# using FileIO: File, @format_str

export MixedModelSummary, LinearMixedModelSummary
export save_summary, load_summary

"""
    MixedModelSummary{T} <: MixedModel{T}
    MixedModelSummary(m::LinearMixedModel)

Abstract type for a "summary" of a `MixedModel` with a reduced memory footprint.

Concrete subtypes do not the model matrices of a `MixedModel`
and thus will consume far less memory, especially for models with many
observations. However, they may store relevant parameters and derived
values for implementing common `StatsAPI`` methods that don't depend
on the original data.

See also [`LinearMixedModelSummary`](@ref)
"""
abstract type MixedModelSummary{T} <: MixedModel{T} end

# Not for use with rank deficient models
"""
    LinearMixedModelSummary{T<:AbstractFloat} <: MixedModelSummary{T}
    LinearMixedModelSummary(m::LinearMixedModel)

A "summary" of a `LinearMixedModel` with a reduced memory footprint.

This type does not store the model matrices of a `LinearMixedModel`
and thus will consume far less memory, especially for models with many
observations. Instead, the relevant entities for summarizing a model are
stored:
- fixed effects coefficients and associated variance-covariance matrix
- random effects covariances
- the θ vector and `OptSummary` used in optimization
- (conditional modes and variances are not currently stored)
- the log likelihood
- information about the model and residual degrees of freedom

Using these values, it is possible to provide implementations for many
but not all methods in `StatsAPI` and `MixedModels`.

!!! warning
    All field names and associated storage format should be considered private
    implementation details. Use appropriate methods to e.g. extract the
    log likelihood or the variance-covariance matrix. Stability of the
    internal structure is **not** guaranteed, even between non-breaking
    releases.
"""
struct LinearMixedModelSummary{T<:AbstractFloat} <: MixedModelSummary{T}
    β::Vector{T}
    cnames::Vector{String}
    se::Vector{T}
    θ::Vector{T}
    dims::NamedTuple{(:n, :p, :nretrms),NTuple{3,Int}}
    varcorr::VarCorr
    formula::FormulaTerm
    optsum::OptSummary{T}
    loglik::Real # we can compute deviance, AIC, AICc, BIC from this
    varcov::Matrix{T}
    pca::NamedTuple # MixedModels.PCA
end

MixedModelSummary(m::LinearMixedModel) = LinearMixedModelSummary(m)

function LinearMixedModelSummary(m::LinearMixedModel{T}) where {T}
    β = coef(m)
    cnames = coefnames(m)
    se = stderror(m)
    θ = m.θ
    dims = m.dims
    varcorr = VarCorr(m)
    formula = m.formula
    optsum = m.optsum
    loglik = loglikelihood(m)
    varcov = vcov(m)
    pca = MixedModels.PCA(m)

    return LinearMixedModelSummary{T}(β, cnames, se, θ, dims, varcorr, formula, optsum,
                                      loglik, varcov, pca)
end

# freebies: StatsAPI.aic, StatsAPI.aicc, StatsAPI.bic

StatsAPI.coef(mms::MixedModelSummary) = mms.β
StatsAPI.coefnames(mms::MixedModelSummary) = mms.cnames
function StatsAPI.coeftable(mms::MixedModelSummary)
    co = copy(coef(mms))
    se = copy(stderror(mms))
    z = co ./ se
    pvalue = 2 .* normccdf.(abs.(z))
    names = copy(coefnames(mms))

    return StatsBase.CoefTable(hcat(co, se, z, pvalue),
                               ["Coef.", "Std. Error", "z", "Pr(>|z|)"],
                               names,
                               4, # pvalcol
                               3)
end
StatsAPI.loglikelihood(mms::MixedModelSummary) = mms.loglik
StatsAPI.deviance(mms::MixedModelSummary) = -2 * loglikelihood(mms)
StatsAPI.stderror(mms::MixedModelSummary) = mms.se
function StatsAPI.dof(mms::MixedModelSummary)
    return mms.dims[:p] + length(mms.θ) + dispersion_parameter(mms)
end
StatsAPI.dof_residual(mms::MixedModelSummary) = nobs(mms) - dof(mms)
StatsAPI.nobs(mms::MixedModelSummary) = mms.dims[:n]
function StatsAPI.vcov(mms::MixedModelSummary; corr=false)
    vv = mms.varcov
    return corr ? StatsBase.cov2cor!(vv, stderror(mms)) : vv
end

StatsModels.formula(mms::MixedModelSummary) = mms.formula

# freebies: MixedModels.issingular

# MixedModels.fixef[names]
MixedModels.fnames(mms::MixedModelSummary) = keys(mms.pca)
MixedModels.lowerbd(mms::MixedModelSummary) = mms.optsum.lowerbd
MixedModels.VarCorr(mms::MixedModelSummary) = mms.varcorr
# MixedModels.nθ
# only stored on the covariance scale
# don't yet support doing this on the correlation scale
MixedModels.PCA(mms::MixedModelSummary) = mms.pca
function MixedModels.rePCA(mms::MixedModelSummary)
    return NamedTuple{keys(mms.pca)}(getproperty.(values(mms.pca), :cumvar))
end

# GLM.dispersion_parameter(mms::LinearMixedModelSummary)
# linear only
StatsAPI.islinear(mms::LinearMixedModelSummary) = true

function GLM.dispersion(mms::LinearMixedModelSummary, sqr::Bool=false)
    vc = VarCorr(mms)
    d = vc.s
    return sqr ? d * d : d
end
# fitted, residuals, leverage, etc -- full model
# ranefTables and condVarTables -- need the full model; sorry bud
# modelmatrix, etc -- yeah na

# TODO: show methods
# TODO: maybe FileIO.save?

function save_summary(filename, summary)
    return jldsave(filename; summary=summary)
end

function load_summary(filename)
    return jldopen(filename, "r") do file
        "summary" == only(keys(file)) ||
            error("Was expecting only find a summary, " *
                  "found $(collect(keys(dict)))")
        vv = file["summary"]
        vv isa MixedModelSummary ||
            error("Was expecting to find a MixedModelSummary, " *
                  "found $(typeof(vv))")
        return vv
    end
end

end # module

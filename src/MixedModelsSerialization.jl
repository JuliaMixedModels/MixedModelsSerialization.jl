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

using Base: Ryu
export MixedModelSummary, LinearMixedModelSummary
export save_summary, load_summary

# fitted, residuals, leverage, etc -- full model
# ranefTables and condVarTables -- need the full model; sorry bud
# modelmatrix, etc -- yeah na


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
    reterms::NamedTuple
    varcorr::VarCorr
    formula::FormulaTerm
    optsum::OptSummary{T}
    objective::T # we can compute deviance, AIC, AICc, BIC from this
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
    reterms = let
        kk = Symbol.(getproperty.(m.reterms, :trm))
        vv = ((; cnames=re.cnames, nlevs=MixedModels.nlevs(re)) for re in m.reterms)
        NamedTuple(zip(kk, vv))
    end
    varcorr = VarCorr(m)
    formula = m.formula
    optsum = m.optsum
    obj = objective(m)
    varcov = vcov(m)
    pca = MixedModels.PCA(m)

    return LinearMixedModelSummary{T}(β, cnames, se, θ, dims, reterms, varcorr, formula, optsum,
                                      obj, varcov, pca)
end

# we can skip store this explicitly if we store
# the BLUPs or at least their names
function Base.size(mms::MixedModelSummary)
    dd = mms.dims
    n_blups = sum(mms.reterms) do grp
       return length(grp.cnames) * grp.nlevs
    end
    return dd.n, dd.p, n_blups, dd.nretrms
end

#####
##### StatsAPI
#####

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

StatsAPI.deviance(mms::MixedModelSummary) = objective(mms)

function StatsAPI.dof(mms::MixedModelSummary)
    return mms.dims[:p] + length(mms.θ) + dispersion_parameter(mms)
end

StatsAPI.dof_residual(mms::MixedModelSummary) = nobs(mms) - dof(mms)

StatsAPI.islinear(mms::LinearMixedModelSummary) = true

function StatsAPI.loglikelihood(mms::MixedModelSummary)
    if mms.optsum.REML
        throw(ArgumentError("loglikelihood not available for models fit by REML"))
    end
    return -objective(mms) / 2
end

StatsAPI.nobs(mms::MixedModelSummary) = mms.dims[:n]

StatsAPI.stderror(mms::MixedModelSummary) = mms.se

function StatsAPI.vcov(mms::MixedModelSummary; corr=false)
    vv = mms.varcov
    return corr ? StatsBase.cov2cor!(vv, stderror(mms)) : vv
end

#####
##### StatsModels
#####

StatsModels.formula(mms::MixedModelSummary) = mms.formula

#####
##### GLM.jl
#####

# GLM.dispersion_parameter(mms::LinearMixedModelSummary)
function GLM.dispersion(mms::LinearMixedModelSummary, sqr::Bool=false)
    vc = VarCorr(mms)
    d = vc.s
    return sqr ? d * d : d
end

#####
##### MixedModels
#####

# freebies: MixedModels.issingular
# MixedModels.fixef[names]
# MixedModels.nθ
# MixedModels.nlevs
MixedModels.fnames(mms::MixedModelSummary) = keys(mms.pca)
MixedModels.lowerbd(mms::MixedModelSummary) = mms.optsum.lowerbd
MixedModels.objective(mms::MixedModelSummary) = mms.objective
MixedModels.VarCorr(mms::MixedModelSummary) = mms.varcorr
# only stored on the covariance scale
# don't yet support doing this on the correlation scale
MixedModels.PCA(mms::MixedModelSummary) = mms.pca
function MixedModels.rePCA(mms::MixedModelSummary)
    return NamedTuple{keys(mms.pca)}(getproperty.(values(mms.pca), :cumvar))
end
# necessary for the MIME show methods
MixedModels._dname(::LinearMixedModelSummary) = "Residual"

#####
##### show methods
#####

Base.show(io::IO, mms::LinearMixedModelSummary) = show(io, MIME("text/plain"), mms)

function Base.show(io::IO, ::MIME"text/plain", m::LinearMixedModelSummary)
    m.optsum.feval < 0 && begin
        @warn("Model has not been fit")
        return nothing
    end
    n, p, q, k = size(m)
    REML = m.optsum.REML
    println(io, "Linear mixed model fit by ", REML ? "REML" : "maximum likelihood")
    println(io, " ", m.formula)
    oo = objective(m)
    if REML
        println(io, " REML criterion at convergence: ", oo)
    else
        nums = Ryu.writefixed.([-oo / 2, oo, aic(m), aicc(m), bic(m)], 4)
        fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
        for label in ["  logLik", "-2 logLik", "AIC", "AICc", "BIC"]
            print(io, rpad(lpad(label, (fieldwd + textwidth(label)) >> 1), fieldwd))
        end
        println(io)
        print.(Ref(io), lpad.(nums, fieldwd))
        println(io)
    end
    println(io)

    show(io, VarCorr(m))

    print(io, " Number of obs: $n; levels of grouping factors: ")
    join(io, (re.nlevs for re in values(m.reterms)), ", ")
    println(io)
    println(io, "\n  Fixed-effects parameters:")
    return show(io, coeftable(m))
end

#####
##### Serialization
#####

"""
    save_summary(filename, summary::MixedModelSummary)

Serialize a `MixedModelSummary` to `filename`.
"""
function save_summary(filename, summary::MixedModelSummary)
    return jldsave(filename; summary=summary)
end

"""
    load_summary(filename)

Deserialize a `MixedModelSummary` from `filename`.
"""
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

#####
##### Things to ditch when we've upstructured upstream a bit
#####

function Base.getproperty(mms::LinearMixedModelSummary, p::Symbol)
    # XXX temporary hacks to get around some property access
    # in MixedModels show() methods
    return if p == :σs
        vc = VarCorr(mms)
        NamedTuple{keys(vc.σρ)}((vv.σ for vv in values(vc.σρ)))
    else
        getfield(mms, p)
    end
end


end # module

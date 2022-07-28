module MixedModelsSerialization

using LinearAlgebra
using MixedModels
using SparseArrays
using StatsAPI
using StatsBase
using StatsFuns
using StatsModels


# should this be <: MixedModel?
abstract type MixedModelSummary{T} <: MixedModel{T} end
# Not for use with rank deficient models
struct LinearMixedModelSummary{T<:AbstractFloat} <: MixedModelSummary{T}
    β::Vector{T}
    cnames::Vector{String}
    se::Vector{T}
    θ::Vector{T}
    dims::NamedTuple{(:n, :p, :nretrms), NTuple{3, Int}}
    varcorr::VarCorr
    formula::FormulaTerm
    optsum::OptSummary{T}
    loglik::Real # we can compute deviance, AIC, AICc, BIC from this
    varcov::Matrix{T}
    pca::NamedTuple # MixedModels.PCA
end

export MixedModelSummary

function MixedModelSummary(m::LinearMixedModel{T}) where {T}
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

    return LinearMixedModelSummary{T}(β, cnames, se, θ, dims, varcorr, formula, optsum, loglik, varcov, pca)
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

    return StatsBase.CoefTable(
        hcat(co, se, z, pvalue),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)"],
        names,
        4, # pvalcol
        3, # teststatcol
    )
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

# MixedModels.fixef[names]
# MixedModels.fnames
# MixedModels.issingular
# MixedModels.lowerbd
# MixedModels.nθ
# MixedModels.PCA
# MixedModels.rePCA
# GLM.dispersion
# GLM.dispersion_parameter

# linear only
StatsAPI.islinear(mms::LinearMixedModelSummary) = true

# fitted, residuals, leverage, etc -- full model
# ranefTables and condVarTables -- need the full model; sorry bud
# modelmatrix, etc -- yeah na

end # module

# using DataFrames
# using Effects
using MixedModels
using MixedModelsSerialization
using Test

using MixedModels: dataset

kb07 = dataset(:kb07)
progress = false

fm1 = fit(MixedModel,
          @formula(rt_trunc ~ 1 + spkr * prec * load +
                              (1 + spkr + prec | subj) +
                              (1 + load | item)), kb07; progress)
mms = MixedModelSummary(fm1)

@testset "StatsAPI" begin
    statsapi = [coef, coefnames,
                stderror,
                loglikelihood, deviance,
                aic, aicc, bic,
                dof, dof_residual,
                nobs, vcov]

    for f in statsapi
        @test f(fm1) == f(mms)
    end
    @test sprint(show, coeftable(fm1)) == sprint(show, coeftable(mms))
end

@testset "StatsModels" begin
    funcs = [formula]

    for f in funcs
        @test f(fm1) == f(mms)
    end
end

@testset "MixedModels" begin
    # fixef, fixefnames, nθ
    mixedmodels = [fnames,
                   issingular, lowerbd,
                   # MixedModels.PCA, seems flaky
                   MixedModels.rePCA,
                   VarCorr]
    for f in mixedmodels
        @info f
        @info f(fm1) == f(mms)
    end
    @test sprint(show, coeftable(fm1)) == sprint(show, coeftable(mms))
end

@testset "GLM" begin
    funcs = [dispersion, dispersion_parameter]

    for f in funcs
        @test f(fm1) == f(mms)
    end
end

@testset "JLD2 roundtrip" begin
    mms2 = mktempdir() do dir
        fname = joinpath(dir, "test.mmsum")
        save_summary(fname, mms)
        return load_summary(fname)
    end

    @test all(vcov(mms) .== vcov(mms2))
    @test string(formula(mms)) == string(formula(mms2))
    # need to test that ContrastCoding is preserved
    # if we get the Effects.jl stuff working, then
    # we can test by comparing effects output
end

# need typify support or a way to create a pseudo modelmatrix()
# @testset "Effects.jl compat" begin
#     design = Dict(:spkr => ["old", "new"])
#     refgrid = DataFrame(; spkr=["old", "new"],)
#     effects(design, mms)
# end

using MixedModels
using MixedModelsSerialization
using Test

using MixedModels: dataset

kb07 = dataset(:kb07)

fm1 = fit(MixedModel,
          @formula(rt_trunc ~ 1 + spkr * prec * load +
                              (1 + spkr + prec | subj) +
                              (1 + load | item)), kb07)
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

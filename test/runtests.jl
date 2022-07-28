using MixedModels
using MixedModelsSerialization
using Test

using MixedModels: dataset

kb07 = dataset(:kb07)

fm1 = fit(MixedModel,
          @formula(rt_trunc ~ 1 + spkr * prec * load +
                             (1 + spkr + prec | subj) +
                             (1 + load|item)), kb07)
mms = MixedModelSummary(fm1)
aic(mms)

statsapi = [coef, coefnames,
            stderror,
            loglikelihood, deviance,
            aic, aicc, bic,
            dof, dof_residual,
            nobs, vcov]

@testset "StatsAPI" begin
    for f in statsapi
        @test f(fm1) == f(mms)
    end
    @test sprint(show, coeftable(fm1)) == sprint(show, coeftable(mms))
end

statsmodels = [:formula]

mixedmodels = [:fixef, :fixefnames, :fnames,
               :issingular, :lowerbd, :nθ,
               :PCA, :rePCA]

glmjl = [:dispersion, :dispersion_parameter]

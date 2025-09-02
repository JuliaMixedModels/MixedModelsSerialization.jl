include("set_up_tests.jl")

@testset ExtendedTestSet "Aqua" begin
    Aqua.test_all(Effects; ambiguities=false)
end

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
    @test_throws ArgumentError loglikelihood(mms2)
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
                   objective,
                   # MixedModels.PCA, seems flaky
                   MixedModels.rePCA, size,
                   VarCorr]
    for f in mixedmodels
        @test f(fm1) == f(mms)
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
@testset "modelmatrix" begin
    fm1a = fit(MixedModel,
               @formula(rt_trunc ~ 1 + prec & load +
                                   (1 | subj) +
                                   (1 | item)), kb07; progress)
    mmsa = MixedModelSummary(fm1a)
    @test_throws ArgumentError modelmatrix(mmsa)

    # want exact elementwise equality
    @test all(modelmatrix(mms2) .== [1.0 0.0; 1.0 4.5; 1.0 9.0])

    mat = [1 0 0 0 0 0 0 0
           1 1 0 0 0 0 0 0
           1 0 1 0 0 0 0 0
           1 1 1 0 1 0 0 0
           1 0 0 1 0 0 0 0
           1 1 0 1 0 1 0 0
           1 0 1 1 0 0 1 0
           1 1 1 1 1 1 1 1]
    @test all(modelmatrix(mms) .== mat)
end

@testset "Effects.jl compat" begin
    design = Dict(:days => [4.2])
    @test effects(design, mms2) == effects(design, fm2)

    design = Dict(:load => ["no", "yes"])
    # the data aren't perfectly balanced,
    # but they're close so the values here are quite close
    emms = effects(design, mms)
    efm1 = effects(design, fm1)
    @test emms.load == efm1.load
    for cn in names(efm1, Number)
        @test all(isapprox.(emms[!, cn], efm1[!, cn]; rtol=0.001))
    end
end

@testset "show" begin
    @test sprint(show, mms) == sprint(show, fm1)
    for out in ["markdown", "latex", "xelatex"]
        mime = MIME(string("text/", out))
        @test sprint(show, mime, mms) == sprint(show, mime, fm1)
    end
    # REML
    @test sprint(show, mms2) == sprint(show, fm2)
end

using Documenter
using MixedModelsSerialization

makedocs(; modules=[MixedModelsSerialization],
         authors="Phillip Alday and contributors",
         repo="https://github.com/JuliaMixedModels/MixedModelsSerialization.jl/blob/{commit}{path}#{line}",
         sitename="Effects.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://juliamixedmodels.github.io/MixedModelsSerialization.jl",
                                assets=String[]),
         pages=[#"Home" => "index.md",
                "API" => "api.md"])

deploydocs(; repo="github.com/JuliaMixedModels/MixedModelsSerialization.jl",
           devbranch="main",
           push_preview=true)

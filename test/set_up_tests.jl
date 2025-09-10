using Aqua
using Effects
using MixedModels
using MixedModelsSerialization
using Test
using TestSetExtensions

using MixedModelsDatasets: dataset

kb07 = dataset(:kb07)
progress = false

fm1 = fit(MixedModel,
          @formula(rt_trunc ~ 1 + spkr * prec * load +
                              (1 + spkr + prec | subj) +
                              (1 + load | item)), kb07; progress)
mms = MixedModelSummary(fm1)

fm2 = fit(MixedModel,
          @formula(reaction ~ 1 + days + (1 | subj)),
          dataset(:sleepstudy);
          progress,
          REML=true)
mms2 = MixedModelSummary(fm2)

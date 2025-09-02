module MixedModelsSerializationEffectsExt

using DataFrames
using Effects
using MixedModels
using MixedModelsSerialization
using StatsBase

function Effects.effects!(reference_grid::DataFrame, model::LinearMixedModelSummary; kwargs...)
    # we don't want to use the MixedModel specific version since we don't store all the things
   return @invoke effects!(reference_grid::DataFrame, model::RegressionModel; kwargs...)
end


end # module

# trms = mms.formula.rhs[1].terms

_vals(::InterceptTerm, ::TupleTerm) = Float64[] # [1.]
_vals(ct::ContinuousTerm, ::TupleTerm) = [ct.min, ct.mean, ct.max]
_vals(ct::CategoricalTerm, ::TupleTerm) = ct.contrasts.levels

# we could probably figure this out it's an "exotic" edge case
# and generally not advisable from a statistical perspective, so NOPE
# XXX if you add support here, you need to change _names()
function _vals(it::InteractionTerm, rhs::TupleTerm)
    trms = string.(filter(x -> !isa(x, InteractionTerm), rhs))
    for tt in it.terms
        string(tt) in trms ||
            throw(ArgumentError("`modelmatrix` is not supported on models " *
                                "with an interaction without a corresponding" *
                                " main effect."))
    end
    return Float64[]
end

function _names(rhs::TupleTerm)
    return Symbol.(filter(x -> !isa(x, Union{InterceptTerm,InteractionTerm}), rhs))
end

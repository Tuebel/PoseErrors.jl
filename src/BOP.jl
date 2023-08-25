# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    match_errors
Greedily match the poses of one object class in a scene to the ground truth poses.
Supply the errors for a single parameter like τ in VSD.
* `scores`: `[score(est) for est in estimates]`
* `errors_per_est`: `[[error(est, gt) for gt in annotations] for es in estimates]`

Returns a vector of matched errors, defaults to infinity errors to handle less estimates than ground truth annotations.
"""
function match_errors(scores::AbstractVector{<:Real}, errors_per_est::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    # Match length(gt) errors
    matched_gt_errors = fill(T(Inf), length(first(errors_per_est)))
    # match each pose at most once
    matched_gt_indices = Int[]
    # greedily: sort estimates according to their score, highest score first
    sorted_est_indices = sortperm(scores; lt=Base.isgreater)
    sorted_est_errors = errors_per_est[sorted_est_indices]
    # match at most length(gt) errors
    n_matches = min(length(sorted_est_indices), length(matched_gt_errors))
    est_errors = sorted_est_errors[1:n_matches]
    # greedily select the best gt
    for gt_errors in est_errors
        gt_indices = sortperm(gt_errors; lt=Base.isless)
        for gt_idx in gt_indices
            if !(gt_idx in matched_gt_indices)
                push!(matched_gt_indices, gt_idx)
                matched_gt_errors[gt_idx] = gt_errors[gt_idx]
                break
            end
        end
    end
    matched_gt_errors
end

"""
    match_bop19_errors(scores, errors_per_est)
If for each estimate - ground truth combination multiple errors have been calculated, e.g. in VSD BOP for θ ∈ 0:.05:.5.
`errors_per_est` = `[[[err_for_θ for err_for_θ in gt] for gt in est] for est in estimates]`
"""
function match_bop19_errors(scores, errors_per_est)
    # Matrix with dimss [θ, ground truth, estimate]
    err_θ_gt_est = stack(stack(errors_per_est))
    matched_θ_gt = map(eachslice(err_θ_gt_est; dims=1)) do err_gt_est
        vec_gt_est = [Vector(col) for col in eachcol(err_gt_est)]
        match_errors(scores, vec_gt_est)
    end
    # Results are stored in data frame with gt as rows not θ
    [Vector(col) for col in eachrow(stack(matched_θ_gt))]
end

"""
    threshold_errors(matched_errors, θ)
Compare the matched errors to a threshold or a range / vector of thresholds.
Returns the number of `[correct poses, annotated poses]`
"""
function threshold_errors(matched_errors, θ)
    correct = matched_errors' .< θ
    [sum(correct), length(correct)]
end

"""
    recall([n_correct,n_labels]::AbstractVector)
The fraction of annotated object instances, for which a correct pose is estimated, is referred to as recall. 
Poses are considered correct for `error < threshold`.
"""
function recall(correct_and_labels::AbstractVector{<:Real}...)
    cumulative = sum(correct_and_labels)
    cumulative[1] / cumulative[2]
end

# BOP evaluation
const BOP19_THRESHOLDS = 0.05:0.05:0.5
const BOP_δ = 0.015
const ITODD_δ = 0.005
const BOP18_τ = 0.02
const BOP18_θ = 0.3
const ADDS_θ = 0.1

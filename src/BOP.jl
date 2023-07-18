# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# TODO I can vectorize VSD of multiple poses -> [[tau1_pose1, tau1_pose2,...],[tau2_pose1, tau2_pose2,...],[tau3_...], ...]

"""
    match_errors
Greedily match the poses of one object class in a scene to the ground truth poses.
Only a single τ is considered in this method.
* `scores`: `[score(est) for est in estimates]`
* `errors_per_gt`: `[[error(est, gt) for est in estimates] for gt in annotations]`

Returns a vector of matched errors, defaults to infinity errors to handle less estimates than ground truth annotations.
"""
function match_errors(scores::AbstractVector{<:Real}, errors_per_gt::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    matched_errors = fill(T(Inf), length(errors_per_gt))
    # match each pose at most once
    matched_indices = Vector{Int}()
    # greedily: Start with highest score
    sorted_indices = sortperm(scores; lt=Base.isgreater)
    for (gt_idx, errors) in enumerate(errors_per_gt)
        # greedily
        sorted_errors = errors[sorted_indices]
        for (err_idx, error) in enumerate(sorted_errors)
            if !(err_idx in matched_indices)
                push!(matched_indices, err_idx)
                matched_errors[gt_idx] = error
                break
            end
        end
    end
    matched_errors
end

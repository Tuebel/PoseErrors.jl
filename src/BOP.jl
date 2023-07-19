# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    match_errors
Greedily match the poses of one object class in a scene to the ground truth poses.
Supply the errors for a single parameter like τ in VSD.
* `scores`: `[score(est) for est in estimates]`
* `errors_per_gt`: `[[error(est, gt) for est in estimates] for gt in annotations]`

Returns a vector of matched errors, defaults to infinity errors to handle less estimates than ground truth annotations.
"""
function match_errors(scores::AbstractVector{<:Real}, errors_per_gt::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    matched_errors = fill(T(Inf), length(errors_per_gt))
    # match each pose at most once
    matched_indices = Int[]
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

# TODO Calc errors_per_gt and save them to a file... DataFrame[:img_id, :gt_id]?

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

# TODO remove only compatibility of current test set
"""
    average_recall(errors, thresholds)
The fraction of annotated object instances, for which a correct pose is estimated, is referred to as recall. 
Poses are considered correct for `error < threshold`.
"""
average_recall(errors, thresholds) = recall(threshold_errors(errors, thresholds))


# TODO Diss: if matching are known: accuracy = recall. Should I still use the greedy error matching? Otherwise results might differ between BOP benchmark and my own tests. RFID never really worked, thus it is hard to argue that the matchings will be available.

# BOP evaluation
const BOP19_THRESHOLDS = 0.05:0.05:0.5
const BOP_δ = 0.015
const ITODD_δ = 0.005
const BOP_18_τ = 0.02
const BOP_18_θ = 0.3

# BUG This is not the recall but the accuracy of a single prediction. Missing the number of targets, i.e. how many poses should have been detected.

# NOTE in practice this means: For each tau and theta, the estimated poses are matched to the (>= 10% visible) ground truth poses. If the match was successful, i.e. error < theta, the pose is considered correct. For each GT pose at most one estimate is matched. Also each estimated pose is matched at most once. Matching is eagerly based on the confidence score.

# TODO Move BOP recalls to BOP.jl

"""
    bop19_recalls(distance_context, cv_camera, mesh, measured_depth, estimate, ground_truth, [δ=0.015])
Conveniently evaluate the average recalls for ADD-S, MDD-S and VSD using the BOP19 thresholds.
Returns tuple of recalls `(adds, mdds, vsd)`.
δ is used as tolerance for the visibility masks, ITODD uses δ=ITODD_δ=0.005.
"""
function bop19_recalls(distance_context::OffscreenContext, cv_camera::CvCamera, mesh::Mesh, measured_depth::AbstractMatrix, estimate::Pose, ground_truth::Pose, δ=BOP_δ)
    points = mesh.position
    diameter = model_diameter(points)
    # ADDS / MDDS
    gt_affine, es_affine = AffineMap.((ground_truth, estimate,))
    adds_err = adds_error(points, es_affine, gt_affine)
    mdds_err = mdds_error(points, es_affine, gt_affine)
    adds, mdds = distance_recall_bop19.(diameter, (adds_err, mdds_err))
    # VSD
    vsd = bop19_vsd_recall(distance_context, cv_camera, mesh, diameter, measured_depth, estimate, ground_truth, δ)
    return (adds, mdds, vsd)
end

"""
    vsd_bop19_recall(distance_context, cv_camera, mesh, diameter, measured_dist, es_pose, gt_pose, [δ=0.015])
Conveniently evaluate the average recall for VSD using the BOP19 thresholds.
Provide a pre-calculated diameter since this would be the bottleneck of an otherwise parallelized implementation.
δ is used as tolerance for the visibility masks, ITODD uses δ=ITODD_δ=0.005.
"""
function bop19_vsd_recall(distance_context::OffscreenContext, cv_camera::CvCamera, mesh::Mesh, diameter, measured_dist::AbstractMatrix, es_pose::Pose, gt_pose::Pose, δ=BOP_δ)
    measured_dist = same_device(distance_context.render_data, measured_dist)
    τ = same_device(measured_dist, diameter * BOP19_THRESHOLDS)
    vsd_err = vsd_error(distance_context, cv_camera, mesh, measured_dist, es_pose, gt_pose, δ, τ)
    # TODO BOP only considers objects with at least 10% visibility
    # Run on CPU
    discrepancy_recall_bop19(Array(vsd_err))
end

# Move from device "from" to device "to"
same_device(to::CuArray, from::AbstractArray) = CuArray(from)
same_device(to::Array, from::AbstractArray) = Array(from)

"""
    distance_recall_bop18(diameter, errors)
For distance based metrics like ADD(S), MDD(S) or MSSD.
According to (Hinterstoisser et al. 2012) a pose is considered correct if `error < 0.1 object_diameter`.
"""
distance_recall_bop18(diameter, errors) = average_recall(errors, 0.1 * diameter)

"""
    distance_recall_bop18(diameter, errors)
For distance based metrics like ADD(S), MDD(S) or MSSD.
According to BOP19, a pose is considered correct if `error < diameter * threshold` where `threshold ∈ 0.05:0.05:0.5`.
"""
distance_recall_bop19(diameter, distances) = average_recall(distances, BOP19_THRESHOLDS * diameter)

"""
    discrepancy_recall_bop19(errors)
For discrepancy based methods like (V)SD.
According to BOP18, a pose is considered correct if `error < threshold` where `threshold ∈ 0.05:0.05:0.5`.
"""
discrepancy_recall_bop18(discrepancies) = average_recall(discrepancies, 0.3)

"""
    discrepancy_recall_bop19(errors)
For discrepancy based methods like (V)SD.
According to BOP19, a pose is considered correct if `error < 0.3`.
"""
discrepancy_recall_bop19(errors) = average_recall(errors, BOP19_THRESHOLDS)

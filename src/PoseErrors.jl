module PoseErrors

export add_error
export adds_error
export mdds_error

# Geometry
using CoordinateTransformations
using Distances
using GeometryBasics
using Rotations
using StaticArrays

# Matching points and calculating distances ADD-S & MDD-S
using NearestNeighbors
using Statistics

# For projection based methods
using SciGL

# Point Distance Metrics

"""
    add_error(points, estimate, ground_truth)
Average Distance of Model Points for objects with no indistinguishable views (Hinterstoisser et al. 2012).
Reimplementation of [https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py](BOP-toolkit).
"""
add_error(points, estimate, ground_truth) = mean(model_point_distances(points, estimate, ground_truth))

"""
    adds_error(points, estimate, ground_truth)
Average Distance of Model Points for objects with indistinguishable views (Hinterstoisser et al. 2012).
Also known as ADD-S or ADI.
Reimplementation of [https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py](BOP-toolkit).
"""
adds_error(points, estimate, ground_truth) = mean(nearest_neighbor_distances(points, estimate, ground_truth))

"""
    mdds_error(points, estimate, ground_truth)
Maximum Distance of Model Points for objects with indistinguishable views.
Adaption of the ADD-S error to avoid higher frequency surface features and provide a better indicator for grasp success.
Compared to the Maximum Symmetry-Aware Surface Distance (MSSD) used in the BOP challenge, this method avoids having to define / identify symmetries explicitly.
"""
mdds_error(points, estimate, ground_truth) = maximum(nearest_neighbor_distances(points, estimate, ground_truth))

"""
    nearest_neighbor_distances(points, estimate, ground_truth)
Returns the distance of each ground truth point to it's corresponding nearest neighbor in the estimates.
"""
function nearest_neighbor_distances(points, estimate, ground_truth)
    es_points = transform_points(points, estimate)
    gt_points = transform_points(points, ground_truth)

    tree = KDTree(es_points, Euclidean())
    _, distances = nn(tree, gt_points)
    return distances
end

"""
    model_point_distances(points, estimate, ground_truth)
Returns the distance of each ground truth model point to it's corresponding model point in the estimates.
"""
function model_point_distances(points, estimate, ground_truth)
    es_points = transform_points(points, estimate)
    gt_points = transform_points(points, ground_truth)
    colwise(Euclidean(), es_points, gt_points)
end

"""
    transform_points(points, pose)
Returns an AbstractVector{<:SVector} which can be processed by NearestNeighbors.jl
"""
transform_points(points::AbstractVector{<:AbstractVector}, pose::AffineMap) = pose.(SVector.(points))
transform_points(points::AbstractMatrix, pose) = transform_points([SVector{3}(x) for x in eachcol(points)], pose)

# Projection / Rendering Based Metrics
# TEST

"""
    visible_surface_discrepancy(depth_context, estimate, ground_truth, measurement, δ, τ)
Calculate the VSD according to [BOP19](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/).
δ is used as tolerance for the visibility masks and τ is the misalignment tolerance.
"""
function visible_surface_discrepancy(depth_context::OffscreenContext, estimate::Scene, ground_truth::Scene, measurement::AbstractArray, δ, τ)
    es_img, gt_img = draw_distance(depth_context, estimate, ground_truth)
    visible_es, visible_gt = pixel_visible.(es_img, measurement, δ), pixel_visible.(gt_img, measurement, δ)
    surface_discrepancy(visible_es, visible_gt, ground_truth, τ)
end

"""
    pixel_visible(render, measurement, δ)
If the rendered pixel is in front of the measurement with a tolerance of δ, the render distance is returned.
Otherwise, zero is returned.
"""
pixel_visible(render, measurement, δ) = render <= measurement + δ ? render : zero(render)

"""
    surface_discrepancy(depth_context, estimate, ground_truth, τ)
Calculate the surface discrepancy according to [BOP19](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/) by rendering two scenes.
τ is the misalignment tolerance, for the VSD, the images must have been masked.
"""
function surface_discrepancy(depth_context::OffscreenContext, estimate::Scene, ground_truth::Scene, τ)
    es_img, gt_img = draw_distance(depth_context, estimate, ground_truth)
    surface_discrepancy(es_img, gt_img, ground_truth, τ)
end

"""
    surface_discrepancy(estimate, ground_truth, τ)
Calculate the surface discrepancy according to [BOP19](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/) for two rendered distance images.
τ is the misalignment tolerance.
For the calculation of the VSD, the images must have been masked.
"""
function surface_discrepancy(estimate::AbstractArray, ground_truth::AbstractArray, τ)
    union_count = sum(@. estimate > 0 || ground_truth > 0)
    # early stopping and no division by zero
    if iszero(union_count)
        return one(eltype(estimate))
    end
    costs = discrepancy_cost.(estimate, ground_truth, τ)
    # Average of the costs for the union pixels
    sum(costs) / union_count
end

"""
    discrepancy_cost(dist_a, dist_b, τ)
Step function costs according to [BOP19](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/).
Returns true (1) adding cost and false (0) for no cost
"""
function discrepancy_cost(dist_a, dist_b, τ)
    a_valid, b_valid = (dist_a, dist_b) .> 0
    if !a_valid && !b_valid
        # Do not add any cost if not part of union
        return false
    elseif a_valid ⊻ b_valid
        # Part of intersection. Cost if misalignment tolerance is violated
        return abs(dist_b - dist_a) > τ
    else
        # Not part of intersection -> always cost
        return true
    end
end

"""
    draw_distance(depth_context, estimate, ground_truth)
Returns a tuple of the depth images for the estimate and ground truth.
"""
function draw_distance(distance_context::OffscreenContext, estimate::Scene, ground_truth::Scene)
    if last(size(distance_context) > 1)
        imgs = draw(distance_context, [estimate, ground_truth])
        es_img = @view(imgs[:, :, 1])
        gt_img = @view(imgs[:, :, 2])
    else
        # Buffer is overwritten → copy it
        es_img = copy(draw(distance_context, estimate))
        gt_img = draw(distance_context, ground_truth)
    end
    es_img, gt_img
end

end # module PoseErrors

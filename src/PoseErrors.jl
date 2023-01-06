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

"""
    add_error(points, ground_truth, estimate)
Average Distance of Model Points for objects with no indistinguishable views (Hinterstoisser et al. 2012).
Reimplementation of [https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py](BOP-toolkit).
"""
add_error(points, ground_truth, estimate) = mean(model_point_distances(points, ground_truth, estimate))

"""
    adds_error(points, ground_truth, estimate)
Average Distance of Model Points for objects with indistinguishable views (Hinterstoisser et al. 2012).
Also known as ADD-S or ADI.
Reimplementation of [https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py](BOP-toolkit).
"""
adds_error(points, ground_truth, estimate) = mean(nearest_neighbor_distances(points, ground_truth, estimate))

"""
    mdds_error(points, ground_truth, estimate)
Maximum Distance of Model Points for objects with indistinguishable views.
Adaption of the ADD-S error to avoid higher frequency surface features and provide a better indicator for grasp success.
Compared to the Maximum Symmetry-Aware Surface Distance (MSSD) used in the BOP challenge, this method avoids having to define / identify symmetries explicitly.
"""
mdds_error(points, ground_truth, estimate) = maximum(nearest_neighbor_distances(points, ground_truth, estimate))

"""
    nearest_neighbor_distances(points, ground_truth, estimate)
Returns the distance of each ground truth point to it's corresponding nearest neighbor in the estimates.
"""
function nearest_neighbor_distances(points, ground_truth, estimate)
    gt_points = transform_points(points, ground_truth)
    es_points = transform_points(points, estimate)

    tree = KDTree(es_points, Euclidean())
    _, distances = nn(tree, gt_points)
    return distances
end

"""
    model_point_distances(points, ground_truth, estimate)
Returns the distance of each ground truth model point to it's corresponding model point in the estimates.
"""
function model_point_distances(points, ground_truth, estimate)
    gt_points = transform_points(points, ground_truth)
    es_points = transform_points(points, estimate)
    colwise(Euclidean(), gt_points, es_points)
end

"""
    transform_points(points, pose)
Returns an AbstractVector{<:SVector} which can be processed by NearestNeighbors.jl
"""
transform_points(points::AbstractVector{<:AbstractVector}, pose::AffineMap) = pose.(SVector.(points))
transform_points(points::AbstractMatrix, pose) = transform_points([SVector{3}(x) for x in eachcol(points)], pose)

end # module PoseErrors

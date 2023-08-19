# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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
transform_points(points::AbstractVector{<:AbstractVector}, pose::AffineMap) = pose.(convert_points(points))
transform_points(points::AbstractMatrix, pose) = transform_points(convert_points(points), pose)

"""
    model_diameter(points)
Calculate the maximum distance of two points in the model which is the diameter of the object.
"""
function model_diameter(points)
    # Distances.jl does not like the StaticArray implementation of GeometryBasics.jl
    points = convert_points(points)
    # Type stable zero initialization
    diameter = evaluate(Euclidean(), first(points), first(points))
    for (idx, point_a) in enumerate(points)
        # Previous and current point do not need to be compared again
        for point_b in drop(points, idx)
            dist = evaluate(Euclidean(), point_a, point_b)
            diameter = dist > diameter ? dist : diameter
        end
    end
    return diameter
end

"""
    convert_points(points)
Support different point formats: vectors of points/vectors, meshes, matrices [point,n_points]
"""
convert_points(points::AbstractVector{<:AbstractVector}) = SVector{3}.(points)
convert_points(points::AbstractMatrix) = SVector{3}.(eachcol(points))
convert_points(points::Mesh) = convert_points(points.position)

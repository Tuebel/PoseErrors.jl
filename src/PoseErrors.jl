module PoseErrors

export add_error
export adds_error
export mdds_error
export vsd_error

export depth_to_distance
export model_diameter
export surface_discrepancy
export visibility_es, visibility_gt

export BOP19_THRESHOLDS, BOP_δ, ITODD_δ
export bop19_recalls, bop19_vsd_recall
export discrepancy_recall_bop18
export discrepancy_recall_bop19
export distance_recall_bop18
export distance_recall_bop19

# BOP dataset evaluation
export match_errors

# Geometry
using CoordinateTransformations
using Distances
using GeometryBasics: Mesh
using LinearAlgebra
using Rotations
using StaticArrays

# Matching points and calculating distances ADD-S & MDD-S
using Base.Iterators: drop
using NearestNeighbors
using Statistics

# For projection based methods
using Accessors
using CUDA
using SciGL

# BOP dataset evaluation
include("BOP.jl")

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
convert_points(points::AbstractMatrix) = [SVector{3}(x) for x in eachcol(points)]
convert_points(points::Mesh) = convert_points(points.position)

# Projection / Rendering Based Metrics

"""
    vsd_error(distance_context, cv_camera, mesh, measured_dist, es_poses, gt_pose, [δ=0.015, τ=0.02])
Calculate the visible surface discrepancy according to [BOP19](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/).
Note that the `distance_context` and `measured_dist` must be / produce a distance map not a depth image.
δ is used as tolerance for the visibility masks and τ is the misalignment tolerance.
Multiple estimated poses `es_poses` as well as a range of `τ` might be provided which results in a vector of `n_taus` vectors of n_poses. 

Default values `δ=15mm` and `τ=20mm` are the ones used in BOP18, BOP19 and later use a range of `τ=0.05:0.05:0.5` of the object diameter.
The BOP18 should only be used for parameter tuning and not evaluating the final scores.
"""
function vsd_error(distance_context::OffscreenContext, cv_camera::CvCamera, mesh::Mesh, measured_dist::AbstractMatrix, es_poses, gt_pose::Pose, δ::Real=BOP_δ, τ::Real=BOP_18_τ)
    camera = Camera(cv_camera)
    model = upload_mesh(distance_context, mesh)

    gt_scene = scene(camera, model, gt_pose)
    gt_dist = draw(distance_context, gt_scene) |> copy
    gt_visible = visibility_gt.(gt_dist, measured_dist, δ)
    gt_masked = gt_visible .* gt_dist

    es_scenes = scene(camera, model, es_poses)
    es_dist = draw(distance_context, es_scenes)
    es_masked = @. visibility_es(es_dist, measured_dist, δ, gt_visible) * es_dist

    surface_discrepancy(es_masked, gt_masked, τ)
end

"""
    scene(camera, model, poses)
Create a single scene or a vector of scenes for the given poses.
"""
scene(camera, model, pose::Pose) = Scene(camera, [@set model.pose = pose])
scene(camera, model, poses::AbstractVector{<:Pose}) = [scene(camera, model, pose) for pose in poses]

"""
    visibility_gt(rendered_dist, measured_dist, δ)
If `rendered_dist` is in front of `measured_dist` with a tolerance distance of `δ`, the pixel is considered visible.
If `measured_dist` is invalid, the pixel is also considered visible.
However, for both cases `rendered_dist` must be valid, i.e. greater than 0.
"""
visibility_gt(rendered_dist, measured_dist, δ) = pixel_valid(rendered_dist) & (no_depth(measured_dist) | surface_visible(rendered_dist, measured_dist, δ))

"""
    visibility_mask_es(rendered_dist, measured_dist, δ, gt_visible)
If `rendered_dist` is in front of `measured_dist` with a tolerance distance of `δ`, the pixel is considered visible.
If `measured_dist` is invalid, the pixel is also considered visible.
Also, the pixels of the ground truth visibility mask `gt_visible` are considered visible.
However, for both all previous cases `rendered_dist` must be valid, i.e. greater than 0.
"""
visibility_es(rendered_dist, measured_dist, δ, gt_visible) = pixel_valid(rendered_dist) & (no_depth(measured_dist) | surface_visible(rendered_dist, measured_dist, δ) | gt_visible)

# For broadcasting - large kernel is more efficient than many small
# Also improved reusability in visibility_gt & visibility_es
surface_visible(rendered_dist, measured_dist, δ) = rendered_dist <= (measured_dist + δ)
no_depth(measured_dist) = measured_dist <= 0
pixel_valid(rendered_dist) = rendered_dist > 0

"""
    surface_discrepancy(es_dist, gt_dist, τ)
Calculate the surface discrepancy according to [BOP19](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/) for two rendered distance images.
τ is the misalignment tolerance and can be given as a vector to improve performance.
For the calculation of the VSD, the images must have been masked.
"""
function surface_discrepancy(es_dist::AbstractArray, gt_dist::AbstractArray, τ::Real)
    # VSD is a modification of the CoU metric with additional costs for large distances 
    union = dropsum(@. es_dist > 0 || gt_dist > 0; dims=(1, 2))
    complement = dropsum(discrepancy_cost.(es_dist, gt_dist, τ); dims=(1, 2))
    complement_over_union = complement ./ union
    # union == 0 → no pixel rendered → pose out of view → definitely wrong  → return limit
    inf_to_one.(complement_over_union)
end

"""
    dropsum(x; dims)
Combines sum and dropdims along dims.
"""
dropsum(x; dims) = dropdims(sum(x; dims=dims); dims=dims)

"""
    inf_to_one(x)
If x is infinity the one is returned, x otherwise
"""
inf_to_one(x) = isinf(x) ? one(x) : x

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
        # Part of the complement → always cost
        return true
    else
        # Part of intersection → cost if misalignment tolerance is violated
        return abs(dist_b - dist_a) > τ
    end
end

"""
    depth_to_distance(I::CartesianIndex, z, f_x, f_y, c_x, c_y, s)
Given the pixel position as cartesian index I and the corresponding depth value, the distance is calculated by reprojecting the pixel to 3D via the OpenCV camera parameters.
"""
function depth_to_distance(I::CartesianIndex, z, f_x, f_y, c_x, c_y, s)
    # OpenCV starts uses 0 based indexing
    u, v = Tuple(I) .- 1
    # Inverse of projection from https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    y = (v - c_y) * z / f_y
    x = ((u - c_x) * z - s * y) / f_x
    LinearAlgebra.norm((x, y, z))
end

"""
    depth_to_distance(depth_img, cv_camera::CvCamera)
Calculates the distance image from the depth image using the OpenCv camera parameters for reprojecting the pixel to 3D.
"""
depth_to_distance(depth_img, cv_camera::CvCamera) =
    depth_to_distance.(CartesianIndices(depth_img), depth_img, cv_camera.f_x, cv_camera.f_y, cv_camera.c_x, cv_camera.c_y, cv_camera.s)

# Performance Scores

"""
    average_recall(errors, thresholds)
The fraction of annotated object instances, for which a correct pose is estimated, is referred to as recall. 
Poses are considered correct for `error < threshold`.
"""
average_recall(errors, thresholds) = mean([e < θ for e in errors, θ in thresholds])

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

end # module PoseErrors

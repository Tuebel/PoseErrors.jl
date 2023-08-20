# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


"""
    vsd_error(distance_context, cv_camera, mesh, measured_dist, es_poses, gt_pose, [δ=0.015, τ=0.02])
Calculate the visible surface discrepancy according to [BOP19](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/).
Note that the `distance_context` and `measured_dist` must be / produce a distance map not a depth image.
δ is used as tolerance for the visibility masks and τ is the misalignment tolerance.
Multiple estimated poses `es_poses` as well as a range of `τ` might be provided which results in a vector of `n_taus` vectors of n_poses. 

Default values `δ=15mm` and `τ=20mm` are the ones used in BOP18, BOP19 and later use a range of `τ=0.05:0.05:0.5` of the object diameter.
The BOP18 should only be used for parameter tuning and not evaluating the final scores.
"""
function vsd_error(distance_context::OffscreenContext, cv_camera::CvCamera, mesh::Mesh, measured_dist::AbstractMatrix, es_poses, gt_pose::Pose, δ=BOP_δ, τ=BOP_18_τ)
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
τ is the misalignment tolerance.
Supply τ as a vector to improve performance since the same renderings can be reused.
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

surface_discrepancy(es_dist::AbstractArray, gt_dist::AbstractArray, τ::AbstractVector{<:Real}) = [surface_discrepancy(es_dist, gt_dist, x) for x in τ]

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
    reproject_3D(u, v, z, f_x, f_y, c_x, c_y, s)
Supply the pixel 2D position as Cartesian index in julia convention (starts at 1) and the corresponding depth value. The 3D coordinates are calculated via the OpenCV camera parameters.
"""
function reproject_3D(u, v, z, f_x, f_y, c_x, c_y, s)
    # OpenCV uses 0 based indexing
    u, v = (u, v) .- 1
    # Inverse of projection from https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    y = (v - c_y) * z / f_y
    x = ((u - c_x) * z - s * y) / f_x
    [x, y, z]
end

reproject_3D(u, v, z, camera::CvCamera) = reproject_3D(u, v, z, camera.f_x, camera.f_y, camera.c_x, camera.c_y, camera.s)

"""
    depth_to_distance(I::CartesianIndex, z, f_x, f_y, c_x, c_y, s)
Given the pixel position as cartesian index I and the corresponding depth value, the distance is calculated by reprojecting the pixel to 3D via the OpenCV camera parameters.
"""
depth_to_distance(I::CartesianIndex, z, camera::CvCamera) = LinearAlgebra.norm(reproject_3D(Tuple(I)..., z, camera))

"""
    depth_to_distance(depth_img, cv_camera::CvCamera)
Calculates the distance image from the depth image using the OpenCv camera parameters for reprojecting the pixel to 3D.
"""
depth_to_distance(depth_img, cv_camera::CvCamera) =
    depth_to_distance.(CartesianIndices(depth_img), depth_img, Ref(cv_camera))

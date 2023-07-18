# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using CoordinateTransformations
using PoseErrors
using Rotations
using SciGL
using Statistics
using Test

points = rand(3, 1_000)
pose0 = Translation(1.0, 0, 0) ∘ LinearMap(RotXYZ(0.5, 0, 0))
pose1 = LinearMap(RotXYZ(0.1, 0, 0)) ∘ pose0

# Test whether the errors strictly increase from first to last
function error_increases(errors::AbstractVector{T}) where {T}
    init = typemin(T)
    for e in errors
        if e > init
            init = e
        else
            return false
        end
    end
    return true
end

@testset "Nearest neighbor distances" begin
    # Sanity check of nearest_neighbor_distances: Same pose should have zero as distance, different a Euclidean distance > 0
    dists = @inferred PoseErrors.nearest_neighbor_distances(points, pose0, pose0)
    @test mapreduce(iszero, &, dists)
    dists = @inferred PoseErrors.nearest_neighbor_distances(points, pose0, pose1)
    @test mapreduce(x -> x > 0, &, dists)
end

@testset "Model diameter" begin
    diameter = @inferred model_diameter(points)
    # For 1_000 random points, diameter will be close to sqrt(3)
    @test sqrt(3) - 0.2 < diameter < sqrt(3)
end

@testset "ADD" begin
    add0 = @inferred add_error(points, pose0, pose0)
    @test add0 == 0
    add1 = @inferred add_error(points, pose0, pose1)
    @test add1 > 0
end

@testset "ADD-S" begin
    add_s0 = @inferred adds_error(points, pose0, pose0)
    @test add_s0 == 0
    add_s1 = @inferred adds_error(points, pose0, pose1)
    @test add_s1 > 0
end

@testset "MDD-S" begin
    mdd_s0 = @inferred mdds_error(points, pose0, pose0)
    @test mdd_s0 == 0
    mdd_s1 = @inferred mdds_error(points, pose0, pose1)
    @test mdd_s1 > 0
end

# Comparison of the bounds
@testset "Lower bounds of ADD, ADD-S, MDD-S" begin
    add1 = @inferred add_error(points, pose0, pose1)
    add_s1 = @inferred adds_error(points, pose0, pose1)
    mdd_s1 = @inferred mdds_error(points, pose0, pose1)
    @test add1 >= add_s1
    @test mdd_s1 >= add_s1
end

# Surface discrepancy
WIDTH, HEIGHT, DEPTH = 640, 480, 10
cv_camera = CvCamera(WIDTH, HEIGHT, 1.2 * WIDTH, 1.2 * HEIGHT, WIDTH / 2, HEIGHT / 2)
camera = cv_camera |> Camera
cube_path = joinpath(dirname(pathof(SciGL)), "..", "examples", "meshes", "cube.obj")
cube_scale = Scale(0.3)
cube_mesh = load(cube_path) |> cube_scale
cube_points = cube_mesh.position

# Ground truth
pose_gt = Pose(Translation(0, 0, 1.3), RotY(0.55))
# Estimated
poses = [Pose(Translation(0, 0.02, 1.3 + 0.02 * x), RotYX(0.5 + 0.1 * x, 0.1 * x)) for x in 1:6]
pose_es = poses[2]

# Measured scene - typically a depth image not a distance map
depth_context = depth_offscreen_context(WIDTH, HEIGHT, DEPTH, Array)
pose_occlusion = Pose(Translation(0, 0, 1.3), RotY(0))
occlusion_scale = Scale(0.5, 0.5, 0.1)
occlusion_mesh = load(cube_path) |> occlusion_scale
occlusion = upload_mesh(depth_context, occlusion_mesh)
occlusion = @set occlusion.pose = pose_occlusion
cube_ms = upload_mesh(depth_context, cube_mesh)
cube_ms = @set cube_ms.pose = pose_gt
ms_scene = Scene(camera, [cube_ms, occlusion])
ms_depth = draw(depth_context, ms_scene) |> copy

# VSD uses distance maps
distance_context = distance_offscreen_context(WIDTH, HEIGHT, DEPTH, Array)
cube = upload_mesh(distance_context, cube_mesh)

# Ground truth scene
cube = @set cube.pose = pose_gt
gt_scene = Scene(camera, [cube])
gt_dist = draw(distance_context, gt_scene) |> copy

# Estimate scene
cube = @set cube.pose = pose_es
es_scene = Scene(camera, [cube])
es_dist = draw(distance_context, es_scene) |> copy

# Depth to distance
ms_dist = depth_to_distance(ms_depth, cv_camera)
@testset "Depth to distance" begin
    # in the middle where the closest points
    @test minimum(ms_depth) ≈ minimum(ms_dist)
    # outer points have largest z values, distance even larger
    @test maximum(ms_depth) < maximum(ms_dist)
end

# Pixel visibility
δ = ITODD_δ
gt_visible = visibility_gt.(gt_dist, ms_dist, δ)
es_visible = visibility_es.(es_dist, ms_dist, δ, gt_visible)
# much of it is occluded and not visible
es_visible_gt = visibility_gt.(es_dist, ms_dist, δ)

@testset "Visibility Mask" begin
    # Sanity checks only parts should be visible
    @test sum(gt_visible) < sum(gt_dist .> 0)
    @test sum(es_visible) < sum(es_dist .> 0)
    # The parts of the gt visibility mask should have been added
    @test sum(es_visible) > sum(es_visible_gt)
    # Pour the equation from the paper into a function without thinking what it means
    function visibility_es_naive(rendered_dist, measured_dist, δ, gt_visible)
        es_visible = visibility_gt.(rendered_dist, measured_dist, δ)
        @. es_visible | (gt_visible & (rendered_dist > 0))
    end
    @test es_visible == visibility_es_naive(es_dist, ms_dist, δ, gt_visible)
end

@testset "Surface visibility" begin
    @test maximum(gt_visible) > 0
    @test maximum(es_visible) > 0
    @test gt_visible != es_visible
    @test sum(gt_dist .> 0) > sum(gt_visible .> 0)
end

τ = 0.05 * model_diameter(cube_mesh)
@testset "Surface Discrepancy" begin
    sd = @inferred surface_discrepancy(es_dist, gt_dist, τ)
    @test 0 < sd < 1
    @test sd < surface_discrepancy(es_dist, gt_dist, τ * 0.1)
end

@testset "Visible Surface Discrepancy" begin
    # Single pose, single τ
    vsd = @inferred vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, pose_es, pose_gt, δ, τ)
    vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, pose_es, pose_gt, δ, 0.05 * model_diameter(cube_mesh))
    @test 0 < vsd < 1
    @test vsd != surface_discrepancy(es_dist, gt_dist, τ)

    # Many poses, single τ
    vsd = @inferred vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, poses, pose_gt, δ, τ)
    @test size(vsd) == (6,)
    @test error_increases(vsd)

    # Single pose, many τ
    taus = (reverse(BOP19_THRESHOLDS)) * model_diameter(cube_mesh)
    # Relatively large pose error required to see any difference in the vsd error
    vsd = @inferred vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, poses[2], pose_gt, δ, taus)
    @test size(vsd) == (10,)
    @test error_increases(vsd)

    # Many poses, many τ
    vsd = @inferred vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, poses, pose_gt, δ, taus)
    @test size(vsd) == (10,)
    for err in vsd
        @test size(err) == (6,)
    end
    # For all τ, the error should increase. At least in this example. Different geometries might not steadily grow.
    @test sum(error_increases.(vsd)) == 10
end

@testset "Performance scores / average recall" begin
    # Point Distance
    mdd_s = @inferred mdds_error(cube_points, AffineMap(pose_gt), AffineMap(pose_es))
    recall = @inferred distance_recall_bop18(model_diameter(cube_points), mdd_s)
    @test recall == (mdd_s < 0.1 * model_diameter(cube_points))
    @test 0 <= recall <= 1

    recall = @inferred distance_recall_bop19(model_diameter(cube_points), mdd_s)
    @test recall == sum(mdd_s .< BOP19_THRESHOLDS * model_diameter(cube_points)) / length(BOP19_THRESHOLDS)
    @test 0 <= recall <= 1

    # Visual Surface Discrepancy
    vsd = @inferred vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, pose_es, pose_gt)
    recall = @inferred discrepancy_recall_bop18(vsd)
    @test recall == (vsd < 0.3)

    vsd = [vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, pose_es, pose_gt, ITODD_δ, τ) for τ in model_diameter(cube_points) * BOP19_THRESHOLDS]
    recall = @inferred discrepancy_recall_bop19(vsd)
    @test recall == mean([e < θ for e in vsd, θ in BOP19_THRESHOLDS])

    adds = @inferred adds_error(cube_points, AffineMap(pose_gt), AffineMap(pose_es))
    adds_recall = @inferred distance_recall_bop19(model_diameter(cube_points), adds)
    mdds = @inferred mdds_error(cube_points, AffineMap(pose_gt), AffineMap(pose_es))
    mdds_recall = @inferred distance_recall_bop19(model_diameter(cube_points), mdds)

    #Does vectorized version result in same VSD error?
    vsd = [vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, pose_es, pose_gt, ITODD_δ, τ) for τ in model_diameter(cube_points) * BOP19_THRESHOLDS]
    vsd_p = vsd_error(distance_context, cv_camera, cube_mesh, ms_dist, pose_es, pose_gt, ITODD_δ, Array(model_diameter(cube_points) * BOP19_THRESHOLDS))
    @test vsd == vsd_p

    vsd_recall = @inferred discrepancy_recall_bop19(vsd)
    adds_r, mdds_r, vsd_r = bop19_recalls(distance_context, cv_camera, cube_mesh, ms_dist, pose_es, pose_gt, ITODD_δ)
    @test adds_recall == adds_r
    @test mdds_recall == mdds_r
    @test vsd_recall == vsd_r
end

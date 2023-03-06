# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
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
WIDTH, HEIGHT, DEPTH = 640, 480, 1
gl_context = depth_offscreen_context(WIDTH, HEIGHT, DEPTH, Array)

camera = CvCamera(WIDTH, HEIGHT, 1.2 * WIDTH, 1.2 * HEIGHT, WIDTH / 2, HEIGHT / 2) |> Camera
cube_path = joinpath(dirname(pathof(SciGL)), "..", "examples", "meshes", "cube.obj")
cube_scale = Scale(0.3)
cube_mesh = load(cube_path) |> cube_scale
cube = upload_mesh(gl_context, cube_mesh)
cube_points = cube_mesh.position

# Ground truth scene
pose_gt = Pose(Translation(0, 0, 1.3), RotY(0.55))
cube = @set cube.pose = pose_gt
gt_scene = Scene(camera, [cube])
gt_img = draw(gl_context, gt_scene) |> copy

# Measured scene
pose_occlusion = Pose(Translation(0, 0, 1.3), RotY(0))
occlusion_scale = Scale(0.5, 0.5, 0.1)
occlusion_mesh = load(cube_path) |> occlusion_scale
occlusion = upload_mesh(gl_context, occlusion_mesh)
occlusion = @set occlusion.pose = pose_occlusion
ms_scene = Scene(camera, [cube, occlusion])
ms_img = draw(gl_context, ms_scene) |> copy

# Estimate scene
pose_es = Pose(Translation(0, 0.02, 1.3), RotYX(0.505, 0.05))
cube = @set cube.pose = pose_es
es_scene = Scene(camera, [cube])
es_img = draw(gl_context, es_scene) |> copy

# Pixel visibility
δ = 0.015
gt_mask = PoseErrors.pixel_visible.(gt_img, ms_img, δ)
es_mask = PoseErrors.pixel_visible.(es_img, ms_img, δ)


@testset "Surface visibility" begin
    @test maximum(gt_mask) > 0
    @test maximum(es_mask) > 0
    @test gt_mask != es_mask
    @test sum(gt_img .> 0) > sum(gt_mask .> 0)
end

τ = 0.02
@testset "Surface Discrepancy" begin
    sd = @inferred surface_discrepancy(gl_context, es_scene, gt_scene, τ)
    @test 0 < sd < 1
    @test sd == surface_discrepancy(es_img, gt_img, τ)
    @test sd < surface_discrepancy(gl_context, es_scene, gt_scene, τ * 0.1)
end

@testset "Visible Surface Discrepancy" begin
    vsd = @inferred vsd_error(gl_context, es_scene, gt_scene, ms_img, δ, τ)
    @test 0 < vsd < 1
    @test vsd != surface_discrepancy(gl_context, es_scene, gt_scene, τ)
end

@testset "Performance scores / average recall" begin
    # Point Distance
    mdd_s = @inferred mdds_error(cube_points, AffineMap(pose_gt), AffineMap(pose_es))
    recall = @inferred distance_recall_bop18(model_diameter(cube_points), mdd_s)
    @test recall == (mdd_s < 0.1 * model_diameter(cube_points))
    @test 0 <= recall <= 1

    recall = @inferred distance_recall_bop19(model_diameter(cube_points), mdd_s)
    bop_range = 0.05:0.05:0.5
    @test recall == sum(mdd_s .< bop_range * model_diameter(cube_points)) / length(bop_range)
    @test 0 <= recall <= 1


    # Visual Surface Discrepancy
    vsd = @inferred vsd_error(gl_context, es_scene, gt_scene, ms_img)
    recall = @inferred discrepancy_recall_bop18(vsd)
    @test recall == (vsd < 0.3)

    vsd = [vsd_error(gl_context, es_scene, gt_scene, ms_img, 0.015, τ) for τ in model_diameter(cube_points) * bop_range]
    recall = @inferred discrepancy_recall_bop19(vsd)
    @test recall == mean([e < θ for e in vsd, θ in bop_range])

    adds = @inferred adds_error(cube_points, AffineMap(pose_gt), AffineMap(pose_es))
    adds_recall = @inferred distance_recall_bop19(model_diameter(cube_points), adds)
    mdds = @inferred mdds_error(cube_points, AffineMap(pose_gt), AffineMap(pose_es))
    mdds_recall = @inferred distance_recall_bop19(model_diameter(cube_points), mdds)
    vsd = [vsd_error(gl_context, es_scene, gt_scene, ms_img, 0.015, τ) for τ in model_diameter(cube_points) * bop_range]
    vsd_recall = @inferred discrepancy_recall_bop19(vsd)

    adds_r, mdds_r, vsd_r = bop19_recalls(gl_context, camera, cube_mesh, ms_img, pose_es, pose_gt)
    @test adds_recall == adds_r
    @test mdds_recall == mdds_r
    @test vsd_recall == vsd_r
end

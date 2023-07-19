# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using ImageTransformations
using Interpolations
using PoseErrors
using SciGL
using Test

WIDTH = 600
HEIGHT = 400
FX, FY = 1.3 .* (HEIGHT, HEIGHT)
CX, CY = @. round(Int, 0.5 * (WIDTH, HEIGHT))
CUBE_SCALE = Scale(0.2)
RE_SIZE = (100, 100)

cv_camera = CvCamera(WIDTH, HEIGHT, FX, FY, CX, CY)
cam_pose = one(Pose)

@testset "Crop primitives" begin
    # Clamp to area which the camera can capture
    @test PoseErrors.clamp_boundingbox(-10, 30, 0, 20, 20, 20) == (0, 20, 0, 20)
    @test PoseErrors.clamp_boundingbox(0, 20, -10, 30, 20, 20) == (0, 20, 0, 20)
    @test PoseErrors.clamp_boundingbox(5, 15, 5, 15, 20, 20) == (5, 15, 5, 15)

    # should be in the center
    center3d = [0.0, 0.0, 0.9]
    center2d = PoseErrors.crop_center(cv_camera, center3d, cam_pose)
    @test center2d == [WIDTH, HEIGHT] ./ 2 .+ 1

    # Should be in lower right quarter
    center3d = [0.3, 0.1, 0.9]
    center2d = PoseErrors.crop_center(cv_camera, center3d, cam_pose)
    @test WIDTH > center2d[1] > WIDTH / 2
    @test HEIGHT > center2d[2] > HEIGHT / 2
end

# Draw an image to crop
gl_context = depth_offscreen_context(WIDTH, HEIGHT, 1, Array)

cube_path = joinpath(dirname(pathof(SciGL)), "..", "examples", "meshes", "cube.obj")
cube_mesh = CUBE_SCALE(load(cube_path))
cube_diameter = model_diameter(cube_mesh)

cube = upload_mesh(gl_context, cube_mesh)
cube = @set cube.pose.translation = Translation(0.2, 0.2, 0.7)
camera = Camera(cv_camera)
scene = Scene(camera, [cube])
full_img = draw(gl_context, scene) |> copy

@testset "Crop image" begin
    center2d = @inferred PoseErrors.crop_center(cv_camera, cube.pose.translation.translation, camera.pose)
    # Visually verified via full_img[center2d...] = 1
    @test round.(center2d) == [450, 350]

    bounding_box = @inferred PoseErrors.crop_boundingbox(cv_camera, cube.pose.translation.translation, cube_diameter)
    # Verified visually
    @test bounding_box == (257, 600, 157, 400)
    crop_img = @inferred PoseErrors.crop_image(full_img, bounding_box...)
    @test size(crop_img) == (344, 244)
    @test minimum(crop_img[crop_img.>0]) ≈ 0.6
end

# Resize cropped image to viewport size
bounding_box = PoseErrors.crop_boundingbox(cv_camera, cube.pose.translation.translation, cube_diameter)

gl_context = depth_offscreen_context((RE_SIZE)..., 1, Array)
cube_diameter = model_diameter(cube_mesh)
cube = upload_mesh(gl_context, cube_mesh)
cube = @set cube.pose.translation = Translation(0.2, 0.2, 0.7)
camera = crop(cv_camera, bounding_box...)
scene = Scene(camera, [cube])
crop_render = draw(gl_context, scene) |> copy
crop_img = PoseErrors.crop_image(full_img, bounding_box...)
resized = @inferred PoseErrors.depth_resize(crop_img, RE_SIZE...)
sum_drawn = sum(resized .> 0)

@testset "Resize image" begin
    EPS = 3e-3
    # "correct" nearest neighbor implementation with bad results
    resized = @inferred PoseErrors.depth_resize(crop_img, RE_SIZE...)
    @test size(resized) == RE_SIZE
    @test minimum(resized[resized.>0]) ≈ 0.6
    sum_const = sum(abs.(resized - crop_render) .> EPS)
    @test sum_const / sum_drawn < 0.1

    # my (bad) attempt of a custom implementation without interpolations
    resized = PoseErrors.depth_resize_custom(crop_img, RE_SIZE)
    @test size(resized) == RE_SIZE
    @test minimum(resized[resized.>0]) ≈ 0.6
    sum_custom = sum(abs.(resized - crop_render) .> EPS)

    # "wrong" interpolation - horrible corners great surfaces
    resized = @inferred imresize(crop_img, RE_SIZE...; method=Linear())
    @test size(resized) == RE_SIZE
    # Interpolates values which a camera would not capture
    @test minimum(resized[resized.>0]) < 0.6
    sum_linear = sum(abs.(resized - crop_render) .> EPS)
    @test sum_linear / sum_drawn < 0.05

    # Expectations from the comments above
    @test sum_linear < sum_const < sum_custom
end

# Slim objects
CUBE_SCALE = Scale(0.01, 0.01, 0.2)
cube_mesh = CUBE_SCALE(load(cube_path))
cube_diameter = model_diameter(cube_mesh)

# Draw an image to crop
gl_context = depth_offscreen_context(WIDTH, HEIGHT, 1, Array)
cube = upload_mesh(gl_context, cube_mesh)
cube = @set cube.pose.translation = Translation(0.2, 0.2, 0.7)
camera = Camera(cv_camera)
scene = Scene(camera, [cube])
full_img = draw(gl_context, scene) |> copy
crop_img = PoseErrors.crop_image(full_img, bounding_box...)
resized = @inferred PoseErrors.depth_resize(crop_img, RE_SIZE...)

gl_context = depth_offscreen_context((RE_SIZE)..., 1, Array)
cube = upload_mesh(gl_context, cube_mesh)
cube = @set cube.pose.translation = Translation(0.2, 0.2, 0.7)
camera = crop(cv_camera, bounding_box...)
scene = Scene(camera, [cube])
crop_render = draw(gl_context, scene) |> copy

@testset "Resize slim objects" begin
    EPS = 3e-3
    # "correct" nearest neighbor implementation with bad results
    resized = @inferred PoseErrors.depth_resize(crop_img, RE_SIZE...)
    @test size(resized) == RE_SIZE
    @test minimum(resized[resized.>0]) ≈ 0.6
    sum_const = sum(abs.(resized - crop_render) .> EPS)
    @test sum_const / sum_drawn < 0.1

    # my attempt of a custom implementation without interpolations
    resized = PoseErrors.depth_resize_custom(crop_img, RE_SIZE)
    @test size(resized) == RE_SIZE
    @test minimum(resized[resized.>0]) ≈ 0.6
    sum_custom = sum(abs.(resized - crop_render) .> EPS)

    # "wrong" interpolation - horrible corners great surfaces
    resized = @inferred imresize(crop_img, RE_SIZE...; method=Linear())
    @test size(resized) == RE_SIZE
    # Interpolates values which a camera would not capture
    @test minimum(resized[resized.>0]) < 0.6
    sum_linear = sum(abs.(resized - crop_render) .> EPS)
    @test sum_linear / sum_drawn < 0.05

    # Slim objects / objects with self occlusions are dominated by edges
    @test sum_const < sum_linear < sum_custom
end

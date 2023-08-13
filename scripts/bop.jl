# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using DataFrames
using FileIO
using PoseErrors
using SciGL

subset_path = joinpath(pwd(), "datasets", "tless", "test_primesense")
scene_ids = bop_scene_ids(subset_path)
bop_scene_path.(subset_path, scene_ids)
gt_df = gt_targets(subset_path, scene_ids[12])
@assert nrow(gt_df) == 293

gt = PoseErrors.gt_dataframe(subset_path, scene_ids[12])
info = PoseErrors.gt_info_dataframe(subset_path, scene_ids[12])
gt_info = leftjoin(info, gt; on=[:scene_id, :img_id, :gt_id])

row = gt_df[1, :]
WIDTH = 400
HEIGHT = 300
DEPTH = 1
gl_context = depth_context = depth_offscreen_context(WIDTH, HEIGHT, DEPTH, Array)

# Scene
camera = crop_camera(row)
mesh = upload_mesh(gl_context, load_mesh(row))
@reset mesh.pose = Pose(row.gt_t, row.gt_R)
scene = Scene(camera, [mesh])

# Draw result for visual validation (OpenGL not julia convention → transposed)
color_img = load_color_image(row, WIDTH, HEIGHT)
render_img = draw(gl_context, scene)
render_img ./ maximum(render_img) .|> Gray
mask_img = load_mask_image(row, WIDTH, HEIGHT)
depth_img = load_depth_image(row, WIDTH, HEIGHT)
# Mask the color and depth images
masked_depth = depth_img .* mask_img
masked_depth ./ maximum(depth_img) .|> Gray
color_img .* mask_img

# Test targets
test_df = test_targets(subset_path, scene_ids[12])
row = test_df[1, :]
color_img = load_color_image(row, WIDTH, HEIGHT);
depth_img = load_depth_image(row, WIDTH, HEIGHT);
depth_img ./ maximum(depth_img) .|> Gray
mask_img = load_mask_image(row, WIDTH, HEIGHT)
color_img .* mask_img

# Train targets
subset_path = joinpath(pwd(), "datasets", "lmo", "train")
train_df = train_targets(subset_path, 2)
# LM training contains one object per image
@assert findfirst(x -> x > 1, train_df.inst_count) == nothing
row = train_df[1, :]
color_img = load_color_image(row, WIDTH, HEIGHT);
depth_img = load_depth_image(row, WIDTH, HEIGHT);
depth_img ./ maximum(depth_img) .|> Gray
mask_img = load_mask_image(row, WIDTH, HEIGHT)
color_img .* mask_img

subset_path = joinpath(pwd(), "datasets", "tless", "test_primesense")
train_df = train_targets(subset_path, 12)
@assert train_df.inst_count[1] == 2
row = train_df[1, :]
color_img = load_color_image(row, WIDTH, HEIGHT);
depth_img = load_depth_image(row, WIDTH, HEIGHT);
depth_img ./ maximum(depth_img) .|> Gray
mask_img = load_mask_image(row, WIDTH, HEIGHT)
color_img .* mask_img

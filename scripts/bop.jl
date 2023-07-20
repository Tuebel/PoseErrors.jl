# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using DataFrames
using FileIO
using PoseErrors
using SciGL

# Datasets
s_df = scene_dataframe("tless", "test_primesense", 12)
@assert nrow(s_df) == 297

row = s_df[1, :]
WIDTH = 400
HEIGHT = 300
DEPTH = 1
gl_context = depth_context = depth_offscreen_context(WIDTH, HEIGHT, DEPTH, Array)

# Scene
camera = crop_camera(row)
mesh = upload_mesh(gl_context, load_mesh(row))
@reset mesh.pose = Pose(row.cam_t_m2c, row.cam_R_m2c)
scene = Scene(camera, [mesh])

# Draw result for visual validation (OpenGL not julia convention → transposed)
color_img = load_color_image(row, WIDTH, HEIGHT)
render_img = draw(gl_context, scene)
render_img ./ maximum(render_img) .|> Gray
mask_img = load_mask_image(row, WIDTH, HEIGHT)
depth_img = load_depth_image(row, WIDTH, HEIGHT)
depth_img ./ maximum(depth_img) .|> Gray
# Mask the color and depth images
depth_img .* mask_img ./ maximum(depth_img) .|> Gray
color_img .* mask_img

# TODO load camera noise depending on dataset name? Probabilistic Robotics: Larger Noise than expected? Tune parameter?

# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CoordinateTransformations
using PoseErrors
using Rotations
using Test

points = rand(3, 10)
pose0 = Translation(1.0, 0, 0) ∘ LinearMap(RotXYZ(0.5, 0, 0))
pose1 = LinearMap(RotXYZ(0.1, 0, 0)) ∘ pose

# Sanity check of nearest_neighbor_distances: Same pose should have zero as distance, different a Euclidean distance > 0
dists = @inferred PoseErrors.nearest_neighbor_distances(points, pose0, pose0)
@test mapreduce(iszero, &, dists)
dists = @inferred PoseErrors.nearest_neighbor_distances(points, pose0, pose1)
@test mapreduce(x -> x > 0, &, dists)

add_s0 = @inferred adds_error(points, pose0, pose0)
@test add_s0 == 0
add_s1 = @inferred adds_error(points, pose0, pose1)
@test add_s1 > 0

mdd_s0 = @inferred mdds_error(points, pose0, pose0)
@test mdd_s0 == 0
mdd_s1 = @inferred mdds_error(points, pose0, pose1)
@test mdd_s1 > 0
@test mdd_s1 >= add_s1

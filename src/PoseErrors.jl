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
export recall
export threshold_errors

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

include("Common.jl")
# point distance
include("ADD.jl")
# visual surface discrepancy
include("VSD.jl")

# BOP dataset evaluation
include("BOP.jl")

end # module PoseErrors

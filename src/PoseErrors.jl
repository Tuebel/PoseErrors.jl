module PoseErrors

export add_error
export adds_error
export mdds_error
export vsd_error

export normalized_add_error
export normalized_adds_error
export normalized_mdds_error
export normalized_vsd_error

export depth_to_distance
export model_diameter
export reproject_3D
export surface_discrepancy
export visibility_es, visibility_gt

export BOP19_THRESHOLDS, BOP_δ, ITODD_δ, BOP18_τ, ADDS_θ, BOP18_θ
export bop19_recalls, bop19_vsd_recall
export discrepancy_recall_bop18
export discrepancy_recall_bop19
export distance_recall_bop18
export distance_recall_bop19

# BOP dataset loading and preprocessing
export crop_camera
export load_color_image
export load_depth_image
export load_mask_image
export load_mesh
export load_mesh_eval
export load_segmentation
export load_visib_mask_image
export bop_scene_ids
export bop_scene_path

export gt_targets
export test_targets
export train_targets

# BOP dataset evaluation
export match_errors
export match_bop19_errors
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
include("CropImage.jl")
include("BOPDataFrames.jl")
include("BOPTargetsDataFrames.jl")
include("BOP.jl")

end # module PoseErrors

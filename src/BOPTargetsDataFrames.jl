# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using JSONTables

"""
    test_targets(datasubset_path, scene_id; [targets_file, detections_file, remove_bad])
Load the test targets for a specified scene with the corresponding detections.
By default only detections with a `score > 0.5` are included or at least `inst_count` detections if enough detections are available.
Returns a DataFrame with the columns `scene_id, img_id, obj_id, inst_count, diameter, mesh_path, bbox, score, segmentation`.
"""
function test_targets(datasubset_path, scene_id; targets_file="test_targets_bop19.json", detections_file="default_detections.json", remove_bad=true)
    # BOP test targets
    targets_df = targets_dataframe(datasubset_path, scene_id, targets_file)

    # Images & Camera parameters
    img_df = image_dataframe(datasubset_path, scene_id)
    leftjoin!(targets_df, img_df; on=[:scene_id, :img_id])
    cam_df = camera_dataframe(datasubset_path, scene_id, img_df)
    leftjoin!(targets_df, cam_df; on=[:scene_id, :img_id])

    # Object diameter and mesh_path
    obj_df = object_dataframe(dirname(datasubset_path))
    leftjoin!(targets_df, obj_df, on=:obj_id)

    # Default detections and segmentations
    det_df = detections_dataframe(datasubset_path, scene_id, detections_file)
    # If no detection is available it does not make sense to include the row
    targets_df = innerjoin(targets_df, det_df; on=[:scene_id, :img_id, :obj_id])
    remove_bad ? remove_bad_detections(targets_df) : targets_df
end

"""
    gt_targets(datasubset_path, scene_id; [targets_file])
Load the test targets for a specified scene with the corresponding *ground truth* visible detections and pose.
Returns a DataFrame with the columns `scene_id, img_id, obj_id, diameter, mesh_path, inst_count, bbox, color_path, depth_path, mask_path, mask_visib_path, depth_scale, gt_t, gt_R, img_size, cv_camera`.
"""
function gt_targets(datasubset_path, scene_id; targets_file="test_targets_bop19.json")
    # BOP test targets
    targets_df = targets_dataframe(datasubset_path, scene_id, targets_file)

    # Images & Camera parameters
    img_df = image_dataframe(datasubset_path, scene_id)
    leftjoin!(targets_df, img_df; on=[:scene_id, :img_id])
    cam_df = camera_dataframe(datasubset_path, scene_id, img_df)
    leftjoin!(targets_df, cam_df; on=[:scene_id, :img_id])

    # Object diameter and mesh_path
    obj_df = object_dataframe(dirname(datasubset_path))
    leftjoin!(targets_df, obj_df, on=:obj_id)

    # Ground truth
    gt_df = gt_dataframe(datasubset_path, scene_id)
    info_df = gt_info_dataframe(datasubset_path, scene_id)
    # Only visib_fract >= 0.1 is considered valid → gt_info_df might include less entries on purpose
    gt_info_df = leftjoin(info_df, gt_df; on=[:scene_id, :img_id, :gt_id])

    leftjoin(targets_df, gt_info_df; on=[:scene_id, :img_id, :obj_id])
end

"""
    train_targets(datasubset_path, scene_id)
Load the training targets for a specified scene which are all images and annotated poses.
Suitable for train and validation datasets.
Returns a DataFrame with the columns `scene_id, img_id, obj_id, diameter, mesh_path, inst_count, bbox, color_path, depth_path, mask_path, mask_visib_path, depth_scale, gt_t, gt_R, img_size, cv_camera`.
"""
function train_targets(datasubset_path, scene_id)
    # Ground truth
    gt_df = gt_dataframe(datasubset_path, scene_id)
    info_df = gt_info_dataframe(datasubset_path, scene_id)
    # Only visib_fract >= 0.1 is considered valid → gt_info_df might include less entries on purpose
    gt_info_df = leftjoin(info_df, gt_df; on=[:scene_id, :img_id, :gt_id])
    # Calculate instant count for valid views
    img_obj_groups = groupby(gt_info_df, [:img_id, :obj_id])
    gt_info_df = transform(img_obj_groups, nrow => :inst_count)

    # Images & Camera parameters
    img_df = image_dataframe(datasubset_path, scene_id)
    leftjoin!(gt_info_df, img_df; on=[:scene_id, :img_id])
    cam_df = camera_dataframe(datasubset_path, scene_id, img_df)
    leftjoin!(gt_info_df, cam_df; on=[:scene_id, :img_id])

    # Object diameter and mesh_path
    obj_df = object_dataframe(dirname(datasubset_path))
    leftjoin!(gt_info_df, obj_df, on=:obj_id)
end

"""
    targets_dataframe(datasubset_path, scene_id, [targets_file="test_targets_bop19.json"])
Load the test targets from the specified file.
Only detections for the `scene_id` are returned as a DataFrame with columns `scene_id, img_id, obj_id, inst_count`
"""
function targets_dataframe(datasubset_path, scene_id, targets_file="test_targets_bop19.json")
    targets_json = jsontable(joinpath(datasubset_path, "..", targets_file))
    targets_df = DataFrame(targets_json)
    filter!(row -> row.scene_id == scene_id, targets_df)
    # common naming convention
    rename!(targets_df, :im_id => :img_id)
    targets_df
end

"""
    detections_dataframe(datasubset_path, scene_id, [detections_file="default_detections.json"])
Load the detections from the specified file.
Only detections for the `scene_id` are returned as a DataFrame with columns `scene_id, img_id, obj_id, bbox, score,  segmentation`
"""
function detections_dataframe(datasubset_path, scene_id, detections_file="default_detections.json")
    json = jsontable(joinpath(datasubset_path, "..", detections_file))
    df = DataFrame(json)
    filter!(row -> row.scene_id == scene_id, df)
    # convert bounding box format
    @. df.bbox = convert_bop_bbox(df.bbox)
    # common naming convention
    rename!(df, :category_id => :obj_id, :image_id => :img_id)
    # time is not required for evaluation
    select!(df, Not(:time))
end

"""
    convert_detection_bbox(bbox)
(left, top, width, height) → (left, right, top, bottom)
"""
function convert_bop_bbox(bbox)
    left, top, width, height = bbox
    left, top = (left, top) .+ 1
    left, left + width, top, top + height
end

"""
    remove_bad_detections(test_targets)
Avoid too many evaluations by removing bad detections, i.e. if `score <= 0.5`.
If enough detections are available, at least the known number of instances is returned.
"""
function remove_bad_detections(test_targets)
    groups = groupby(test_targets, [:scene_id, :img_id, :obj_id])
    combine(groups) do group
        last_good = findlast(s -> s > 0.5, group.score)
        # if nothing is found inst_count should determine the maximum
        last_good = isnothing(last_good) ? 0 : last_good
        # ideally all good detections or at least the required amount of detections should be evaluated
        desired = max(last_good, group[1, :].inst_count)
        # return at most all elements of the group.
        last_index = min(nrow(group), desired)
        group[1:last_index, :]
    end
end
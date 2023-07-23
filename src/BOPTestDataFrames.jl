# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using JSONTables

# TODO dataframe for the evaluation (gt)
# TODO dataframe for the test targets + gt pose / mask
# TODO dataframe for the test targets + default detections

"""
    targets_dataframe(datasubset_path, scene_id, [detections_file="bop23_default_task4.json"])
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
    detections_dataframe(datasubset_path, scene_id, [detections_file="bop23_default_task4.json"])
Load the detections from the specified file.
Only detections for the `scene_id` are returned as a DataFrame with columns `scene_id, img_id, obj_id, bbox, score,  segmentation`
"""
function detections_dataframe(datasubset_path, scene_id, detections_file="bop23_default_task4.json")
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
    left, left + width, top, top + height
end

"""
    scene_test_targets(datasubset_path, scene_id; targets_file="test_targets_bop19.json", detections_file="bop23_default_task4.json", remove_bad=true)
Load the test targets for a specified scene with the corresponding detections.
By default only detections with a `score > 0.5` are included or at least `inst_count` detections if enough detections are available.
Returns a DataFrame with the columns `scene_id, img_id, obj_id, inst_count, bbox, score, segmentation`.
"""
function scene_test_targets(datasubset_path, scene_id; targets_file="test_targets_bop19.json", detections_file="bop23_default_task4.json", remove_bad=true)
    # BOP test images
    targets_df = targets_dataframe(datasubset_path, scene_id, targets_file)
    # Images & Camera parameters
    img_df = image_dataframe(datasubset_path, scene_id)
    leftjoin!(targets_df, img_df; on=[:scene_id, :img_id])
    cam_df = camera_dataframe(datasubset_path, scene_id, img_df)
    leftjoin!(targets_df, cam_df; on=[:scene_id, :img_id])
    # Default detections and segmentations
    det_df = detections_dataframe(datasubset_path, scene_id, detections_file)
    # If no detection is available it does not make sense to include the row
    targets_df = innerjoin(targets_df, det_df; on=[:scene_id, :img_id, :obj_id])
    remove_bad ? remove_bad_detections(targets_df) : targets_df
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
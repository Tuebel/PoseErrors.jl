# PoseErrors.jl
A good overview and rationale behind 6D pose error metrics can be found in the [BOP-challenge](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#evaluationmethodology).
They prefer Maximum Symmetry-Aware Surface Distance (MSSD) over Averade Average Distance of Model Points with indistinguishable views (ADD-S = ADI), because they can yield low errors with bad visual alignment.
Moreover, ADD-S is dominated by higher-frequency surface parts (e.g. a Thread).
Maximum distances do not suffer from the surface sampling density as much.

However, annotating the symmetries is tedious and heavily depends on the choice of coordinate frames.
For a large set of objects like surgical instruments which are not exported in a standardized / symmetry aligned frame, this is impractical.
So, similar to ([Gorschlüter et al. 2022](https://doi.org/10.3390/jimaging8030053)) we use ADD-S and VDS ([Hodan et. al 2016](https://doi.org/10.1007/978-3-319-49409-8_52)) as metrics.

# Setup for BOP dataset evaluation
Extract the BOP datasets as described on their [website](https://bop.felk.cvut.cz/datasets/).
Move the detections JSON to the matching datasets test directory and rename it to `default_detections.json`, e.g. *datasets/tless/default_detections.json*.

You could also you the keyword argument `detections_file` of `scene_test_targets` to specify another file in the test directory.

## Loading the BOP datasets
Your first entry points should be the methods to load the according targets.
These load the required image files, mesh files, camera parameters, etc.
* `gt_targets`: Loads the ground truth pose as `:gt_t` and `:gt_R`, the **ground truth** visible bounding box, and the **gt** mask image paths.
* `test_targets`: the **estimated** bounding box, and the **estimated** segmentation masks.

Iterate each row and load cropped images via:
* `load_color_img(row, width, height)` 
* `load_depth_img(row, width, height)` scaled in meters as Float32
* `load_mask_img(row, width, height)` masks the visible object surface either gt from disk or from the detections file in the test targets.

# Maximum Distance of Model Points for indistinguishable views (MDD-S)
To avoid defining symmetries and the influence of the mesh sampling, we implement the MDD-S by replacing the ``avg`` in ADD-S with ``max``:
```math
e_{MDDS}(\hat{\mathbf{P}},\mathbf{P};\mathcal{M})= \max_{\mathbf{x}_1 \in \mathcal{M}} \min_{\mathbf{x}_2 \in \mathcal{M}} \parallel \hat{\mathbf{P}} \mathbf{x}_1 - \mathbf{P} \mathbf{x}_2 \parallel_2
```
As ([Hodan et. al 2016](https://doi.org/10.1007/978-3-319-49409-8_52)) describe ADD-S as the lower bound of the ADD error.
However, the symmetric transformation is a subtle difference which results in MDD-S not being a lower bound of MSSD.
It heavily depends on the set of global symmetries used and makes comparability impossible.

![MDD-S not lower boud of MDDS](./assets/mdds_no_bound.jpg)

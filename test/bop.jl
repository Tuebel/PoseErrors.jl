# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using PoseErrors
using Test

scores = [2, 3]
errors_per_gt = [[3.0, 2.1], [1.1, 3.5], [0.5, 0.6]]
matched_errors = match_errors(scores, errors_per_gt)
@test matched_errors == [2.1, 1.1, Inf]

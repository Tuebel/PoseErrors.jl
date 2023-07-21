# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using PoseErrors
using Statistics
using Test

@testset "error matching, thresholding, recall" begin
    # Matching
    scores = [2, 3]
    errors_per_gt = [[3.0, 2.1], [1.1, 3.5], [0.5, 0.6]]
    matched_errors = match_errors(scores, errors_per_gt)
    @test matched_errors == [2.1, 1.1, Inf]

    # Thresholds
    @test threshold_errors(matched_errors, 1.5) == [1, 3]
    @test threshold_errors(matched_errors, 1.5:2.5) == [3, 6]
    @test threshold_errors(matched_errors, [1.5, 2.5]) == [3, 6]
    # Must be < threshold
    @test threshold_errors(matched_errors, [1.5, 2.1]) == [2, 6]

    # recall
    @test recall(threshold_errors(matched_errors, 1.5)) == mean([e < θ for e in matched_errors, θ in 1.5])
    @test recall(threshold_errors(matched_errors, 0.5:2.5)) == mean([e < θ for e in matched_errors, θ in 0.5:2.5])
end

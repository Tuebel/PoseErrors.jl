# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using PoseErrors
using Statistics
using Test

@testset "error matching, thresholding, recall" begin
    # 3 estimates, 2 gt
    scores = [2, 3, 1]
    errors_per_est = [[1.0, 2.1], [1.1, 3.5], [0.5, 0.6]]
    matched_errors = match_errors(scores, errors_per_est)
    @test matched_errors == [1.1, 2.1]

    # 2 estimates, 3 gt
    scores = [2, 1]
    errors_per_est = [[1.0, 2.1, 1.1], [3.5, 0.5, 0.6]]
    matched_errors = match_errors(scores, errors_per_est)
    @test matched_errors == [1.0, 0.5, Inf]
    # Different order for Inf
    errors_per_est = [[1.0, 2.1, 0.6], [3.5, 0.5, 0.6]]
    matched_errors = match_errors(scores, errors_per_est)
    @test matched_errors == [Inf, 0.5, 0.6]

    # Thresholds
    matched_errors = [1.1, 2.1, 1.0]
    @test threshold_errors(matched_errors, 1.5) == [2, 3]
    @test threshold_errors(matched_errors, 1.5:2.5) == [5, 6]
    @test threshold_errors(matched_errors, [1.5, 2.5]) == [5, 6]
    # Must be < threshold
    @test threshold_errors(matched_errors, [1.5, 2.1]) == [4, 6]

    # recall
    @test recall(threshold_errors(matched_errors, 1.5)) == mean([e < θ for e in matched_errors, θ in 1.5])
    @test recall(threshold_errors(matched_errors, 0.5:2.5)) == mean([e < θ for e in matched_errors, θ in 0.5:2.5])
end

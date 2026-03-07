// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/moe_compressed.hpp"

namespace ov::op::internal {

/// \brief MOE3GemmFusedCompressed that supports compressed and fused MOE for GEMM3_SWIGLU.
class OPENVINO_API MOE3GemmFusedCompressed : public MOECompressed {
public:
    OPENVINO_OP("MOE3GemmFusedCompressed");

    MOE3GemmFusedCompressed() = default;

    /// \brief Constructs a MOE3GemmFusedCompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_seq, num_experts] routing weights for all experts
    ///   2: w0_weight - expert weights for first projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   3: w0_scale - expert scale for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   4: w0_zp - expert zp for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   5: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   6: w1_scale - expert scale for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   7: w1_zp - expert zp for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   8: w2_weight - expert weights for final projection,
    ///   shape [num_experts, hidden_size, group_num, group_size]
    ///   9: w2_scale - expert scale for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    ///   10: w2_zp - expert zp for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    ///
    ///   Optional inputs for Shared Expert:
    ///   11: shared_gate_weight - shared expert weights for first projection,
    ///   shape [1, inter_size, group_num, group_size]
    ///   12: shared_gate_scale - shared expert scale for first projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   13: shared_gate_zp - shared expert zp for first projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   14: shared_up_weight - shared expert weights for second projection,
    ///   shape [1, inter_size, group_num, group_size]
    ///   15: shared_up_scale - shared expert scale for second projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   16: shared_up_zp - shared expert zp for second projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   17: shared_down_weight - shared expert weights for final projection,
    ///   shape [1, hidden_size, group_num, group_size]
    ///   18: shared_down_scale - shared expert scale for final projection,
    ///   shape [1, hidden_size, group_num, 1]
    ///   19: shared_down_zp - shared expert zp for final projection,
    ///   shape [1, hidden_size, group_num, 1]
    ///   20: shared_gate_gate_weight - shared expert gate weight for gating,
    ///   shape [hidden_size]
    /// \param config Configuration for the MOE 3GEMM SWIGLU fused operation
    MOE3GemmFusedCompressed(const OutputVector& args, const Config config);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::op::internal

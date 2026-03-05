// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/moe.hpp"

namespace ov::op::internal {

/// \brief MOECompressed experts that support compressed weights for GEMM3_SWIGLU MOE.
class OPENVINO_API MOECompressed : public MOE {
public:
    OPENVINO_OP("MOECompressed");

    MOECompressed() = default;
    MOECompressed(const OutputVector& args) : MOE(args) {}

    struct Config : public MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t num_expert = 0;
        size_t num_shared_expert = 0;
        size_t top_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group).
        // other non-zero value means group compression with this given group_size.
        size_t group_size = 0; 
        // In CB, intermediate shapes are expanded to {SeqLen, 1, HiddenSize}
        // In Non-CB, intermediate shapes are expanded to {Batch, SeqLen, HiddenSize}
        size_t has_batch_dim = 0;
        bool has_zp = false;
        ov::element::Type out_type = ov::element::dynamic;
        Config() = default;
        Config(const MOE::Config& moe_config) : MOE::Config(moe_config) {}
    };

    /// \brief Constructs a MOECompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_experts, ...] normalized weights for selected experts
    ///      (input to final multiplication)
    ///   2: router_topk_output_indices - [..., topk] indices of selected top-k experts
    ///   3: w0_weight - expert weights for first projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   4: w0_scale - expert scale for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   5: w0_zp - expert zp for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   6: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   7: w1_scale - expert scale for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   8: w1_zp - expert zp for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   9: w2_weight - expert weights for final projection,
    ///   shape [num_experts, hidden_size, group_num, group_size]
    ///   10: w2_scale - expert scale for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    ///   11: w2_zp - expert zp for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    ///
    ///   Optional inputs for Shared Expert:
    ///   12: shared_gate_weight - shared expert weights for first projection,
    ///   shape [1, inter_size, group_num, group_size]
    ///   13: shared_gate_scale - shared expert scale for first projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   14: shared_gate_zp - shared expert zp for first projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   15: shared_up_weight - shared expert weights for second projection,
    ///   shape [1, inter_size, group_num, group_size]
    ///   16: shared_up_scale - shared expert scale for second projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   17: shared_up_zp - shared expert zp for second projection,
    ///   shape [1, inter_size, group_num, 1]
    ///   18: shared_down_weight - shared expert weights for final projection,
    ///   shape [1, hidden_size, group_num, group_size]
    ///   19: shared_down_scale - shared expert scale for final projection,
    ///   shape [1, hidden_size, group_num, 1]
    ///   20: shared_down_zp - shared expert zp for final projection,
    ///   shape [1, hidden_size, group_num, 1]
    ///   21: shared_gate_gate_weight - shared expert gate weight for gating,
    ///   shape [hidden_size]
    /// \param config Configuration for the MOE operation
    MOECompressed(const OutputVector& args, const Config& config);

    const Config& get_config() const;
    void set_config(const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config;
};

}  // namespace ov::op::internal

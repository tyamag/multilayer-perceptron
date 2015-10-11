#include "Layer.hpp"

#include <numeric>
#include <random>

#include "MathUtil.hpp"

namespace ccilab {

Layer::Layer(const int num_nodes)
    : num_nodes_(num_nodes) {
}

void Layer::Init(const int child_layer_nodes_num,
    const double default_weight_min, const double default_weight_max) {
    outputs_.resize(num_nodes_);
    errors_.resize(num_nodes_);
    for (int i = 0; i < num_nodes_; ++i) {
        outputs_[i] = 0.0;
        errors_[i] = 0.0;
    }

    // d‚Ý‚ð—”‚Å‰Šú‰»
    if (child_layer_nodes_num < 1) {
        return;
    }

    std::random_device rand;
    std::mt19937 mt(rand());
    std::uniform_real_distribution<> rand_dist(default_weight_min, default_weight_max);
    weights_list_.resize(child_layer_nodes_num);
    for (auto& weights : weights_list_) {
        weights.resize(num_nodes_ );
        for (auto& weight : weights) {
            weight = rand_dist(mt);
        }
    }

    bias_weights_.resize(child_layer_nodes_num);
    for (auto& bias_weight : bias_weights_) {
        bias_weight = rand_dist(mt);
    }
}

void Layer::CalculateOutputs(const std::vector<double>& inputs) {
    outputs_ = inputs;
}

void Layer::CalculateOutputs(const Layer& parent_layer) {
    for (int i = 0; i < num_nodes_; ++i) {
        const auto& weights = parent_layer.weights_list()[i];
        const auto& bias_weights = parent_layer.bias_weights();
        const auto& inputs = parent_layer.outputs();
        double output = std::inner_product(weights.begin(), weights.end(), inputs.begin(), 0.0);
        output += bias_weights[i];
        outputs_[i] = ccilab::Sigmoid(output);
    }
}

void Layer::CalculateErrors(const std::vector<double>& answers) {
    for (int i = 0; i < num_nodes_; ++i) {
        errors_[i] = -(answers[i] - outputs_[i]) * outputs_[i] * (1.0 - outputs_[i]);
    }
}

void Layer::CalculateErrors(const Layer& child_layer) {
    for (int i = 0; i < num_nodes_; ++i) {
        const auto& child_errors = child_layer.errors();
        double sum = 0;
        for (int j = 0; j < child_layer.num_nodes(); ++j) {
            sum += child_errors[j] * weights_list_[j][i];
        }
        errors_[i] = sum * outputs_[i] * (1.0 - outputs_[i]);
    }
}

void Layer::UpdateWeights(const Layer& child_layer, const double learning_rate) {
    for (int i = 0; i < num_nodes_; ++i) {
        for (int j = 0; j < child_layer.num_nodes(); ++j) {
            weights_list_[j][i] += -learning_rate * child_layer.errors()[j] * outputs_[i];
        }
    }

    for (int i = 0; i < child_layer.num_nodes(); ++i) {
        bias_weights_[i] += -learning_rate * child_layer.errors()[i] * 1;
    }
}

}  // namespace ccilab

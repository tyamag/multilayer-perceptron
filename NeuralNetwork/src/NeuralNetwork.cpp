#include "NeuralNetwork.hpp"

namespace ccilab {

NeuralNetwork::NeuralNetwork(const NeuralNetworkParameter& param)
    : learning_rate_(param.learning_rate_) {
    Init(param);
}

bool NeuralNetwork::Init(const NeuralNetworkParameter& param) {
    for (const int num_nodes : param.num_nodes_list_) {
        if (num_nodes < 0) {
            return false;
        }
        layers_.emplace_back(num_nodes);
    }

    return ConnectLayers(param);
}

bool NeuralNetwork::Train(const std::vector<std::vector<double> >& inputs_list,
    const std::vector<std::vector<double> >& answers_list, const int num_epoch) {
    if (inputs_list.empty() || answers_list.empty()
        || inputs_list.size() != answers_list.size()) {
        return false;
    }

    for (int i = 0; i < num_epoch; ++i) {
        for (unsigned int in_idx = 0; in_idx < inputs_list.size(); ++in_idx) {
            Foward(inputs_list[in_idx]);
            Backward(answers_list[in_idx]);
        }
    }

    return true;
}

const std::vector<double>& NeuralNetwork::Foward(const std::vector<double>& inputs) {
    // ���͑w
    layers_[0].CalculateOutputs(inputs);

    // ���ԑw�`�o�͑w
    for (unsigned int i = 1; i < layers_.size(); ++i) {
        const auto& parent_layer = layers_[i - 1];
        layers_[i].CalculateOutputs(parent_layer);
    }

    return layers_[layers_.size() - 1].outputs();
}

void NeuralNetwork::Backward(const std::vector<double>& answers) {
    // �o�͑w
    int out_layer_idx = layers_.size() - 1;
    layers_[out_layer_idx].CalculateErrors(answers);

    // ���ԑw
    for (int i = out_layer_idx - 1; i > 0; --i) {
        const auto& child_layer = layers_[i + 1];
        layers_[i].CalculateErrors(child_layer);
    }

    // ���ԑw�`���͑w�̏d�݂��X�V
    for (int i = out_layer_idx - 1; i >= 0; --i) {
        const auto& child_layer = layers_[i + 1];
        layers_[i].UpdateWeights(child_layer, learning_rate_);
    }
}

double NeuralNetwork::CalculateError(const std::vector<double>& answers) const {
    double error = 0.0;
    const auto& out_layer = layers_[layers_.size() - 1];
    for (unsigned int i = 0; i < out_layer.outputs().size(); ++i) {
        const auto& output = out_layer.outputs()[i];
        const auto& answer = answers[i];
        error += (output - answer) * (output - answer);
    }
    error /= 2;

    return error;
}

bool NeuralNetwork::ConnectLayers(const NeuralNetworkParameter& param) {
    if (layers_.empty()) {
        return false;
    }

    for (unsigned int i = 0; i < layers_.size(); ++i) {
        const int child_layer_nodes_num =
            (i == layers_.size() - 1) ? 0 : layers_[i + 1].num_nodes();
        const double initial_weight_min = param.initial_weight_minmax.first;
        const double initial_weight_max = param.initial_weight_minmax.second;
        layers_[i].Init(child_layer_nodes_num, initial_weight_min, initial_weight_max);
    }

    return true;
}

}  // namespace ccilab

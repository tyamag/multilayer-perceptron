#include "NeuralNetLib/NeuralNet.hpp"

#define PICOJSON_USE_INT64
#include "picojson/picojson.h"

namespace ccilab {

NeuralNet::NeuralNet(const NeuralNetParameter& param)
    : learning_rate_(param.learning_rate_) {
    Init(param);
}

NeuralNet::NeuralNet(const std::string& model_file_path)
    : learning_rate_(0) {
    Init(model_file_path);
}

bool NeuralNet::Train(const std::vector<std::vector<double> >& inputs_list,
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

const std::vector<double>& NeuralNet::Foward(const std::vector<double>& inputs) {
    // 入力層
    layers_[0].CalculateOutputs(inputs);

    // 中間層～出力層
    for (unsigned int i = 1; i < layers_.size(); ++i) {
        const auto& parent_layer = layers_[i - 1];
        layers_[i].CalculateOutputs(parent_layer);
    }

    return layers_[layers_.size() - 1].outputs();
}

void NeuralNet::Backward(const std::vector<double>& answers) {
    // 出力層
    int out_layer_idx = layers_.size() - 1;
    layers_[out_layer_idx].CalculateErrors(answers);

    // 中間層
    for (int i = out_layer_idx - 1; i > 0; --i) {
        const auto& child_layer = layers_[i + 1];
        layers_[i].CalculateErrors(child_layer);
    }

    // 中間層～入力層の重みを更新
    for (int i = out_layer_idx - 1; i >= 0; --i) {
        const auto& child_layer = layers_[i + 1];
        layers_[i].UpdateWeights(child_layer, learning_rate_);
    }
}

double NeuralNet::CalculateError(const std::vector<double>& answers) const {
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

void NeuralNet::SaveModel() {
    picojson::object net_json;

    // レイヤー
    // TODO Layer クラスに実装する
    picojson::array layers_json;
    for (const auto& layer : layers_) {
        // ノード数
        picojson::object layer_json;
        const int64_t num_nodes = static_cast<int64_t>(layer.num_nodes());
        layer_json["num_nodes"] = picojson::value(num_nodes);

        // 重み
        picojson::array weights_list_json;
        for (const auto& weights : layer.weights_list()) {
            picojson::array weights_json;
            for (const auto weight : weights) {
                weights_json.push_back(picojson::value(weight));
            }
            weights_list_json.push_back(picojson::value(weights_json));
        }
        layer_json["weights_list"] = picojson::value(weights_list_json);

        // バイアス
        picojson::array bias_weights_json;
        for (const auto& bias_weight : layer.bias_weights()) {
            bias_weights_json.push_back(picojson::value(bias_weight));
        }
        layer_json["bias_weights"] = picojson::value(bias_weights_json);

        layers_json.push_back(picojson::value(layer_json));
    }
    net_json["layers"] = picojson::value(layers_json);

    // 学習率
    net_json["learning_rate"] = picojson::value(learning_rate_);

    picojson::value model_json(net_json);
    std::cout << model_json.serialize() << std::endl;
}

bool NeuralNet::Init(const NeuralNetParameter& param) {
    for (unsigned int i = 0; i < param.num_nodes_list_.size(); ++i) {
        const int num_nodes = param.num_nodes_list_[i];
        if (num_nodes < 0) {
            return false;
        }
        const int num_child_nodes =
            (i == param.num_nodes_list_.size() - 1) ? 0 : param.num_nodes_list_[i + 1];
        const double initial_weight_min = param.initial_weight_minmax.first;
        const double initial_weight_max = param.initial_weight_minmax.second;
        layers_.emplace_back(num_nodes, num_child_nodes, initial_weight_min, initial_weight_max);
    }

    return true;
}

bool NeuralNet::Init(const std::string& model_file_path) {

}

}  // namespace ccilab

#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <utility>

#include "Layer.hpp"

namespace ccilab {

/**
 * @brief ニューラルネットワーク初期化に用いるパラメーター
 */
struct NeuralNetworkParameter {
    std::vector<int> num_nodes_list_;                 // 各層のノード数
    std::pair<double, double> initial_weight_minmax;  // 重み初期値の最小, 最大
    double learning_rate_;                            // 学習率
};

/**
 * @brief ニューラルネットワーク.
 */
class NeuralNetwork {
public:
    /**
     * @brief ネットワークのパラメータをもとに初期化する
     * @param ネットワークのパラメータ
     */
    NeuralNetwork(const NeuralNetworkParameter& param);
    ~NeuralNetwork() {}

    /**
     * @brief ネットワークを学習する
     * @param inputs 入力データ
     * @param answers 正解データ
     * @param num_epoch エポック数(学習回数)
     */
    bool Train(const std::vector<std::vector<double> >& inputs_list,
        const std::vector<std::vector<double> >& answers_list, int num_epoch);

    /**
     * @brief 順方向伝搬.
     */
    const std::vector<double>& Foward(const std::vector<double>& inputs);

    /**
     * @brief 誤差逆伝搬.
     */
    void Backward(const std::vector<double>& answers);

    /**
     * @brief 誤差を計算して返す.
     */
    double CalculateError(const std::vector<double>& answers) const;

private:
    std::vector<Layer> layers_;  // レイヤー
    double learning_rate_;       // 学習率

    /**
     * @brief ネットワークを初期化する.
     * @param param ネットワークのパラメータ
     */
    bool Init(const NeuralNetworkParameter& param);
};

}  // namespace ccilab

#endif  // NEURAL_NETWORK_HPP_

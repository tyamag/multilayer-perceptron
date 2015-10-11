#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <vector>

namespace ccilab {

/**
 * @brief ネットワークを構成するレイヤー(層).
 */
class Layer {
public:
    explicit Layer(int num_nodes);
    ~Layer() {}

    /**
     * @brief レイヤーを初期化する.
     * @param child_layer_nodes_num 子層(次の層)のノード数
     * @param default_weight_min 重み初期値の最小
     * @param default_weight_max 重み初期値の最大
     */
    void Init(int child_layer_nodes_num, double default_weight_min, double default_weight_max);

    /**
     * @brief すべてのノードの出力値を計算し, 更新する.
     *        入力層の出力値を更新するときに使う.
     * @param inputs 入力データ
     */
    void CalculateOutputs(const std::vector<double>& inputs);

    /**
     * @brief すべてのノードの出力値を計算し, 更新する.
     *        中間層～出力層の出力値を更新するときに使う.
     * @param parent_layer 親層(前の層)
     */
    void CalculateOutputs(const Layer& parent_layer);

    /**
     * @brief すべてのノードの誤差信号を計算し, 更新する.
     *        出力層の誤差信号を更新するときに使う.
     * @param answers 正解データ
     */
    void CalculateErrors(const std::vector<double>& answers);

    /**
     * @brief すべてのノードの誤差信号を計算し, 更新する.
     *        中間層の誤差信号を更新するときに使う.
     * @param child_layer 子層
     */
    void CalculateErrors(const Layer& child_layer);

    /**
     * @brief すべての重みを更新する.
     * @param child_layer 子層
     * @param learning_rate 学習率
     */
    void UpdateWeights(const Layer& child_layer, double learning_rate);

    /**
     * @brief ノード数を返す.
     */
    int num_nodes() const { return num_nodes_; }

    /**
     * @brief 重みを返す.
     */
    const std::vector<std::vector<double> >& weights_list() const { return weights_list_; }

    /**
     * @brief バイアスを返す.
     */
    const std::vector<double>& bias_weights() const { return bias_weights_; }

    /**
     * @brief 出力値を返す.
     */
    const std::vector<double>& outputs() const { return outputs_; }

    /**
     * @brief 誤差信号を返す.
     */
    const std::vector<double>& errors() const { return errors_; }

private:
    int num_nodes_;                             // ノード数
    std::vector<std::vector<double> > weights_list_;  // 重み
    std::vector<double> bias_weights_;                // バイアス
    std::vector<double> outputs_;                     // ニューロンの出力値
    std::vector<double> errors_;                      // 誤差信号
};

}  // namespace ccilab

#endif  // LAYER_HPP_

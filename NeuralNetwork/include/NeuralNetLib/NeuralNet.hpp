#ifndef NEURAL_NET_HPP_
#define NEURAL_NET_HPP_

#include <utility>

#include "NeuralNetLib/Layer.hpp"

namespace ccilab {

/**
 * @brief ニューラルネットワーク初期化に用いるパラメーター
 */
struct NeuralNetParameter {
    std::vector<int> num_nodes_list_;                 // 各層のノード数
    std::pair<double, double> initial_weight_minmax;  // 重み初期値の最小, 最大
    double learning_rate_;                            // 学習率
};

/**
 * @brief ニューラルネットワーク.
 */
class NeuralNet {
public:
    /**
     * @brief ネットワークのパラメータをもとに初期化する
     * @param ネットワークのパラメータ
     */
    explicit NeuralNet(const NeuralNetParameter& param);

    /**
     * @brief 学習済みのモデルからネットワークを初期化する.
     * @param model_file_path モデルのファイルパス
     */
    explicit NeuralNet(const std::string& model_file_path);
    ~NeuralNet() {}

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

    /**
     * @brief 学習したモデルをファイルに書き出す.
     *        モデルは json 形式で保存される.
     */
    void SaveModel();

private:
    std::vector<Layer> layers_;  // レイヤー
    double learning_rate_;       // 学習率

    /**
     * @brief ネットワークを初期化する.
     * @param param ネットワークのパラメータ
     */
    bool Init(const NeuralNetParameter& param);

    /**
     * @brief 学習済みのモデルからネットワークを初期化する.
     * @param model_file_path モデルのファイル名
     */
    bool Init(const std::string& model_file_path);
};

}  // namespace ccilab

#endif  // NEURAL_NET_HPP_

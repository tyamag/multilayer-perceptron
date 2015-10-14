// exclusive or の学習

#include <iostream>

#include <NeuralNetLib/NeuralNet.hpp>

int main() {
    // ニューラルネットの構築
    ccilab::NeuralNetParameter net_param = {
        {2, 2, 1},    // 各層のノード数(入力層, 中間層, 出力層)
        {-0.1, 0.1},  // 重み初期値最小最大
        0.2,          // 学習率
    };
    ccilab::NeuralNet neural_net(net_param);

    // 入力データ
    std::vector<std::vector<double> > inputs_list = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    };

    // 正解データ
    std::vector<std::vector<double> > answers_list = {
        {0},
        {1},
        {1},
        {0},
    };

    // ネットワークの訓練
    const int num_epoch = 50000;
    neural_net.Train(inputs_list, answers_list, num_epoch);

    // 結果表示
    for (const auto& inputs : inputs_list) {
        const auto& outputs = neural_net.Foward(inputs);
        for (const auto& output : outputs) {
            std::cout << output << ", ";
        }
        std::cout << std::endl;
    }

    neural_net.SaveModel();

    return 0;
}

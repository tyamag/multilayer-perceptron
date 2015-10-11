// exclusive or �̊w�K

#include "NeuralNetwork.hpp"

#include <iostream>

int main() {
    // �j���[�����l�b�g�̍\�z
    ccilab::NeuralNetworkParameter net_param = {
        {2, 2, 1},    // �e�w�̃m�[�h��(���͑w, ���ԑw, �o�͑w)
        {-0.1, 0.1},  // �d�ݏ����l�ŏ��ő�
        0.2,          // �w�K��
    };
    ccilab::NeuralNetwork neural_net(net_param);

    // ���̓f�[�^
    std::vector<std::vector<double> > inputs_list = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    };

    // �����f�[�^
    std::vector<std::vector<double> > answers_list = {
        {0},
        {1},
        {1},
        {0},
    };

    // �l�b�g���[�N�̌P��
    const int num_epoch = 10000;
    neural_net.Train(inputs_list, answers_list, num_epoch);

    // ���ʕ\��
    for (const auto& inputs : inputs_list) {
        const auto& outputs = neural_net.Foward(inputs);
        for (const auto& output : outputs) {
            std::cout << output << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}

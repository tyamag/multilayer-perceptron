#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <utility>

#include "Layer.hpp"

namespace ccilab {

/**
 * @brief �j���[�����l�b�g���[�N�������ɗp����p�����[�^�[
 */
struct NeuralNetworkParameter {
    std::vector<int> num_nodes_list_;                 // �e�w�̃m�[�h��
    std::pair<double, double> initial_weight_minmax;  // �d�ݏ����l�̍ŏ�, �ő�
    double learning_rate_;                            // �w�K��
};

/**
 * @brief �j���[�����l�b�g���[�N.
 */
class NeuralNetwork {
public:
    /**
     * @brief �l�b�g���[�N�̃p�����[�^�����Ƃɏ���������
     * @param �l�b�g���[�N�̃p�����[�^
     */
    NeuralNetwork(const NeuralNetworkParameter& param);
    ~NeuralNetwork() {}

    /**
     * @brief �l�b�g���[�N���w�K����
     * @param inputs ���̓f�[�^
     * @param answers �����f�[�^
     * @param num_epoch �G�|�b�N��(�w�K��)
     */
    bool Train(const std::vector<std::vector<double> >& inputs_list,
        const std::vector<std::vector<double> >& answers_list, int num_epoch);

    /**
     * @brief �������`��.
     */
    const std::vector<double>& Foward(const std::vector<double>& inputs);

    /**
     * @brief �덷�t�`��.
     */
    void Backward(const std::vector<double>& answers);

    /**
     * @brief �덷���v�Z���ĕԂ�.
     */
    double CalculateError(const std::vector<double>& answers) const;

private:
    std::vector<Layer> layers_;  // ���C���[
    double learning_rate_;       // �w�K��

    /**
     * @brief �l�b�g���[�N������������.
     * @param param �l�b�g���[�N�̃p�����[�^
     */
    bool Init(const NeuralNetworkParameter& param);
};

}  // namespace ccilab

#endif  // NEURAL_NETWORK_HPP_

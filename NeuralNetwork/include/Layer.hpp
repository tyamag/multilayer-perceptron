#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <vector>

namespace ccilab {

/**
 * @brief �l�b�g���[�N���\�����郌�C���[(�w).
 */
class Layer {
public:
    explicit Layer(int num_nodes);
    ~Layer() {}

    /**
     * @brief ���C���[������������.
     * @param child_layer_nodes_num �q�w(���̑w)�̃m�[�h��
     * @param default_weight_min �d�ݏ����l�̍ŏ�
     * @param default_weight_max �d�ݏ����l�̍ő�
     */
    void Init(int child_layer_nodes_num, double default_weight_min, double default_weight_max);

    /**
     * @brief ���ׂẴm�[�h�̏o�͒l���v�Z��, �X�V����.
     *        ���͑w�̏o�͒l���X�V����Ƃ��Ɏg��.
     * @param inputs ���̓f�[�^
     */
    void CalculateOutputs(const std::vector<double>& inputs);

    /**
     * @brief ���ׂẴm�[�h�̏o�͒l���v�Z��, �X�V����.
     *        ���ԑw�`�o�͑w�̏o�͒l���X�V����Ƃ��Ɏg��.
     * @param parent_layer �e�w(�O�̑w)
     */
    void CalculateOutputs(const Layer& parent_layer);

    /**
     * @brief ���ׂẴm�[�h�̌덷�M�����v�Z��, �X�V����.
     *        �o�͑w�̌덷�M�����X�V����Ƃ��Ɏg��.
     * @param answers �����f�[�^
     */
    void CalculateErrors(const std::vector<double>& answers);

    /**
     * @brief ���ׂẴm�[�h�̌덷�M�����v�Z��, �X�V����.
     *        ���ԑw�̌덷�M�����X�V����Ƃ��Ɏg��.
     * @param child_layer �q�w
     */
    void CalculateErrors(const Layer& child_layer);

    /**
     * @brief ���ׂĂ̏d�݂��X�V����.
     * @param child_layer �q�w
     * @param learning_rate �w�K��
     */
    void UpdateWeights(const Layer& child_layer, double learning_rate);

    /**
     * @brief �m�[�h����Ԃ�.
     */
    int num_nodes() const { return num_nodes_; }

    /**
     * @brief �d�݂�Ԃ�.
     */
    const std::vector<std::vector<double> >& weights_list() const { return weights_list_; }

    /**
     * @brief �o�C�A�X��Ԃ�.
     */
    const std::vector<double>& bias_weights() const { return bias_weights_; }

    /**
     * @brief �o�͒l��Ԃ�.
     */
    const std::vector<double>& outputs() const { return outputs_; }

    /**
     * @brief �덷�M����Ԃ�.
     */
    const std::vector<double>& errors() const { return errors_; }

private:
    int num_nodes_;                             // �m�[�h��
    std::vector<std::vector<double> > weights_list_;  // �d��
    std::vector<double> bias_weights_;                // �o�C�A�X
    std::vector<double> outputs_;                     // �j���[�����̏o�͒l
    std::vector<double> errors_;                      // �덷�M��
};

}  // namespace ccilab

#endif  // LAYER_HPP_

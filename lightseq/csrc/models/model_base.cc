#include "model_base.h"
#include "iostream"

namespace lightseq {

GenerateConfig::GenerateConfig(int bos_id, int eos_id, int pad_id, float temperature, bool use_sampling,
                 float top_p, int top_k, int max_new_tokens):
        _eos_id(eos_id),
        _bos_id(bos_id),
        _pad_id(pad_id),
        _temperature(temperature),
        _top_p(top_p),
        _top_k(top_k),
        _use_sampling(use_sampling),
        _max_new_tokens(max_new_tokens) {}

void GenerateConfig::print_config() {
    std::cout << "****** GenerateConfig ******" << std::endl;
    std::cout << "_eos_id: " << _eos_id << std::endl;
    std::cout << "_bos_id: " << _bos_id << std::endl;
    std::cout << "_pad_id: " << _pad_id << std::endl;
    std::cout << "_temperature: " << _temperature << std::endl;
    if(_use_sampling) {
        std::cout << "_top_p: " << _top_p << std::endl;
        std::cout << "_top_k: " << _top_k << std::endl;
    }
    std::cout << "_max_new_tokens: " << _max_new_tokens << std::endl;
    std::cout << std::endl;
}

void LSModelFactory::ModelRegister(std::string class_name, LSModelConstructor obj) {
    if (obj) {
        object_map_.insert(std::map<std::string, LSModelConstructor>::value_type(
            class_name, obj));
    }
}

LSModel* LSModelFactory::CreateModel(std::string class_name, const std::string weight_path,
                    const int max_batch_size) {
    std::map<std::string, LSModelConstructor>::const_iterator iter =
        object_map_.find(class_name);
    if (iter != object_map_.end()) {
        return iter->second(weight_path, max_batch_size);
    } else {
        throw std::runtime_error("Model not supported");
    }
}

} // namespace lightseq 
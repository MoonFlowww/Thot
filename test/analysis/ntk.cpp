#include "../../include/Thot.h"

int main() {
    Thot::Model model("");
    model.use_cuda(torch::cuda::is_available());

    auto[x1, y1, x2, y2] = Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/Image/MNIST", 0.1f, 0.1f);


    return 0;
}
#include <torch/torch.h>

#include "../../../include/Thot.h"

int main() {
    Thot::Model model("");
    model.use_cuda(torch::cuda::is_available());

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::Universal(
    "/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages",
    Thot::Data::Type::JPG{"images", {}},
    Thot::Data::Type::PNG{"masks", {}});

    return 0;
}
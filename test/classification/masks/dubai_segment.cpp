#include "../../../include/Thot.h"

int main() {
    Thot::Model model("");
    model.use_cuda(torch::cuda::is_available());

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::Universal("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/",
            Thot::Data::Type::JPG{"images", {.normalize = true, .pad_to_max_tile=true}},
            Thot::Data::Type::PNG{"masks", {.normalize = true, .pad_to_max_tile=true}}, {.train_fraction = .8f, .test_fraction = .2f, .shuffle = true});

    Thot::Data::Check::Size(x1, "Inputs Raw");
    Thot::Data::Check::Size(y1, "Outputs Raw");
    Thot::Plot::Data::Image(x1, {1});
    Thot::Plot::Data::Image(y1, {1});
    //797 × 644 pixel

    Thot::Data::Transforms::Augmentation::AtmosphericDrift(x1, y1, )
    return 0;
}
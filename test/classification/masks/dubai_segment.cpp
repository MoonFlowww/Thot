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

    //797 × 644 pixel

    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::Flip(x1, y1, {.axes = {"x"}, .frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::Flip(x1, y1, {.axes = {"y"}, .frequency = 0.5f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::CLAHE(x1, y1, {.frequency = 0.5f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::OpticalDistortion(x1, y1, {.frequency = 0.3f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::AtmosphericDrift(x1, y1, {.frequency = 0.3f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::SunAngleJitter(x1, y1, {.frequency = 0.3f, .data_augment = true});

    Thot::Data::Check::Size(x1, "Inputs Raw");
    Thot::Data::Check::Size(y1, "Outputs Raw");

    return 0;
}
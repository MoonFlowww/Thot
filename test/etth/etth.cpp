#include "wavelet.hpp"
#include "../../include/Thot.h"
#include "../../src/data/details/manipulation/normalization/zscore.hpp"

int main() {
    Thot::Model model("Etth1");
    model.to_device(torch::cuda::is_available());

    auto [x1, y1, x2, y2] = Thot::Data::Load::ETTh("/home/moonfloww/Projects/DATASETS/ETT/ETTh1/ETTh1.csv", 0.5, 0.1f, true); // extract 60%
    Thot::Data::Check::Size(x1, "Raw");

    auto xzf1=Thot::Data::Normalization::Zscore(x1, {.lag=250, .temporal_dim=0, .forward_only=true});
    Thot::Data::Check::Size(xzf1, "Rolling");

    auto xzs1=Thot::Data::Normalization::Zscore(x1, {.lag=250, .temporal_dim=0, .forward_only=false});
    Thot::Data::Check::Size(xzs1, "Static");

    Thot::Plot::Data::Timeserie{x1};
    Thot::Plot::Data::Timeserie{xzs1};

    //auto res = thot::signal::wavelet::dwt(x1, /*levels=*/4);
    //auto files = dump_for_gnuplot("/home/moonfloww/CLionProjects/Thot/test/etth/out/wave", x1.to(at::kCPU), res);

    //std::cerr << "Wrote " << files.size() << " files.\n";

    return 0;
}

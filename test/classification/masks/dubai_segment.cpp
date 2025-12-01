#include "../../../include/Nott.h"
#include <array>
#include <iostream>
#include <tuple>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace {
    constexpr std::array<std::array<std::uint8_t, 3>, 6> kClassPalette{{
        std::array<std::uint8_t, 3>{60, 16, 152},
        std::array<std::uint8_t, 3>{110,193, 228},
        std::array<std::uint8_t, 3>{132, 41, 246},
        std::array<std::uint8_t, 3>{155, 155, 155},
        std::array<std::uint8_t, 3>{226, 169, 41},
        std::array<std::uint8_t, 3>{254, 221, 58},
    }};
    torch::Tensor ColorizeClassMap(const torch::Tensor& class_map) {
        TORCH_CHECK(class_map.dim() == 2, "Expected [H, W] class map");

        auto labels = class_map.to(torch::kLong).to(torch::kCPU).contiguous();
        const auto H = labels.size(0);
        const auto W = labels.size(1);

        torch::Tensor rgb = torch::zeros({H, W, 3}, torch::TensorOptions().dtype(torch::kUInt8));
        auto lacc = labels.accessor<int64_t, 2>();
        auto racc = rgb.accessor<std::uint8_t, 3>();

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                auto cls = lacc[y][x];
                if (cls < 0 || cls >= static_cast<long>(kClassPalette.size())) {
                    cls = 0;
                }
                const auto& color = kClassPalette[static_cast<std::size_t>(cls)];
                racc[y][x][0] = color[0];
                racc[y][x][1] = color[1];
                racc[y][x][2] = color[2];
            }
        }

        return rgb;
    }
    torch::Tensor ConvertRgbMasksToOneHot(const torch::Tensor& masks) {
        TORCH_CHECK(masks.dim() == 4, "Expected [B, 3, H, W]");
        TORCH_CHECK(masks.size(1) == 3, "RGB masks must have 3 channels");

        auto uint8_masks = masks.to(torch::kUInt8).contiguous();  // [B, 3, H, W]
        const auto B = uint8_masks.size(0);
        const auto H = uint8_masks.size(2);
        const auto W = uint8_masks.size(3);
        const auto C = static_cast<std::int64_t>(kClassPalette.size());

        // [B, 3, H, W] -> [B, H, W, 3] -> [N, 3]
        auto masks_flat = uint8_masks.permute({0, 2, 3, 1}).reshape({B * H * W, 3});

        std::vector<std::uint8_t> palette_vec;
        palette_vec.reserve(C * 3);
        for (const auto& rgb : kClassPalette) {
            palette_vec.push_back(rgb[0]);
            palette_vec.push_back(rgb[1]);
            palette_vec.push_back(rgb[2]);
        }

        auto palette = torch::tensor(palette_vec, torch::TensorOptions().dtype(torch::kUInt8)).view({C, 3}).to(uint8_masks.device());
        auto masks_i32   = masks_flat.to(torch::kInt32).unsqueeze(1); // [N, 1, 3]
        auto palette_i32 = palette.to(torch::kInt32).unsqueeze(0); // [1, C, 3]

        auto diff  = masks_i32 - palette_i32; // [N, C, 3], int32
        auto dist2 = diff.mul(diff).sum(-1); // [N, C],  int32

        auto min_idx = std::get<1>(dist2.min(1)); // [N]

        auto one_hot_flat = torch::nn::functional::one_hot(min_idx, C).to(torch::kFloat32); // [N, C]
        auto one_hot = one_hot_flat.view({B, H, W, C}).permute({0, 3, 1, 2}); // [B, C, H, W]

        return one_hot;
    }

    torch::Tensor OriginalTile(torch::Tensor&x) {
        std::vector<torch::Tensor> rows; rows.reserve(3);
        for (int i=0; i<3; ++i) {
            auto inp1 = x[3*i+0];
            auto inp2 = x[3*i+1];
            auto inp3 = x[3*i+2];

            torch::Tensor row_inp = torch::cat({inp1, inp2, inp3}, 2);
            rows.push_back(row_inp);
        }
        return torch::cat(rows, 1);
    }

    torch::Tensor Cuts(torch::Tensor& x, std::vector<int> cuts, bool subcuts=false) {
        int H = x.size(1); int W = x.size(2);
        int h=H/cuts[0]; int w=W/cuts[1]; //cuts resolution

        std::size_t samples = subcuts ? (cuts[0]*cuts[1])+((cuts[0]-1)*(cuts[1]-1)) : (cuts[0]*cuts[1]); // x² + (x-1)²
        std::vector<torch::Tensor> out; out.reserve(samples);

        for(int row=0; row<cuts[0]; ++row) { // classic cuts
            for(int col=0; col<cuts[1]; ++col) {
                torch::Tensor patch = x.index({
                    at::indexing::Slice(),
                    at::indexing::Slice(h*row, h*(row+1)),
                    at::indexing::Slice(w*col, w*(col+1))
                });

                out.push_back(patch);
            }
        }

        if (subcuts) {
            int hshift=h*0.5f; int wshift=w*0.5f;
            for(int row=0; row<cuts[0]-1; ++row) {
                for(int col=0; col<cuts[1]-1; ++col) {
                    torch::Tensor patch = x.index({
                        at::indexing::Slice(),
                        at::indexing::Slice(h*row+hshift, h*(row+1)+hshift),
                        at::indexing::Slice(w*col+wshift, w*(col+1)+wshift)
                    });

                    out.push_back(patch);
                }
            }
        }
        return torch::stack(out, 0);
    }

    std::vector<int> EstimCuts(int64_t h, int64_t w, int64_t target) {
        double rh = static_cast<double>(h) / static_cast<double>(target);
        double rw = static_cast<double>(w) / static_cast<double>(target);
        return {
            static_cast<int>(std::llround(rh)),
            static_cast<int>(std::llround(rw))
        };
    }
    std::vector<torch::Tensor> struct2train(auto samples) {
        std::vector<torch::Tensor> xs_sp; xs_sp.reserve(samples.size());
        std::vector<torch::Tensor> ys_sp; ys_sp.reserve(samples.size());
        for (auto& s : samples) {
            xs_sp.push_back(s.superpixel.input);
            ys_sp.push_back(s.superpixel.target);
        } return std::vector<torch::Tensor> {torch::cat(xs_sp, 0), torch::cat(ys_sp, 0)};
    }



    void savePic(torch::Tensor inp, const std::string path, const std::string kind) {
        torch::Tensor t = inp.detach().to(torch::kCPU);
        t = t.clamp(0, 255).to(torch::kU8);
        torch::Tensor t_hwc = t.permute({1, 2, 0}).contiguous();
        int H = t_hwc.size(0);
        int W = t_hwc.size(1);
        cv::Mat img(H, W, CV_8UC3, t_hwc.data_ptr<uint8_t>());
        cv::Mat img_bgr;
        cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(path+"/"+kind+"_raw.png", img_bgr);
        cv::GaussianBlur(img_bgr, img_bgr, cv::Size(), 0.83);
        cv::imwrite(path+"/"+kind+"_smooth.png", img_bgr);
    }
}


int main() {
    Nott::Model model("");
    std::vector<torch::Tensor> xs;
    std::vector<torch::Tensor> ys;

    for (int tile = 1; tile < 9; ++tile) {
        std::string path = "/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile " + std::to_string(tile);
        // cuts(3*3) -> Tile -> cuts(X*Y)
        auto [x1, y1, x2, y2] =
            Nott::Data::Load::Universal(path,
                Nott::Data::Type::JPG{"images", {.grayscale = false, .normalize_colors = false, .normalize_size = true, .color_order = "RGB"}},
                Nott::Data::Type::PNG{"masks",  {.normalize_colors = false, .normalize_size = true, .InterpolationMode = Nott::Data::Transform::Format::Options::InterpMode::Nearest, .color_order = "RGB"}},
                {.train_fraction = 1.f, .test_fraction = 0.f, .shuffle = false});

        x1 = OriginalTile(x1);
        Nott::Data::Check::Size(x1, ("X"+std::to_string(tile) + " Reconstructed Tile"));
        savePic(x1, path, "input");
        y1 = OriginalTile(y1);
        savePic(y1, path, "target");
        Nott::Data::Check::Size(x1, ("Y"+std::to_string(tile) + " Reconstructed Tile"));
        auto EstX= EstimCuts(x1.size(1), x1.size(2), 128);
        auto EstY= EstimCuts(y1.size(1), y1.size(2), 128);
        std::cout << "Estimator {X,Y}: " << EstX << " || n: " << EstX[0]*EstX[1] << " + " << (EstX[0]-1)*(EstX[1]-1) << std::endl;
        x1 = Cuts(x1, EstX, true); // Estim used to minimize artefacts from Downsample
        y1 = Cuts(y1, EstY, true);

        Nott::Data::Check::Size(x1, ("X"+std::to_string(tile) + " After cuts"));
        Nott::Data::Check::Size(y1, ("Y"+std::to_string(tile) + " After cuts"));
        x1 = Nott::Data::Transform::Format::Downsample(x1, {.size = {128, 128}, .interp = Nott::Data::Transform::Format::Options::InterpMode::Bilinear});
        y1 = Nott::Data::Transform::Format::Downsample(y1, {.size = {128, 128}, .interp = Nott::Data::Transform::Format::Options::InterpMode::Nearest});

        xs.push_back(x1);
        ys.push_back(y1);
        std::cout << "\n" <<std::endl;
    } std::cout << "\n" <<std::endl;
    torch::Tensor X = torch::cat(xs, 0);
    torch::Tensor Y = torch::cat(ys, 0);
    Nott::Data::Check::Size(X, "X Total");
    Nott::Data::Check::Size(Y, "Y Total");


    // std::tie(x1, y1) = Nott::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"x"}, .frequency = 1.f, .data_augment = true});
    // std::tie(x1, y1) = Nott::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"y"}, .frequency = 1.f, .data_augment = true});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, true, false});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, yz1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});


    auto Y_onehot = ConvertRgbMasksToOneHot(Y);

    auto block = [&](int in_c, int out_c) {
        return Nott::Block::Sequential({
            Nott::Layer::Conv2d(
                {in_c, out_c, {3,3}, {1,1}, {1,1}},
                Nott::Activation::ReLU,
                Nott::Initialization::HeUniform
            ),
            Nott::Layer::Conv2d(
                {out_c, out_c, {3,3}, {1,1}, {1,1}},
                Nott::Activation::ReLU,
                Nott::Initialization::HeUniform
            ),
        });
    };

    auto upblock = [&](int in_c, int out_c) {
        return Nott::Block::Sequential({
            Nott::Layer::Upsample({.scale = {2,2}, .mode  = Nott::UpsampleMode::Bilinear}),
            Nott::Layer::Conv2d(
                {in_c, out_c, {3,3}, {1,1}, {1,1}},
                Nott::Activation::ReLU,
                Nott::Initialization::HeUniform
            ),
        });
    };

    model.add(block(3, 64), "enc1");
    model.add(Nott::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool1");
    model.add(block(64,  128), "enc2");
    model.add(Nott::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool2");
    model.add(block(128, 256), "enc3");
    model.add(Nott::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool3");
    model.add(block(256, 512), "enc4");
    model.add(Nott::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool4");

    model.add(block(512, 1024), "bottleneck");

    model.add(upblock(1024, 512), "up4");
    model.add(block(1024, 512), "dec4");
    model.add(upblock(512, 256), "up3");
    model.add(block(512, 256), "dec3");
    model.add(upblock(256, 128), "up2");
    model.add(block(256, 128),"dec2");
    model.add(upblock(128, 64), "up1");
    model.add(block(128, 64), "dec1");

    model.add(Nott::Layer::Conv2d({64, 6, {1, 1}, {1, 1}, {0, 0}}, Nott::Activation::Identity), "logits");

    model.links({
        Nott::LinkSpec{Nott::Port::Input("@input"), Nott::Port::Module("enc1")},
        Nott::LinkSpec{Nott::Port::Module("enc1"),  Nott::Port::Module("pool1")},
        Nott::LinkSpec{Nott::Port::Module("pool1"), Nott::Port::Module("enc2")},
        Nott::LinkSpec{Nott::Port::Module("enc2"),  Nott::Port::Module("pool2")},
        Nott::LinkSpec{Nott::Port::Module("pool2"), Nott::Port::Module("enc3")},
        Nott::LinkSpec{Nott::Port::Module("enc3"),  Nott::Port::Module("pool3")},
        Nott::LinkSpec{Nott::Port::Module("pool3"), Nott::Port::Module("enc4")},
        Nott::LinkSpec{Nott::Port::Module("enc4"),  Nott::Port::Module("pool4")},
        Nott::LinkSpec{Nott::Port::Module("pool4"), Nott::Port::Module("bottleneck")},

        Nott::LinkSpec{Nott::Port::Module("bottleneck"), Nott::Port::Module("up4")},
        Nott::LinkSpec{Nott::Port::Join({"up4", "enc4"}, Nott::MergePolicy::Stack), Nott::Port::Module("dec4")},
        Nott::LinkSpec{Nott::Port::Module("dec4"), Nott::Port::Module("up3")},
        Nott::LinkSpec{Nott::Port::Join({"up3", "enc3"}, Nott::MergePolicy::Stack), Nott::Port::Module("dec3")},

        Nott::LinkSpec{Nott::Port::Module("dec3"), Nott::Port::Module("up2") },
        Nott::LinkSpec{Nott::Port::Join({"up2", "enc2"}, Nott::MergePolicy::Stack), Nott::Port::Module("dec2")},
        Nott::LinkSpec{Nott::Port::Module("dec2"), Nott::Port::Module("up1") },
        Nott::LinkSpec{Nott::Port::Join({"up1", "enc1"}, Nott::MergePolicy::Stack), Nott::Port::Module("dec1")},

        Nott::LinkSpec{Nott::Port::Module("dec1"),   Nott::Port::Module("logits")}, // dec3
        Nott::LinkSpec{Nott::Port::Module("logits"), Nott::Port::Output("@output")},
    }, true);




    Y = ConvertRgbMasksToOneHot(Y);
    X= X.to(torch::kFloat32) / 255.0f;

    Nott::Plot::Data::Image(X, {0, 100, 1000, 2000, 5000, 11734});

    const auto total_training_samples = X.size(0);
    const auto B = 32;
    const auto E = 5;
    const auto steps_per_epoch = static_cast<std::size_t>((total_training_samples + B - 1) / B);
    const auto total_training_steps = std::max<std::size_t>(1, E * std::max<std::size_t>(steps_per_epoch, 1));

    //loss weights
    auto class_targets = Y.argmax(1);
    auto class_counts  = torch::bincount(class_targets.flatten(), std::nullopt, kClassPalette.size()).to(torch::kFloat32);
    auto class_weights = (class_counts + 1e-6f).reciprocal();
    class_weights = class_weights / class_weights.mean();
    auto w_cpu = class_weights.to(torch::kCPU).to(torch::kDouble).contiguous();
    const double* ptr = w_cpu.data_ptr<double>();
    std::vector<double> w(ptr, ptr + w_cpu.numel());


    model.set_loss(Nott::Loss::CrossEntropy({ /*.weight = w*/ }));
    model.set_optimizer(
        Nott::Optimizer::AdamW({.learning_rate = 5e-5, .weight_decay = 1e-4}),
        Nott::LrScheduler::CosineAnnealing({
            .T_max = (total_training_steps),
            .eta_min = 1e-6,
            .warmup_steps = std::min<std::size_t>(steps_per_epoch * 5, total_training_steps / 5),
            .warmup_start_factor = 0.1,
        }));

    model.set_regularization({Nott::Regularization::SWAG({
        .coefficient = 5e-4,
        .variance_epsilon = 1e-6,
        .start_step = static_cast<std::size_t>(0.65 * static_cast<double>(total_training_steps)),
        .accumulation_stride = std::max<std::size_t>(1, steps_per_epoch),
        .max_snapshots = 20,
    })});

    model.use_cuda(torch::cuda::is_available());


    // std::tie(X, Y) = Nott::Data::Manipulation::Fraction(X, Y, 0.1f); // 10%
    // Nott::Data::Check::Size(X, "10% X Train");
    model.train(X, Y,
        {.epoch = E,
        .batch_size = B,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{X, Y},
        .graph_mode = Nott::GraphMode::Capture,
        .enable_amp = true,
        // .memory_format = torch::MemoryFormat::ChannelsLast
        });

    model.evaluate(X, Y, Nott::Evaluation::Segmentation,{
        Nott::Metric::Classification::Accuracy,
        Nott::Metric::Classification::Precision,
        Nott::Metric::Classification::Recall,
        Nott::Metric::Classification::JaccardIndexMicro,
        Nott::Metric::Classification::BoundaryIoU,
        Nott::Metric::Classification::HausdorffDistance,
    },{.batch_size = B, .buffer_vram=2});


    std::tie(X, Y) = Nott::Data::Manipulation::Fraction(X, Y, 0.01f);
    Nott::Data::Check::Size(X, "Inference");
    auto logits = model.forward(X);
    auto predicted = logits.argmax(1).to(torch::kCPU);
    auto first_pred = predicted.index({0}).contiguous();
    auto forecast_rgb = ColorizeClassMap(first_pred);

    auto ground_truth = Y.argmax(1).to(torch::kCPU);
    auto gt_rgb = ColorizeClassMap(ground_truth.index({0}).contiguous());

    Nott::Plot::Data::Image(forecast_rgb, {0});
    Nott::Plot::Data::Image(gt_rgb, {0});
    Nott::Plot::Data::Image(X, {0});
    // cv::Mat f_img(H, W, CV_8UC3, forecast_rgb.data_ptr<uint8_t>());
    // cv::Mat f_bgr;
    // cv::cvtColor(f_img, f_bgr, cv::COLOR_RGB2BGR);
    // cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/forecast_raw.png", f_bgr);
    // cv::GaussianBlur(f_bgr, f_bgr, cv::Size(), 0.83);
    // cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/forecast_smooth.png", f_bgr);

    return 0;
}


/*
 TODO: Use Hyperpixel from new cuts over Tiles
    Superpixels over cuts to move from H*W*3(Input) and H*W*6(output) to 5*3(input metrics) and 6 (argmax label output)
    compute cuts superpixel over B, and then in async Smoothing fractal to recreate Cuts labeling from Superpixel to Cuts and then from Cuts to Tile


    Target Skeleton:
    1) Async Cuts from Tiles
    2) Async Superpixels from Cuts
    3) Batchs Superpixels through Net via Metrics (Max, Min, Avg, Std, Skew; for R,G,B)
    4) Async Reconstruction of Cuts from Superpixels + Smoother
    5) Async Reconstruction of Tiles from Cuts + Smoother



    Size of the network will be so small that we will be able to compute mutliple Cuts at once.
    But low size Cuts help for K definition in Voronoi and Argmax over Superpixel labels

    Input&Output Improvement:
        Dimensions:
            Before I: 128*128*3 = 84,672  ||  O: 128*128*6 = 169,344
            After I: 5*3 = 15  ||  O: 6
            IMPROVEMENT I: 14111x  ||  O: 28223x
        Accuracy:
            Can't quantify, since high quantity of data is required (i don't have) + highly unbalanced class and TPR&TNR are treated egaly in loss, network learn how to avoid low-freq(complex) labels by never predicting them


    Reason:
        Tile reconstruction:
            Default Cuts:
                - More base-samples [Acc]
                - Lower Amount of Details [Acc]
                - Maintaining the Quality of Information [Acc] [(x/T)&(y/T)] -> bilinear
                - Lower Resolution [Latency]

            SubCuts:
                - Maximization of Base-Samples [Acc]
                - Redundant information can only appear twice in the dataset, but never at the same position on the image (always TopLeft vs BottomRight || TopRight vs BottomLeft)  leading to P(Overfit)⁻ + Conv=Sum[Freq(window)]=> taking neighbors and not all neurons to all neurons [Acc]

        Geodesic-Based Voronoi Diagram:
            - Able to use metrics as input (no Pics) [Lat+Acc]
                -> 14111 times less input neurons
            - Geodesic lead to P(Unique Class)^+ per superpixel, to use argmax(superpixel) leading to 6 output neurones [Lat+Acc]
                -> 28223 times less output neurons
            - Easy fix of argmax via smoothing during Cuts recreation (Async Post Process) [Acc]



        In Production:
            - Easy setup of asyncs for data Pre/Post-process
            - Superpixel computed by batches (With Hardware, even able to compute Superpixels from multiples Cuts inside a unique Batch)
            - Lightning fast Infra
            - Readable Infra (Data Cutting & Geodesic-Voronoi)
            - Voronoi make pre-segmentation of labels without Learning nor Data-Leakage




    Geodesic Distance Smoother:
        - RPCA
        - FFT + PSD
        - FFT_coef(0.05)
        - Marcenko Pastur w/ Eigen filter ?
    NB: Compute it over tile directly to make it faster
 */



/* // Test is same as train for Nabla bulbe
Epoch [1/50] | Train loss: 0.217209 | Test loss: 0.165778 | ΔLoss: N/A (∇) | duration: 95.63sec
Epoch [2/50] | Train loss: 0.163068 | Test loss: 0.154031 | ΔLoss: -0.011747 (∇) | duration: 95.79sec
Epoch [3/50] | Train loss: 0.152796 | Test loss: 0.149430 | ΔLoss: -0.004602 (∇) | duration: 95.11sec
Epoch [4/50] | Train loss: 0.146659 | Test loss: 0.145577 | ΔLoss: -0.003852 (∇) | duration: 94.93sec
Epoch [5/50] | Train loss: 0.140482 | Test loss: 0.137337 | ΔLoss: -0.008241 (∇) | duration: 95.01sec
Epoch [6/50] | Train loss: 0.130418 | Test loss: 0.124928 | ΔLoss: -0.012409 (∇) | duration: 93.56sec
Epoch [7/50] | Train loss: 0.122049 | Test loss: 0.116866 | ΔLoss: -0.008062 (∇) | duration: 93.50sec
Epoch [8/50] | Train loss: 0.115397 | Test loss: 0.105388 | ΔLoss: -0.011478 (∇) | duration: 93.50sec
Epoch [9/50] | Train loss: 0.114268 | Test loss: 0.127917 | ΔLoss: +0.022529 (∇) | duration: 93.93sec
Epoch [10/50] | Train loss: 0.110198 | Test loss: 0.095986 | ΔLoss: -0.009402 (∇) | duration: 105.43sec
Epoch [11/50] | Train loss: 0.101153 | Test loss: 0.103741 | ΔLoss: +0.007755 (∇) | duration: 106.69sec
Epoch [12/50] | Train loss: 0.092748 | Test loss: 0.086289 | ΔLoss: -0.009697 (∇) | duration: 106.17sec
Epoch [13/50] | Train loss: 0.088982 | Test loss: 0.086340 | ΔLoss: +0.000051 (∇) | duration: 103.05sec
Epoch [14/50] | Train loss: 0.081105 | Test loss: 0.073103 | ΔLoss: -0.013187 (∇) | duration: 103.34sec
Epoch [15/50] | Train loss: 0.078934 | Test loss: 0.072175 | ΔLoss: -0.000927 (∇) | duration: 104.06sec
Epoch [16/50] | Train loss: 0.075805 | Test loss: 0.091221 | ΔLoss: +0.019045 (∇) | duration: 96.86sec
Epoch [17/50] | Train loss: 0.070581 | Test loss: 0.065021 | ΔLoss: -0.007154 (∇) | duration: 100.52sec
Epoch [18/50] | Train loss: 0.072420 | Test loss: 0.064317 | ΔLoss: -0.000704 (∇) | duration: 95.38sec
Epoch [19/50] | Train loss: 0.071357 | Test loss: 0.065981 | ΔLoss: +0.001664 (∇) | duration: 98.08sec
Epoch [20/50] | Train loss: 0.063595 | Test loss: 0.055153 | ΔLoss: -0.009164 (∇) | duration: 94.51sec
Epoch [21/50] | Train loss: 0.053435 | Test loss: 0.048825 | ΔLoss: -0.006328 (∇) | duration: 96.21sec
Epoch [22/50] | Train loss: 0.049913 | Test loss: 0.046706 | ΔLoss: -0.002120 (∇) | duration: 93.60sec
Epoch [23/50] | Train loss: 0.047069 | Test loss: 0.042982 | ΔLoss: -0.003723 (∇) | duration: 93.59sec
Epoch [24/50] | Train loss: 0.042955 | Test loss: 0.040060 | ΔLoss: -0.002923 (∇) | duration: 94.58sec
Epoch [25/50] | Train loss: 0.040667 | Test loss: 0.038410 | ΔLoss: -0.001649 (∇) | duration: 95.96sec
Epoch [26/50] | Train loss: 0.037267 | Test loss: 0.034624 | ΔLoss: -0.003786 (∇) | duration: 98.51sec
Epoch [27/50] | Train loss: 0.037847 | Test loss: 0.033765 | ΔLoss: -0.000859 (∇) | duration: 98.19sec
Epoch [28/50] | Train loss: 0.033383 | Test loss: 0.029803 | ΔLoss: -0.003962 (∇) | duration: 93.50sec
Epoch [29/50] | Train loss: 0.030588 | Test loss: 0.027783 | ΔLoss: -0.002021 (∇) | duration: 94.13sec
Epoch [30/50] | Train loss: 0.029718 | Test loss: 0.027049 | ΔLoss: -0.000734 (∇) | duration: 93.50sec
Epoch [31/50] | Train loss: 0.026721 | Test loss: 0.024883 | ΔLoss: -0.002165 (∇) | duration: 93.81sec
Epoch [32/50] | Train loss: 0.024736 | Test loss: 0.022889 | ΔLoss: -0.001994 (∇) | duration: 93.51sec
Epoch [33/50] | Train loss: 0.023193 | Test loss: 0.021997 | ΔLoss: -0.000892 (∇) | duration: 93.47sec
Epoch [34/50] | Train loss: 0.021912 | Test loss: 0.022896 | ΔLoss: +0.000898 (∇) | duration: 96.04sec
Epoch [35/50] | Train loss: 0.021031 | Test loss: 0.019439 | ΔLoss: -0.002558 (∇) | duration: 96.07sec
Epoch [36/50] | Train loss: 0.019589 | Test loss: 0.019446 | ΔLoss: +0.000007 (∇) | duration: 96.08sec
Epoch [37/50] | Train loss: 0.018850 | Test loss: 0.018018 | ΔLoss: -0.001421 (∇) | duration: 96.09sec
Epoch [38/50] | Train loss: 0.017995 | Test loss: 0.016745 | ΔLoss: -0.001272 (∇) | duration: 96.09sec
Epoch [39/50] | Train loss: 0.017139 | Test loss: 0.016592 | ΔLoss: -0.000154 (∇) | duration: 96.09sec
Epoch [40/50] | Train loss: 0.016656 | Test loss: 0.015644 | ΔLoss: -0.000948 (∇) | duration: 97.35sec
Epoch [41/50] | Train loss: 0.016203 | Test loss: 0.015593 | ΔLoss: -0.000050 (∇) | duration: 97.24sec
Epoch [42/50] | Train loss: 0.015494 | Test loss: 0.014841 | ΔLoss: -0.000752 (∇) | duration: 97.25sec
Epoch [43/50] | Train loss: 0.014906 | Test loss: 0.014285 | ΔLoss: -0.000557 (∇) | duration: 97.66sec
Epoch [44/50] | Train loss: 0.014457 | Test loss: 0.013886 | ΔLoss: -0.000399 (∇) | duration: 97.23sec
Epoch [45/50] | Train loss: 0.014038 | Test loss: 0.013522 | ΔLoss: -0.000364 (∇) | duration: 97.23sec
Epoch [46/50] | Train loss: 0.013667 | Test loss: 0.013494 | ΔLoss: -0.000028 (∇) | duration: 97.39sec
Epoch [47/50] | Train loss: 0.013382 | Test loss: 0.012975 | ΔLoss: -0.000519 (∇) | duration: 96.60sec
Epoch [48/50] | Train loss: 0.013139 | Test loss: 0.012817 | ΔLoss: -0.000158 (∇) | duration: 96.08sec
Epoch [49/50] | Train loss: 0.012878 | Test loss: 0.012620 | ΔLoss: -0.000197 (∇) | duration: 96.08sec
Epoch [50/50] | Train loss: 0.012654 | Test loss: 0.012409 | ΔLoss: -0.000211 (∇) | duration: 97.82sec
[Nott] Reloading best state of the network...
End
Test Inputs: (11735, 3, 128, 128)
Test Targets: (11735, 128, 128)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation: Classification ┃          ┃                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Metric                ┃    Macro ┃ Weighted (support) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Accuracy              ┃ 0.982847 ┃           0.970561 ┃
┃ Precision             ┃ 0.895484 ┃           0.953850 ┃
┃ Recall                ┃ 0.971272 ┃           0.948542 ┃
┃ Jaccard index (micro) ┃ 0.902121 ┃           0.902121 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛

Overall accuracy (micro): 0.948542

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Per-class             ┃          ┃          ┃           ┃          ┃          ┃          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Metric                ┃  Label 0 ┃  Label 1 ┃   Label 2 ┃  Label 3 ┃  Label 4 ┃  Label 5 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Support               ┃ 26572179 ┃ 17905191 ┃ 100457165 ┃   927471 ┃ 26449061 ┃ 19955173 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Accuracy              ┃ 0.984740 ┃ 0.974361 ┃  0.955777 ┃ 0.998528 ┃ 0.996012 ┃ 0.987667 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Precision             ┃ 0.922010 ┃ 0.798134 ┃  0.992994 ┃ 0.766406 ┃ 0.978783 ┃ 0.914579 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Recall                ┃ 0.971787 ┃ 0.970029 ┃  0.921864 ┃ 0.999481 ┃ 0.992522 ┃ 0.971949 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Jaccard index (micro) ┃ 0.897973 ┃ 0.778926 ┃  0.915907 ┃ 0.766102 ┃ 0.971617 ┃ 0.891059 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┛




// Only 6 epochs over 50 led to non-improvement
*/





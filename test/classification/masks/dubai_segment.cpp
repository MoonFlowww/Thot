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
    struct PQNode {
        float dist;
        int   idx;
        bool operator>(const PQNode& other) const { return dist > other.dist; }
    };

    struct Center {
        float x;
        float y;
    };

    struct Sample {
        struct SuperPixel {
            torch::Tensor input;
            torch::Tensor target;
        };

        template<int R, int G, int B>
        struct Metrics {
            torch::Tensor min;
            torch::Tensor max;
            torch::Tensor avg;
            torch::Tensor std;
            torch::Tensor skew;
        };

        SuperPixel superpixel;

        Metrics<0, 0, 0> metric;
        torch::Tensor NormTarget;
    };
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


    std::pair<torch::Tensor, torch::Tensor> RunGeodesicVoronoi(const torch::Tensor& density_hw, const std::vector<Center>& centers){
        auto D = density_hw.to(torch::kFloat32).to(torch::kCPU).contiguous();
        const int64_t H = D.size(0);
        const int64_t W = D.size(1);
        const int64_t N = static_cast<int64_t>(centers.size());

        TORCH_CHECK(N > 0, "RunGeodesicVoronoi: no centers");

        // Total density -> average A from Eq. (4)
        float total_D = D.sum().item<float>();
        float A = total_D / static_cast<float>(N);

        // Gaussian for the structure term (Eq. (10))
        auto gaussian0 = [A](float x) {
            // σ0 = A / α; paper uses α≈0.5–1; choose 0.5 as default
            constexpr float alpha = 0.5f;
            float sigma0 = A / alpha;
            if (sigma0 <= 1e-6f) return 1.0f;
            float z = x / sigma0;
            // un-normalized Gaussian G^0_{σ0}
            return std::exp(-0.5f * z * z);
        };

        auto labels = torch::full({H, W}, -1,
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
        auto dist   = torch::full({H, W}, std::numeric_limits<float>::infinity(),
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        auto lacc = labels.accessor<int64_t, 2>();
        auto tacc = dist.accessor<float, 2>();
        auto dacc = D.accessor<float, 2>();

        // area A_l(d) accumulated per label l (Eq. (10))
        std::vector<float> area_l(N, 0.0f);

        using Node = std::tuple<float,int64_t,int64_t,int64_t>; // (dist,y,x,l)
        struct Cmp {
            bool operator()(Node const& a, Node const& b) const {
                return std::get<0>(a) > std::get<0>(b);
            }
        };
        std::priority_queue<Node,std::vector<Node>,Cmp> pq;

        // Initialize with centers
        for (int64_t l = 0; l < N; ++l) {
            int64_t cy = centers[l].y;
            int64_t cx = centers[l].x;
            if (cy < 0 || cy >= H || cx < 0 || cx >= W) continue;
            lacc[cy][cx] = l;
            tacc[cy][cx] = 0.0f;
            pq.emplace(0.0f, cy, cx, l);
        }

        const int dy[4] = {+1,-1,0,0};
        const int dx[4] = {0,0,+1,-1};

        while (!pq.empty()) {
            auto [cd, y, x, l] = pq.top();
            pq.pop();
            if (cd > tacc[y][x]) continue;

            // update area_l with current pixel (Eq. (10))
            area_l[l] += dacc[y][x];  // ∫ D(x) dx, pixel area=1

            for (int k = 0; k < 4; ++k) {
                int64_t ny = y + dy[k];
                int64_t nx = x + dx[k];
                if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;

                float Dnbr = dacc[ny][nx];
                if (!std::isfinite(Dnbr)) continue;

                // base velocity V(x) = 1 / D(x)  (Eq. (8))
                float Vx = 1.0f / (Dnbr + 1e-6f);

                // structure term: Vl(x,d) = V(x) * G^0_{σ0}(max(0,A_l(d)-A))
                float deltaA = std::max(0.0f, area_l[l] - A);
                float Vl = Vx * gaussian0(deltaA);

                // Eikonal step: Δd ≈ step_length / Vl  (Eq. (11))
                float step = 1.0f;  // grid spacing
                float nd   = cd + step / (Vl + 1e-6f);

                if (nd < tacc[ny][nx]) {
                    tacc[ny][nx] = nd;
                    lacc[ny][nx] = l;
                    pq.emplace(nd, ny, nx, l);
                }
            }
        }

        return {labels, dist};
    }


    // ==========================================
    // Structure-sensitive density D(x)
    // - grayscale
    // - gradient magnitude
    // - local average of gradient (box filter)
    // - E(x) = grad / (avg_grad + gamma)
    // - D(x) = exp(E / nu)
    // ==========================================
    inline torch::Tensor gaussian_kernel_1d(float sigma, int radius) {
        TORCH_CHECK(sigma > 0.0f, "sigma must be > 0");
        TORCH_CHECK(radius > 0, "radius must be > 0");

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto x = torch::arange(-radius, radius + 1, options);
        auto k = torch::exp(-(x * x) / (2.0f * sigma * sigma));
        k /= k.sum();
        return k;
    }

    inline torch::Tensor gaussian_blur_2d(const torch::Tensor& img_hw, float sigma) {
        const int radius = static_cast<int>(std::ceil(3.0f * sigma));
        auto k1d = gaussian_kernel_1d(sigma, radius);
        auto kx  = k1d.view({1, 1, 1, -1});
        auto ky  = k1d.view({1, 1, -1, 1});

        auto x = img_hw.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);

        using namespace torch::nn::functional;
        Conv2dFuncOptions opt_x;
        opt_x = opt_x.padding({radius, 0});
        auto tmp = conv2d(x, kx, opt_x);

        Conv2dFuncOptions opt_y;
        opt_y = opt_y.padding({0, radius});
        auto y = conv2d(tmp, ky, opt_y);

        return y.squeeze(0).squeeze(0);
    }

    torch::Tensor ComputeDensity(const torch::Tensor& img_chw) {
        TORCH_CHECK(img_chw.dim() == 3 && img_chw.size(0) == 3,
                    "ComputeDensity expects RGB [3,H,W]");

        auto img = img_chw.to(torch::kFloat32).to(torch::kCPU).contiguous();
        const int64_t C = img.size(0);
        const int64_t H = img.size(1);
        const int64_t W = img.size(2);

        // 1) gradient magnitude ||∇I|| over color channels
        auto grad_mag = torch::zeros({H, W},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        auto iacc = img.accessor<float, 3>();
        auto gacc = grad_mag.accessor<float, 2>();

        for (int64_t y = 1; y + 1 < H; ++y) {
            for (int64_t x = 1; x + 1 < W; ++x) {
                float gx2 = 0.0f;
                float gy2 = 0.0f;
                for (int64_t c = 0; c < C; ++c) {
                    float left   = iacc[c][y][x - 1];
                    float right  = iacc[c][y][x + 1];
                    float up     = iacc[c][y - 1][x];
                    float down   = iacc[c][y + 1][x];
                    float gx     = 0.5f * (right - left);
                    float gy     = 0.5f * (down - up);
                    gx2         += gx * gx;
                    gy2         += gy * gy;
                }
                gacc[y][x] = std::sqrt(gx2 + gy2);
            }
        }

        // 2) normalized edge magnitude E(x)
        constexpr float sigma = 1.0f; // G_σ in paper (tune if needed)
        constexpr float gamma = 1e-3f; // to ignore very weak edges
        auto g_smooth = gaussian_blur_2d(grad_mag, sigma); // Gσ * ||∇I||

        auto denom = g_smooth + gamma;
        auto E = grad_mag / denom;

        // 3) density D(x) = exp(E(x) / ν)
        constexpr float nu = 0.25f;        // scaling parameter (paper)
        auto D = torch::exp(E / nu);       // [H, W]

        return D;
    }

    std::vector<Center> PlaceInitialSeeds(int64_t H, int64_t W, int num_superpixels)
    {
        const int64_t N = std::max<int64_t>(1, num_superpixels);

        // grid size (Sx * Sy ≈ N)
        int64_t Sx = static_cast<int64_t>(
            std::round(std::sqrt(static_cast<double>(N) *
                                 static_cast<double>(W) /
                                 std::max<int64_t>(1, H))));
        Sx = std::max<int64_t>(1, Sx);
        int64_t Sy = (N + Sx - 1) / Sx;

        const float step_x = static_cast<float>(W) / static_cast<float>(Sx);
        const float step_y = static_cast<float>(H) / static_cast<float>(Sy);

        std::mt19937 rng(12345u);
        std::uniform_real_distribution<float> jitter(-0.25f, 0.25f); // ±25% cell

        std::vector<Center> centers;
        centers.reserve(static_cast<std::size_t>(N));

        int64_t idx = 0;
        for (int64_t j = 0; j < Sy; ++j) {
            for (int64_t i = 0; i < Sx; ++i) {
                if (idx >= N) break;

                float cx = (i + 0.5f + jitter(rng)) * step_x;
                float cy = (j + 0.5f + jitter(rng)) * step_y;

                // clamp to valid pixel range, but keep as float
                cx = std::clamp(cx, 0.0f, static_cast<float>(W - 1));
                cy = std::clamp(cy, 0.0f, static_cast<float>(H - 1));

                centers.push_back(Center{cx, cy});
                ++idx;
            }
            if (idx >= N) break;
        }

        return centers;
    }



    torch::Tensor GenerateSpeedField(const torch::Tensor& density_hw) {
        TORCH_CHECK(density_hw.dim() == 2, "Expected [H, W] density");
        return density_hw.to(torch::kFloat32).to(torch::kCPU).contiguous();
    }

    std::tuple<torch::Tensor, torch::Tensor> EvolveBoundaries(
        const torch::Tensor& speed_field,
        const std::vector<Center>& centers) {
        return RunGeodesicVoronoi(speed_field, centers);
    }

    struct CenterUpdateResult {
        std::vector<Center> centers;
        bool split_performed;
    };


    CenterUpdateResult RelocateAndSplitCenters(const torch::Tensor& labels, const torch::Tensor& geodesic, const torch::Tensor& speed_field, const std::vector<Center>& init_centers,
            int target_superpixels, int max_splits_per_iter, float phi, float Ts, float Tc, float eps, int post_iterations) {
        std::vector<Center> centers = init_centers;

        const int64_t H = speed_field.size(0);
        const int64_t W = speed_field.size(1);

        auto lacc = labels.accessor<int64_t, 2>();
        auto gacc = geodesic.accessor<float, 2>();
        auto dacc = speed_field.accessor<float, 2>();

        //(last iter)
        bool did_any_split = false;

        for (int iter = 0; iter < post_iterations; ++iter){
            const int S = static_cast<int>(centers.size());

            std::vector<double> area(S, 0.0);
            std::vector<double> sum_x(S, 0.0), sum_y(S, 0.0), sum_w(S, 0.0);
            std::vector<double> Sxx(S, 0.0), Syy(S, 0.0), Sxy(S, 0.0), Sw_shape(S, 0.0);

            for (int64_t y = 0; y < H; ++y)
            {
                for (int64_t x = 0; x < W; ++x)
                {
                    int l = lacc[y][x];
                    if (l < 0 || l >= S) continue;

                    float D  = dacc[y][x];  // local density
                    float dg = gacc[y][x];  // geodesic distance

                    area[l] += static_cast<double>(D);

                    float cx = centers[l].x;
                    float cy = centers[l].y;
                    float dx = float(x) - cx;
                    float dy = float(y) - cy;
                    float len = std::sqrt(dx*dx + dy*dy) + eps;

                    float W_x = std::exp(-dg / phi); // relocation weight
                    float w_center = W_x * dg / len;

                    sum_x[l] += w_center * x;
                    sum_y[l] += w_center * y;
                    sum_w[l] += w_center;

                    float w_shape = dg * dg * (dx*dx + dy*dy);
                    Sxx[l] += w_shape * dx * dx;
                    Syy[l] += w_shape * dy * dy;
                    Sxy[l] += w_shape * dx * dy;
                    Sw_shape[l] += w_shape;
                }
            }

            double total_area = 0.0;
            for (int l = 0; l < S; ++l) total_area += area[l];
            double A_mean = (S > 0 ? total_area / S : 0.0);

            // Center Relloçcation
            std::vector<Center> relocated;
            relocated.reserve(S);

            for (int l = 0; l < S; ++l)
            {
                if (sum_w[l] > 0.0) {
                    float nx = sum_x[l] / sum_w[l];
                    float ny = sum_y[l] / sum_w[l];

                    nx = std::clamp(nx, 0.0f, float(W - 1));
                    ny = std::clamp(ny, 0.0f, float(H - 1));

                    relocated.push_back({nx, ny});
                } else {
                    relocated.push_back(centers[l]);
                }
            }

            // metrics for split
            std::vector<int> to_split;

            for (int l = 0; l < S; ++l)
            {
                if (Sw_shape[l] <= 0.0) continue;

                double cov_xx = Sxx[l] / Sw_shape[l];
                double cov_yy = Syy[l] / Sw_shape[l];
                double cov_xy = Sxy[l] / Sw_shape[l];

                double trace = cov_xx + cov_yy;
                double det   = cov_xx * cov_yy - cov_xy * cov_xy;
                double tmp   = std::sqrt(std::max(0.0, trace*trace*0.25 - det));

                double eig1  = trace*0.5 + tmp;
                double eig2  = trace*0.5 - tmp;
                if (eig2 <= 0.0) eig2 = 1e-12;

                double shape_ratio = eig1 / eig2;
                double size_ratio  = (A_mean > 0.0 ? area[l] / A_mean : 0.0);

                if (shape_ratio > Tc || size_ratio > Ts)
                    to_split.push_back(l);
            }

            int remaining_budget = std::max(0, target_superpixels - S);
            int allowed = std::min(remaining_budget, max_splits_per_iter);

            if (allowed <= 0 || to_split.empty()) {
                // no more splits possible
                centers = relocated;
                break;
            }

            std::sort(to_split.begin(), to_split.end(),
                [&](int a, int b){ return area[a] > area[b]; });

            if (int(to_split.size()) > allowed)
                to_split.resize(allowed);

            did_any_split = true;

            // Center Splitting (iter)
            std::vector<char> marked(S, 0);
            for (int idx : to_split) marked[idx] = 1;

            std::vector<Center> next_centers;
            next_centers.reserve(S + to_split.size());

            // keep all non-splitted centers
            for (int l = 0; l < S; ++l)
                if (!marked[l]) next_centers.push_back(relocated[l]);

            for (int l : to_split)
            {
                if (Sw_shape[l] <= 0.0)
                {
                    next_centers.push_back(relocated[l]);
                    continue;
                }

                double cx = relocated[l].x;
                double cy = relocated[l].y;

                // eigenvector for split direction
                double cov_xx = Sxx[l] / Sw_shape[l];
                double cov_yy = Syy[l] / Sw_shape[l];
                double cov_xy = Sxy[l] / Sw_shape[l];

                double trace = cov_xx + cov_yy;
                double det   = cov_xx * cov_yy - cov_xy * cov_xy;
                double tmp   = std::sqrt(std::max(0.0, trace*trace*0.25 - det));
                double eig1  = trace*0.5 + tmp;

                double nx = cov_xy;
                double ny = eig1 - cov_xx;
                double nlen = std::sqrt(nx*nx + ny*ny);
                if (nlen <= 0.0) { nx = 1.0; ny = 0.0; }
                else { nx /= nlen; ny /= nlen; }

                double c1x=0, c1y=0, c2x=0, c2y=0, w1=0, w2=0;

                // compute two sub-centers
                for (int64_t y = 0; y < H; ++y)
                {
                    for (int64_t x = 0; x < W; ++x)
                    {
                        if (lacc[y][x] != l) continue;

                        float dg = gacc[y][x];
                        float dx = float(x) - cx;
                        float dy = float(y) - cy;
                        float proj = dx*nx + dy*ny;
                        float len = std::sqrt(dx*dx + dy*dy) + eps;
                        float w = dg / len;

                        if (proj >= 0) { c1x += w*x; c1y += w*y; w1+=w; }
                        else { c2x += w*x; c2y += w*y; w2+=w; }
                    }
                }

                if (w1 > 0) { c1x /= w1; c1y /= w1; } else { c1x = cx+nx*0.5; c1y = cy+ny*0.5; }
                if (w2 > 0) { c2x /= w2; c2y /= w2; } else { c2x = cx-nx*0.5; c2y = cy-ny*0.5; }

                next_centers.push_back({float(std::clamp(c1x, 0.0, double(W-1))), float(std::clamp(c1y, 0.0, double(H-1)))});
                next_centers.push_back({float(std::clamp(c2x, 0.0, double(W-1))), float(std::clamp(c2y, 0.0, double(H-1)))});
            }

            // limit centers
            if ((int)next_centers.size() > target_superpixels)
                next_centers.resize(target_superpixels);

            centers = std::move(next_centers);
        }

        return CenterUpdateResult{centers, did_any_split};
    }

    torch::Tensor ComputeSuperpixelLabels(const torch::Tensor& density_hw, int num_superpixels) {
        TORCH_CHECK(density_hw.dim() == 2, "Expected [H, W] density");

        auto dens = GenerateSpeedField(density_hw);
        const int64_t H = dens.size(0);
        const int64_t W = dens.size(1);

        const int max_outer_iters = 20;
        const int min_outer_iters = 3;
        const float phi = 0.5f;
        const float Ts = 2.0f;
        const float Tc = 4.0f;
        const int max_splits_per_iter = 2;
        const float eps = 1e-6f;

        const int N = std::max(1, num_superpixels);

        std::vector<Center> centers = PlaceInitialSeeds(H, W, N);
        torch::Tensor labels;
        torch::Tensor geodesic;
        for (int iter = 0; iter < max_outer_iters; ++iter) {
            std::tie(labels, geodesic) = EvolveBoundaries(dens, centers);

            auto update = RelocateAndSplitCenters(labels, geodesic, dens, centers, N, max_splits_per_iter, phi, Ts, Tc, eps, 7);
            centers = std::move(update.centers);

            if (!update.split_performed && iter + 1 >= min_outer_iters && static_cast<int>(centers.size()) >= N) {
                break;
            }
        }


        return labels; // [H, W], long
    }

    std::tuple<torch::Tensor, torch::Tensor> BuildSuperpixelClassTargets(const torch::Tensor& y_chw, const torch::Tensor& labels_hw, int num_sp) {
        TORCH_CHECK(y_chw.dim() == 3, "y_chw must be [C,H,W]");
        TORCH_CHECK(labels_hw.dim() == 2, "labels_hw must be [H,W]");

        auto y= y_chw.to(torch::kFloat32).to(torch::kCPU).contiguous();
        auto labels = labels_hw.to(torch::kLong).to(torch::kCPU).contiguous();

        const int64_t C = y.size(0);
        const int64_t H = y.size(1);
        const int64_t W = y.size(2);

        int64_t max_label = labels.max().item<int64_t>();
        int64_t K = max_label + 1;
        if (K > num_sp)
            K = num_sp;

        auto counts = torch::zeros({K, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto cnt_acc= counts.accessor<float, 2>();
        auto yacc= y.accessor<float, 3>();
        auto lacc = labels.accessor<int64_t, 2>();

        for (int64_t ypix = 0; ypix < H; ++ypix) {
            for (int64_t xpix = 0; xpix < W; ++xpix) {
                int64_t l = lacc[ypix][xpix];
                if (l < 0 || l >= K) {
                    continue;
                }
                for (int64_t c = 0; c < C; ++c) {
                    cnt_acc[l][c] += yacc[c][ypix][xpix];
                }
            }
        }

        auto sum_per_sp  = counts.sum(1, true);
        auto norm_counts = counts / (sum_per_sp + 1e-6f);
        auto labels_sp   = std::get<1>(norm_counts.max(1));

        return {labels_sp, norm_counts};
    }

    torch::Tensor BuildSuperpixelFeatureVectors(const torch::Tensor& img_chw, const torch::Tensor& labels_hw) {
        TORCH_CHECK(img_chw.dim() == 3 && img_chw.size(0) == 3,"BuildSuperpixelFeatureVectors expects [3,H,W]");
        TORCH_CHECK(labels_hw.dim() == 2, "BuildSuperpixelFeatureVectors expects labels [H,W]");

        auto img    = img_chw.to(torch::kFloat32).to(torch::kCPU).contiguous();
        auto labels = labels_hw.to(torch::kLong).to(torch::kCPU).contiguous();

        const int64_t C = img.size(0);  // 3
        const int64_t H = img.size(1);
        const int64_t W = img.size(2);

        // Determine number of labels K
        int64_t max_label = labels.max().item<int64_t>();
        int64_t K = max_label + 1;

        constexpr int kNumMetrics = 5;
        const int64_t F = kNumMetrics * C; // 15

        // accumulators per (label, channel)
        std::vector<double> min_v(K * C,  std::numeric_limits<double>::infinity());
        std::vector<double> max_v(K * C, -std::numeric_limits<double>::infinity());
        std::vector<double> sum_v(K * C,  0.0);
        std::vector<double> sum2_v(K * C, 0.0);
        std::vector<double> sum3_v(K * C, 0.0);
        std::vector<int64_t> cnt_v(K * C, 0);

        auto lacc = labels.accessor<int64_t, 2>();
        auto iacc = img.accessor<float, 3>(); // [C,H,W]

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                int64_t l = lacc[y][x];
                if (l < 0 || l >= K) {
                    continue;
                }
                for (int64_t c = 0; c < C; ++c) {
                    double v = static_cast<double>(iacc[c][y][x]);
                    std::size_t idx = static_cast<std::size_t>(l * C + c);

                    min_v[idx] = std::min(min_v[idx], v);
                    max_v[idx] = std::max(max_v[idx], v);
                    sum_v[idx]  += v;
                    sum2_v[idx] += v * v;
                    sum3_v[idx] += v * v * v;
                    cnt_v[idx]  += 1;
                }
            }
        }

        auto out = torch::empty({K, F}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto oacc = out.accessor<float, 2>();

        for (int64_t l = 0; l < K; ++l) {
            for (int64_t c = 0; c < C; ++c) {
                std::size_t idx = static_cast<std::size_t>(l * C + c);
                double cnt = static_cast<double>(cnt_v[idx]);

                double mean   = (cnt > 0) ? sum_v[idx]  / cnt : 0.0;
                double mean2  = (cnt > 0) ? sum2_v[idx] / cnt : 0.0;
                double mean3  = (cnt > 0) ? sum3_v[idx] / cnt : 0.0;
                double var    = std::max(0.0, mean2 - mean * mean);
                double stddev = std::sqrt(var);
                double skew   = 0.0;
                if (stddev > 1e-6) {
                    double m3_central = mean3 - 3.0 * mean * mean2 + 2.0 * mean * mean * mean;
                    skew = m3_central / (stddev * stddev * stddev + 1e-6);
                }

                // Pack as [min, max, mean, std, skew] for this channel
                oacc[l][0 * C + c] = static_cast<float>(min_v[idx]);
                oacc[l][1 * C + c] = static_cast<float>(max_v[idx]);
                oacc[l][2 * C + c] = static_cast<float>(mean);
                oacc[l][3 * C + c] = static_cast<float>(stddev);
                oacc[l][4 * C + c] = static_cast<float>(skew);
            }
        }

        return out; // [K, 15]
    }


    torch::Tensor BuildSuperpixelOverlay(const torch::Tensor& img_chw, int num_superpixels) {
        TORCH_CHECK(img_chw.dim() == 3, "Expected [C, H, W] image");
        TORCH_CHECK(img_chw.size(0) == 3, "Expected RGB image [3, H, W]");

        // Work on CPU float
        auto img = img_chw.detach().to(torch::kFloat32).to(torch::kCPU).contiguous();
        const int64_t C = img.size(0);
        const int64_t H = img.size(1);
        const int64_t W = img.size(2);

        // Compute density + superpixels
        auto density = ComputeDensity(img);
        auto labels  = ComputeSuperpixelLabels(density, num_superpixels);
        auto centers = PlaceInitialSeeds(H, W, num_superpixels);
        auto lacc = labels.accessor<int64_t, 2>();

        auto overlay = img.clone();
        double maxval = overlay.max().item<double>();
        if (maxval > 1.5) {
            // Assume 0–255
            overlay.div_(255.0f);
        }
        auto oacc = overlay.accessor<float, 3>();

        // Build boundary mask: pixel is boundary if any 4-neighbor has different label
        std::vector<uint8_t> boundary(static_cast<std::size_t>(H * W), 0);
        const int dx[4] = {1, -1, 0, 0};
        const int dy[4] = {0, 0, 1, -1};

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                int64_t k = lacc[y][x];
                bool is_boundary = false;
                for (int dir = 0; dir < 4; ++dir) {
                    int64_t xx = x + dx[dir];
                    int64_t yy = y + dy[dir];
                    if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
                    if (lacc[yy][xx] != k) {
                        is_boundary = true;
                        break;
                    }
                }
                if (is_boundary) {
                    boundary[static_cast<std::size_t>(y * W + x)] = 1;
                }
            }
        }

        // Paint boundaries in red on top of the image
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                if (!boundary[static_cast<std::size_t>(y * W + x)]) continue;
                oacc[0][y][x] = 1.0f; // R
                oacc[1][y][x] = 0.0f; // G
                oacc[2][y][x] = 0.0f; // B
            }
        }
        // Plot the initial Voronoi seeds as a small cross on top of the image
        auto mark_cross = [&](int cx, int cy) {
            constexpr int kHalf = 2;
            for (int dx = -kHalf; dx <= kHalf; ++dx) {
                int xx = cx + dx;
                if (xx < 0 || xx >= W || cy < 0 || cy >= H) continue;
                oacc[0][cy][xx] = 1.0f; // R
                oacc[1][cy][xx] = 0.0f; // G
                oacc[2][cy][xx] = 0.0f; // B
            }
            for (int dy = -kHalf; dy <= kHalf; ++dy) {
                int yy = cy + dy;
                if (yy < 0 || yy >= H || cx < 0 || cx >= W) continue;
                oacc[0][yy][cx] = 1.0f; // R
                oacc[1][yy][cx] = 0.0f; // G
                oacc[2][yy][cx] = 0.0f; // B
            }
        };

        for (const auto& c : centers) {
            int cx = static_cast<int>(std::lround(c.x));
            int cy = static_cast<int>(std::lround(c.y));
            if (cx < 0 || cx >= static_cast<int>(W) || cy < 0 || cy >= static_cast<int>(H)) {
                continue;
            }
            mark_cross(cx, cy);
        }
        return overlay; // [3, H, W], float32 in [0,1]
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
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    // std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});

    constexpr int kNumSuperpixels = 8;

    auto input_overlay = BuildSuperpixelOverlay(X.index({0}), kNumSuperpixels);
    Nott::Plot::Data::Image(input_overlay.unsqueeze(0), {0}, Nott::Plot::Data::ImagePlotOptions{.layoutTitle = "Input with superpixel segmentation", .showColorBox = true});

    {
        auto img0     = X.index({0});
        auto density  = ComputeDensity(img0);
        auto centers0 = PlaceInitialSeeds(density.size(0), density.size(1), kNumSuperpixels);

        torch::Tensor labels0, geodesic0;
        std::tie(labels0, geodesic0) = RunGeodesicVoronoi(density, centers0);

        auto d_min = geodesic0.min().item<float>();
        auto d_max = geodesic0.max().item<float>();
        std::cout << "geodesic min/max = " << d_min << " / " << d_max << "\n";

        auto d_norm = (geodesic0 - d_min) / (d_max - d_min + 1e-6f);
        auto d_rgb  = d_norm.unsqueeze(0).repeat({3, 1, 1});

        Nott::Plot::Data::Image(
            d_rgb.unsqueeze(0), {0},
            Nott::Plot::Data::ImagePlotOptions{
                .layoutTitle  = "Geodesic distance map",
                .showColorBox = true
            });
    }

    std::vector<Sample> samples;
    const int64_t size = X.size(0);
    samples.reserve(static_cast<std::size_t>(size));

    auto Y_onehot = ConvertRgbMasksToOneHot(Y);

    for (int64_t b = 0; b < size; ++b) {
        // std::cout << "b[" << b << "]"<< std::endl;
        auto img_b = X.index({b});
        auto y_b   = Y_onehot.index({b});

        auto density_b = ComputeDensity(img_b);
        auto labels_hw = ComputeSuperpixelLabels(density_b, kNumSuperpixels);

        auto sp_features = BuildSuperpixelFeatureVectors(img_b, labels_hw);

        auto [labels_sp, norm_counts] = BuildSuperpixelClassTargets(y_b, labels_hw, kNumSuperpixels);
        Sample record;
        record.superpixel.input = sp_features;
        record.superpixel.target = labels_sp;
        record.NormTarget = norm_counts;

        Sample::Metrics<0, 0, 0> metrics;
        metrics.min  = sp_features.min();
        metrics.max  = sp_features.max();
        metrics.avg  = sp_features.mean();
        metrics.std  = sp_features.std(false);
        metrics.skew = torch::zeros_like(metrics.avg); // placeholder
        record.metric = metrics;

        samples.push_back(std::move(record));
    }


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
    auto out = struct2train(samples);
    model.train(out[0], out[1],
        {.epoch = E,
        .batch_size = B,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{X, Y},
        .graph_mode = Nott::GraphMode::Capture,
        .enable_amp = true,
        // .memory_format = torch::MemoryFormat::ChannelsLast
        });

    model.evaluate(out[0], out[1], Nott::Evaluation::Segmentation,{
        Nott::Metric::Classification::Accuracy,
        Nott::Metric::Classification::Precision,
        Nott::Metric::Classification::Recall,
        Nott::Metric::Classification::JaccardIndexMicro,
        Nott::Metric::Classification::BoundaryIoU,
        Nott::Metric::Classification::HausdorffDistance,
    },{.batch_size = B, .buffer_vram=2});


    std::tie(X, Y) = Nott::Data::Manipulation::Fraction(out[0], out[1], 0.01f);
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

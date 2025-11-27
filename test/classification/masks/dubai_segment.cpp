#include "../../../include/Thot.h"
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
        struct Target {
            torch::Tensor min;
            torch::Tensor max;
            torch::Tensor avg;
            torch::Tensor std;
            torch::Tensor skew;
            torch::Tensor inputRepresentation;
            torch::Tensor targetRepresentation;
            torch::Tensor voronoiInput;
            torch::Tensor voronoiTarget;
        };

        SuperPixel superpixel;
        torch::Tensor inputRepresentation;
        torch::Tensor targetRepresentation;
        Target<0, 0, 0> target;
    };

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


    // multi-source Dijkstra on 4-connected grid
    std::tuple<torch::Tensor, torch::Tensor>
    RunGeodesicVoronoi(const torch::Tensor& dens, const std::vector<Center>& centers) {
        TORCH_CHECK(dens.dim() == 2, "Expected [H, W] density");
        auto dens_cpu = dens.to(torch::kFloat32).to(torch::kCPU).contiguous();

        const int64_t H = dens_cpu.size(0);
        const int64_t W = dens_cpu.size(1);
        const int64_t M = H * W;

        auto dacc = dens_cpu.accessor<float, 2>();

        std::vector<float> dist(static_cast<std::size_t>(M),
                                std::numeric_limits<float>::infinity());
        std::vector<int>   label(static_cast<std::size_t>(M), -1);
        std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;

        for (int s = 0; s < static_cast<int>(centers.size()); ++s) {
            int sx = static_cast<int>(std::lround(centers[s].x));
            int sy = static_cast<int>(std::lround(centers[s].y));
            sx = std::max(0, std::min(static_cast<int>(W - 1), sx));
            sy = std::max(0, std::min(static_cast<int>(H - 1), sy));
            int idx = sy * static_cast<int>(W) + sx;
            if (idx < 0 || idx >= static_cast<int>(M)) continue;
            std::size_t midx = static_cast<std::size_t>(idx);
            if (dist[midx] <= 0.0f) {
                continue;
            }
            dist[midx]  = 0.0f;
            label[midx] = s;
            pq.push(PQNode{0.0f, idx});
        }

        const int dx4[4] = {1, -1, 0, 0};
        const int dy4[4] = {0, 0, 1, -1};

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();
            std::size_t uidx = static_cast<std::size_t>(u);
            if (d > dist[uidx]) continue;

            int uy = u / static_cast<int>(W);
            int ux = u % static_cast<int>(W);
            int lu = label[uidx];
            if (lu < 0) continue;

            float d_u = dacc[uy][ux];

            for (int k = 0; k < 4; ++k) {
                int vx = ux + dx4[k];
                int vy = uy + dy4[k];
                if (vx < 0 || vx >= static_cast<int>(W) ||
                    vy < 0 || vy >= static_cast<int>(H)) {
                    continue;
                    }
                int v = vy * static_cast<int>(W) + vx;
                std::size_t vidx = static_cast<std::size_t>(v);

                float d_v   = dacc[vy][vx];
                // ∫ D(x) ds ≈ 0.5 * (D_u + D_v)
                float cost  = 0.5f * (d_u + d_v);
                float ndist = d + cost;

                if (ndist < dist[vidx]) {
                    dist[vidx]  = ndist;
                    label[vidx] = lu;
                    pq.push(PQNode{ndist, v});
                }
            }
        }

        auto labels = torch::empty({H, W},
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
        auto lacc = labels.accessor<int64_t, 2>();

        auto geodesic = torch::empty({H, W},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto gacc = geodesic.accessor<float, 2>();

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                int idx = static_cast<int>(y * W + x);
                std::size_t midx = static_cast<std::size_t>(idx);
                lacc[y][x]  = static_cast<int64_t>(label[midx]);
                gacc[y][x]  = dist[midx];
            }
        }

        return std::make_tuple(labels, geodesic);
    }

    // ==========================================
    // Structure-sensitive density D(x)
    // - grayscale
    // - gradient magnitude
    // - local average of gradient (box filter)
    // - E(x) = grad / (avg_grad + gamma)
    // - D(x) = exp(E / nu)
    // ==========================================
    torch::Tensor ComputeDensity(const torch::Tensor& img_chw) {
        TORCH_CHECK(img_chw.dim() == 3, "Expected [C, H, W]");
        TORCH_CHECK(img_chw.size(0) == 3, "Expected RGB image as [3, H, W]");

        auto img = img_chw.to(torch::kFloat32).to(torch::kCPU).contiguous();
        const auto C = img.size(0);
        const auto H = img.size(1);
        const auto W = img.size(2);

        // grayscale buffer
        auto gray = torch::zeros({H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto acc  = img.accessor<float, 3>();
        auto gacc = gray.accessor<float, 2>();

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                // assume loader gives 0–255; if it’s already 0–1, this just rescales down, still fine
                float r = acc[0][y][x] / 255.0f;
                float g = acc[1][y][x] / 255.0f;
                float b = acc[2][y][x] / 255.0f;
                gacc[y][x] = 0.2989f * r + 0.5870f * g + 0.1140f * b;
            }
        }


        // Gradient magnitude via central differences
        auto grad = torch::zeros_like(gray);
        auto gradacc = grad.accessor<float, 2>();
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                int64_t xm1 = std::max<int64_t>(0, x - 1);
                int64_t xp1 = std::min<int64_t>(W - 1, x + 1);
                int64_t ym1 = std::max<int64_t>(0, y - 1);
                int64_t yp1 = std::min<int64_t>(H - 1, y + 1);

                float gx = gacc[y][xp1] - gacc[y][xm1];
                float gy = gacc[yp1][x] - gacc[ym1][x];
                gradacc[y][x] = std::sqrt(gx * gx + gy * gy);
            }
        }

        // Local average of gradient magnitude: simple box filter radius=2
        const int radius = 2;
        auto grad_blur = torch::zeros_like(gray);
        auto bacc = grad_blur.accessor<float, 2>();
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                double sum = 0.0;
                int count = 0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int64_t yy = y + dy;
                    if (yy < 0 || yy >= H) continue;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int64_t xx = x + dx;
                        if (xx < 0 || xx >= W) continue;
                        sum += gradacc[yy][xx];
                        ++count;
                    }
                }
                bacc[y][x] = static_cast<float>(sum / std::max(count, 1));
            }
        }

        // E(x) = grad / (blurred_grad + gamma)
        // D(x) = exp(E / nu)
        constexpr float gamma = 0.12f;
        constexpr float nu    = 1.0f;
        auto density = torch::zeros_like(gray);
        auto dacc = density.accessor<float, 2>();

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float num = gradacc[y][x];
                float den = bacc[y][x] + gamma;
                float E   = (den > 0.0f) ? (num / den) : 0.0f;
                if (E < 0.0f) E = 0.0f;
                // clamp exponent so exp() doesn't blow up
                float E_clamped = std::min(E / nu, 10.0f); // exp(10) ~ 2e4
                dacc[y][x] = std::exp(E_clamped);

            }
        }

        return density; // [H, W], float32, CPU
    }

    // ==========================================
    // Geodesic-based superpixels via voronoi (Dijkstra compute)
    // - density D(x) as cost per pixel
    // - multi-source shortest paths on 4-neighbour grid for Geodesic
    // - w(p,q)= exp[a*||I(p)-I(q)||²]
    // ==========================================
    std::vector<Center> PlaceInitialSeeds(int64_t H, int64_t W, int num_superpixels) {
        const int N = std::max(1, num_superpixels);
        int K = std::max(2, N / 4);

        std::vector<Center> centers;
        centers.reserve(static_cast<std::size_t>(N * 2));

        double total_pixels = static_cast<double>(H * W);
        double spacing = std::sqrt(total_pixels / static_cast<double>(K));

        int grid_x = std::max(1,
            static_cast<int>(std::round(static_cast<double>(W) / spacing)));
        int grid_y = std::max(1,
            static_cast<int>(std::round(static_cast<double>(H) / spacing)));

        double step_x = static_cast<double>(W) / static_cast<double>(grid_x);
        double step_y = static_cast<double>(H) / static_cast<double>(grid_y);

        for (int gy = 0; gy < grid_y && static_cast<int>(centers.size()) < K; ++gy) {
            for (int gx = 0; gx < grid_x && static_cast<int>(centers.size()) < K; ++gx) {
                float sx = static_cast<float>((gx + 0.5) * step_x);
                float sy = static_cast<float>((gy + 0.5) * step_y);
                sx = std::max(0.0f, std::min(static_cast<float>(W - 1), sx));
                sy = std::max(0.0f, std::min(static_cast<float>(H - 1), sy));
                centers.push_back(Center{sx, sy});
            }
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

            // keep all non-split centers
            for (int l = 0; l < S; ++l)
                if (!marked[l]) next_centers.push_back(relocated[l]);

            // split centers
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
                        float len  = std::sqrt(dx*dx + dy*dy) + eps;
                        float w    = dg / len;

                        if (proj >= 0) { c1x += w*x; c1y += w*y; w1+=w; }
                        else           { c2x += w*x; c2y += w*y; w2+=w; }
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

        const int   max_outer_iters      = 20;
        const int   min_outer_iters      = 3;
        const float phi                  = 0.5f;
        const float Ts                   = 2.0f;
        const float Tc                   = 4.0f;
        const int   max_splits_per_iter  = 10;
        const float eps                  = 1e-6f;

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

    torch::Tensor BuildRegionStatisticMapsForImage(const torch::Tensor& img_chw,
                                                   const torch::Tensor& labels_hw) {
        TORCH_CHECK(img_chw.dim() == 3, "Expected [C, H, W] image");
        TORCH_CHECK(labels_hw.dim() == 2, "Expected [H, W] labels");
        TORCH_CHECK(img_chw.size(1) == labels_hw.size(0) &&
                    img_chw.size(2) == labels_hw.size(1),
                    "Image and labels spatial dimensions must match");

        auto img    = img_chw.to(torch::kFloat32).to(torch::kCPU).contiguous();
        auto labels = labels_hw.to(torch::kLong).to(torch::kCPU).contiguous();

        const int64_t C = img.size(0);
        const int64_t H = img.size(1);
        const int64_t W = img.size(2);

        auto max_label = labels.max().item<int64_t>();
        const int64_t K = std::max<int64_t>(0, max_label + 1);

        std::vector<double> min_v(static_cast<std::size_t>(K * C), std::numeric_limits<double>::infinity());
        std::vector<double> max_v(static_cast<std::size_t>(K * C), -std::numeric_limits<double>::infinity());
        std::vector<double> sum(static_cast<std::size_t>(K * C), 0.0);
        std::vector<double> sumsq(static_cast<std::size_t>(K * C), 0.0);
        std::vector<double> sumcube(static_cast<std::size_t>(K * C), 0.0);
        std::vector<int64_t> count(static_cast<std::size_t>(K * C), 0);

        auto lacc = labels.accessor<int64_t, 2>();
        auto iacc = img.accessor<float, 3>();

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                int64_t l = lacc[y][x];
                if (l < 0 || l >= K) continue;
                std::size_t base = static_cast<std::size_t>(l * C);
                for (int64_t c = 0; c < C; ++c) {
                    double v = static_cast<double>(iacc[c][y][x]);
                    std::size_t idx = base + static_cast<std::size_t>(c);
                    min_v[idx] = std::min(min_v[idx], v);
                    max_v[idx] = std::max(max_v[idx], v);
                    sum[idx] += v;
                    sumsq[idx] += v * v;
                    sumcube[idx] += v * v * v;
                    count[idx] += 1;
                }
            }
        }

        std::vector<float> mean(static_cast<std::size_t>(K * C), 0.0f);
        std::vector<float> stddev(static_cast<std::size_t>(K * C), 0.0f);
        std::vector<float> skew(static_cast<std::size_t>(K * C), 0.0f);

        for (int64_t l = 0; l < K; ++l) {
            for (int64_t c = 0; c < C; ++c) {
                std::size_t idx = static_cast<std::size_t>(l * C + c);
                if (count[idx] <= 0) {
                    min_v[idx] = 0.0;
                    max_v[idx] = 0.0;
                    continue;
                }

                double cnt   = static_cast<double>(count[idx]);
                double mu    = sum[idx] / cnt;
                double ex2   = sumsq[idx] / cnt;
                double ex3   = sumcube[idx] / cnt;
                double var   = std::max(0.0, ex2 - mu * mu);
                double sigma = std::sqrt(var);
                double c3    = ex3 - 3.0 * mu * ex2 + 2.0 * mu * mu * mu; // E[(x - mu)^3]

                mean[idx]   = static_cast<float>(mu);
                stddev[idx] = static_cast<float>(sigma);
                skew[idx]   = (sigma > 0.0)
                    ? static_cast<float>(c3 / (sigma * sigma * sigma))
                    : 0.0f;
            }
        }

        constexpr int kNumMetrics = 5; // min, max, mean, std, skew
        auto out = torch::empty({kNumMetrics * C, H, W},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto oacc = out.accessor<float, 3>();

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                int64_t l = lacc[y][x];
                if (l < 0 || l >= K) {
                    for (int64_t c = 0; c < C; ++c) {
                        for (int m = 0; m < kNumMetrics; ++m) {
                            oacc[m * C + c][y][x] = 0.0f;
                        }
                    }
                    continue;
                }

                std::size_t base = static_cast<std::size_t>(l * C);
                for (int64_t c = 0; c < C; ++c) {
                    std::size_t idx = base + static_cast<std::size_t>(c);
                    oacc[0 * C + c][y][x] = static_cast<float>(min_v[idx]);
                    oacc[1 * C + c][y][x] = static_cast<float>(max_v[idx]);
                    oacc[2 * C + c][y][x] = mean[idx];
                    oacc[3 * C + c][y][x] = stddev[idx];
                    oacc[4 * C + c][y][x] = skew[idx];
                }
            }
        }

        return out; // [5*C, H, W]
    }

    torch::Tensor BuildSuperpixelFeatureBatch(const torch::Tensor& batch_bchw, int num_superpixels) {
        TORCH_CHECK(batch_bchw.dim() == 4, "Expected [B, C, H, W]");
        TORCH_CHECK(batch_bchw.size(1) == 3, "Expected RGB images with 3 channels");

        auto batch = batch_bchw.to(torch::kFloat32).to(torch::kCPU).contiguous();
        const int64_t B = batch.size(0);
        const int64_t C = batch.size(1);
        const int64_t H = batch.size(2);
        const int64_t W = batch.size(3);

        constexpr int kNumMetrics = 5;
        auto out = torch::empty({B, kNumMetrics * C, H, W},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        for (int64_t b = 0; b < B; ++b) {
            auto img = batch.index({b}); // [3, H, W]
            auto density = ComputeDensity(img);              // [H, W]
            auto labels  = ComputeSuperpixelLabels(density, num_superpixels); // [H, W]
            auto feats = BuildRegionStatisticMapsForImage(img, labels);     // [5*C, H, W]
            out.index_put_({b}, feats);
        }

        return out;
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
                oacc[1][cy][xx] = 1.0f; // G
                oacc[2][cy][xx] = 0.0f; // B
            }
            for (int dy = -kHalf; dy <= kHalf; ++dy) {
                int yy = cy + dy;
                if (yy < 0 || yy >= H || cx < 0 || cx >= W) continue;
                oacc[0][yy][cx] = 1.0f; // R
                oacc[1][yy][cx] = 1.0f; // G
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
}


int main() {
    Thot::Model model("");

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::Universal("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1",
            Thot::Data::Type::JPG{"images", {.grayscale = false, .normalize_colors = false, .normalize_size=true, .color_order = "RGB"}},
            Thot::Data::Type::PNG{"masks", {.normalize_colors = false, .normalize_size=true, .InterpolationMode = Thot::Data::Transform::Format::Options::InterpMode::Nearest, .color_order = "RGB"}
            },{.train_fraction = 1.f, .test_fraction = 0.f, .shuffle = false});

    /*
    auto x1_raw = x1.clone();
    auto y1_raw = y1.clone();
    y1 = ConvertRgbMasksToOneHot(y1);
    y2 = ConvertRgbMasksToOneHot(y2);

    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"x"}, .frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"y"}, .frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, true, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});

    constexpr int kNumSuperpixels = 16;


    Thot::Data::Check::Size(x1, "Inputs Preprocessed");
    Thot::Data::Check::Size(y1, "Train Targets One-hot");

    std::vector<Sample> samples;
    Sample::SuperPixel superpixel{x1, y1};
    Sample::Target<0, 0, 0> target_stats;
    target_stats.min = y1.min();
    target_stats.max = y1.max();
    target_stats.avg = y1.mean();
    target_stats.std = y1.std();
    target_stats.skew = torch::zeros_like(target_stats.avg);
    auto input_overlay = BuildSuperpixelOverlay(x1_raw.index({0}), kNumSuperpixels);
    auto target_overlay = BuildSuperpixelOverlay(y1_raw.index({0}), kNumSuperpixels);
    target_stats.inputRepresentation = input_overlay;
    target_stats.targetRepresentation = target_overlay;
    target_stats.voronoiInput = target_stats.inputRepresentation;
    target_stats.voronoiTarget = target_stats.targetRepresentation;

    Thot::Plot::Data::Image(input_overlay.unsqueeze(0), {0}, Thot::Plot::Data::ImagePlotOptions{.layoutTitle = "Input with superpixel segmentation", .showColorBox = true});
    Thot::Plot::Data::Image(target_overlay.unsqueeze(0),{0}, Thot::Plot::Data::ImagePlotOptions{.layoutTitle = "Raw target with superpixel segmentation", .showColorBox = true});

    auto speed_field = GenerateSpeedField(ComputeDensity(x1_raw.index({0})));
    auto speed_min = speed_field.min().item<float>();
    auto speed_max = speed_field.max().item<float>();
    auto speed_norm = (speed_max > speed_min) ? (speed_field - speed_min) / (speed_max - speed_min) : torch::zeros_like(speed_field);
    auto speed_rgb = speed_norm.unsqueeze(0).repeat({3, 1, 1});

    Thot::Plot::Data::Image(speed_rgb.unsqueeze(0),{0}, Thot::Plot::Data::ImagePlotOptions{.layoutTitle = "Speed generation map", .showColorBox = true});

    Sample record;
    record.superpixel = superpixel;
    record.inputRepresentation = target_stats.inputRepresentation;
    record.targetRepresentation = target_stats.targetRepresentation;
    record.target = target_stats;
    samples.push_back(std::move(record));

    */
    Thot::Data::Check::Size(x1, "Raw Input");
    // Recompose Tile
    std::vector<torch::Tensor> rows_input; rows_input.reserve(3);
    std::vector<torch::Tensor> rows_label; rows_label.reserve(3);
    for (int i=0; i<3; ++i) {
        auto inp1 = x1[3*i+0]; auto lab1 = y1[3*i+0];
        auto inp2 = x1[3*i+1]; auto lab2 = y1[3*i+1];
        auto inp3 = x1[3*i+2]; auto lab3 = y1[3*i+2];

        torch::Tensor row_inp = torch::cat({inp1, inp2, inp3}, /*dim=*/2);
        torch::Tensor row_lab = torch::cat({lab1, lab2, lab3}, /*dim=*/2);
        rows_input.push_back(row_inp); rows_label.push_back(row_lab);
    }
    torch::Tensor inp = torch::cat(rows_input, /*dim=*/1); torch::Tensor lab = torch::cat(rows_label, /*dim=*/1);
    Thot::Data::Check::Size(inp, "Recomposed Tile 1 Input");
    Thot::Data::Check::Size(lab, "Recomposed Tile Mask");
    //Thot::Plot::Data::Image(inp.unsqueeze(0), {0});
    //Thot::Plot::Data::Image(lab.unsqueeze(0), {0});



    {
        {
            torch::Tensor t = inp.detach().to(torch::kCPU);
            t = t.clamp(0, 255).to(torch::kU8);
            torch::Tensor t_hwc = t.permute({1, 2, 0}).contiguous();
            int H = t_hwc.size(0);
            int W = t_hwc.size(1);
            cv::Mat img(H, W, CV_8UC3, t_hwc.data_ptr<uint8_t>());
            cv::Mat img_bgr;
            cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
            cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/input_raw.png", img_bgr);
            cv::GaussianBlur(img_bgr, img_bgr, cv::Size(), 0.83);
            cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/input_smooth.png", img_bgr);
        }

        {
            torch::Tensor t = lab.detach().to(torch::kCPU);
            t = t.clamp(0, 255).to(torch::kU8);
            torch::Tensor t_hwc = t.permute({1, 2, 0}).contiguous();
            int H = t_hwc.size(0);
            int W = t_hwc.size(1);
            cv::Mat img(H, W, CV_8UC3, t_hwc.data_ptr<uint8_t>());
            cv::Mat img_bgr;
            cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
            cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/lab_raw.png", img_bgr);
            cv::GaussianBlur(img_bgr, img_bgr, cv::Size(), 0.83);
            cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/lab_smooth.png", img_bgr);
        }

        {
            auto input = GenerateSpeedField(ComputeDensity(inp)); //.index({0})
            auto speed_min = input.min().item<float>();
            auto speed_max = input.max().item<float>();
            auto speed_norm = (speed_max > speed_min) ? (input - speed_min) / (speed_max - speed_min) : torch::zeros_like(input);
            auto speed_rgb = speed_norm.unsqueeze(0).repeat({3, 1, 1});
            Thot::Data::Check::Size(speed_rgb, "GeoDist");
            torch::Tensor t = speed_rgb.detach().to(torch::kCPU);
            t = (t * 255.0).clamp(0, 255).to(torch::kU8);
            torch::Tensor t_hwc = t.permute({1, 2, 0}).contiguous();
            int H = t_hwc.size(0);
            int W = t_hwc.size(1);
            cv::Mat img(H, W, CV_8UC3, t_hwc.data_ptr<uint8_t>());
            cv::Mat img_bgr;
            cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
            cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/GeoDistance_Raw.png", img_bgr);
        }
    }


    //return 0;
    auto block = [&](int in_c, int out_c) {
        return Thot::Block::Sequential({
            Thot::Layer::Conv2d(
                {in_c, out_c, {3,3}, {1,1}, {1,1}},
                Thot::Activation::GeLU,
                Thot::Initialization::HeUniform
            ),
            Thot::Layer::Conv2d(
                {out_c, out_c, {3,3}, {1,1}, {1,1}},
                Thot::Activation::GeLU,
                Thot::Initialization::HeUniform
            ),
        });
    };

    auto upblock = [&](int in_c, int out_c) {
        return Thot::Block::Sequential({
            Thot::Layer::Upsample({.scale = {2,2}, .mode  = Thot::UpsampleMode::Bilinear}),
            Thot::Layer::Conv2d(
                {in_c, out_c, {3,3}, {1,1}, {1,1}},
                Thot::Activation::GeLU,
                Thot::Initialization::HeUniform
            ),
        });
    };

    constexpr int kNumMetrics = 5;
    constexpr int kInputChannels = 3 * kNumMetrics; // min,max,mean,std,skew per channel

    // ENCODER
    model.add(block(/*kInputChannels*/3, 64), "enc1");
    model.add(Thot::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool1");

    model.add(block(64,  128), "enc2");
    model.add(Thot::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool2");

    model.add(block(128, 256), "enc3");
    model.add(Thot::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool3");

    model.add(block(256, 512), "enc4");
    model.add(Thot::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool4");

    model.add(block(512, 1024), "bottleneck");


    //DECODERS (up to 64x64)
    model.add(upblock(1024, 512), "up4");
    model.add(block(1024, 512), "dec4");
    model.add(upblock(512, 256), "up3");
    model.add(block(512, 256), "dec3");

    model.add(upblock(256, 128), "up2");
    model.add(block(256, 128),"dec2");
    model.add(upblock(128, 64), "up1");
    model.add(block(128, 64), "dec1");
    model.add(Thot::Layer::Conv2d({64, 6, {1, 1}, {1, 1}, {0, 0}}, Thot::Activation::Identity), "logits");

    model.links({
        // encoder path
        Thot::LinkSpec{Thot::Port::Input("@input"), Thot::Port::Module("enc1")},
        Thot::LinkSpec{Thot::Port::Module("enc1"),  Thot::Port::Module("pool1")},
        Thot::LinkSpec{Thot::Port::Module("pool1"), Thot::Port::Module("enc2")},
        Thot::LinkSpec{Thot::Port::Module("enc2"),  Thot::Port::Module("pool2")},
        Thot::LinkSpec{Thot::Port::Module("pool2"), Thot::Port::Module("enc3")},
        Thot::LinkSpec{Thot::Port::Module("enc3"),  Thot::Port::Module("pool3")},
        Thot::LinkSpec{Thot::Port::Module("pool3"), Thot::Port::Module("enc4")},
        Thot::LinkSpec{Thot::Port::Module("enc4"),  Thot::Port::Module("pool4")},
        Thot::LinkSpec{Thot::Port::Module("pool4"), Thot::Port::Module("bottleneck")},

        // decoder with skip
        Thot::LinkSpec{Thot::Port::Module("bottleneck"), Thot::Port::Module("up4")},
        Thot::LinkSpec{Thot::Port::Join({"up4", "enc4"}, Thot::MergePolicy::Stack), Thot::Port::Module("dec4")},
        Thot::LinkSpec{Thot::Port::Module("dec4"), Thot::Port::Module("up3")},
        Thot::LinkSpec{Thot::Port::Join({"up3", "enc3"}, Thot::MergePolicy::Stack), Thot::Port::Module("dec3")},

        Thot::LinkSpec{Thot::Port::Module("dec3"), Thot::Port::Module("up2") },
        Thot::LinkSpec{Thot::Port::Join({"up2", "enc2"}, Thot::MergePolicy::Stack), Thot::Port::Module("dec2")},
        Thot::LinkSpec{Thot::Port::Module("dec2"), Thot::Port::Module("up1") },
        Thot::LinkSpec{Thot::Port::Join({"up1", "enc1"}, Thot::MergePolicy::Stack), Thot::Port::Module("dec1")},


        // head
        Thot::LinkSpec{Thot::Port::Module("dec1"),   Thot::Port::Module("logits")}, // dec3
        Thot::LinkSpec{Thot::Port::Module("logits"), Thot::Port::Output("@output")},
    }, true);

    x1 = inp.unsqueeze(0);
    y1 = ConvertRgbMasksToOneHot(lab.unsqueeze(0));

    x1 = Thot::Data::Transform::Format::Downsample(x1, {.size={512, 512}});
    y1 = Thot::Data::Transform::Format::Downsample(y1, {.size={512, 512}});

    x1 = x1.to(torch::kFloat32) / 255.0f;

    const auto total_training_samples = x1.size(0);
    const auto B = 8;
    const auto steps_per_epoch = static_cast<std::size_t>((total_training_samples + B - 1) / B);
    const auto total_training_steps = std::max<std::size_t>(1, 25 * std::max<std::size_t>(steps_per_epoch, 1));

    model.set_loss(Thot::Loss::BCEWithLogits({}));
    model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate = 5e-5, .weight_decay = 1e-4}),
        Thot::LrScheduler::CosineAnnealing({
            .T_max = (total_training_steps),
            .eta_min = 1e-6,
            .warmup_steps = std::min<std::size_t>(steps_per_epoch * 5, total_training_steps / 5),
            .warmup_start_factor = 0.1,
        }));

    model.set_regularization({Thot::Regularization::SWAG({
        .coefficient = 5e-4,
        .variance_epsilon = 1e-6,
        .start_step = static_cast<std::size_t>(0.65 * static_cast<double>(total_training_steps)),
        .accumulation_stride = std::max<std::size_t>(1, steps_per_epoch),
        .max_snapshots = 20,
    })});

    model.use_cuda(torch::cuda::is_available());

    model.train(x1, y1,
        {.epoch = 500,
         .batch_size = B,
         .restore_best_state = true,
         //.test = std::vector<at::Tensor>{x2, y2},
         .graph_mode = Thot::GraphMode::Capture,
         .enable_amp = true});

    torch::NoGradGuard guard;
    Thot::Data::Check::Size(x1, "Test Inputs");
    Thot::Data::Check::Size(y1, "Test Targets");

    model.evaluate(x1, y1, Thot::Evaluation::Segmentation,{
            Thot::Metric::Classification::Accuracy,
            Thot::Metric::Classification::Precision,
            Thot::Metric::Classification::Recall,
            Thot::Metric::Classification::JaccardIndexMicro,
        },{.batch_size = 8, .buffer_vram=2});


    auto logits = model.forward(x1);
    auto predicted = logits.argmax(1).to(torch::kCPU);
    auto first_pred = predicted.index({0}).contiguous();

    const auto H = static_cast<int>(first_pred.size(0));
    const auto W = static_cast<int>(first_pred.size(1));

    torch::Tensor forecast_rgb = torch::zeros({H, W, 3}, torch::TensorOptions().dtype(torch::kUInt8));
    auto pacc = first_pred.accessor<long, 2>();
    auto racc = forecast_rgb.accessor<std::uint8_t, 3>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto cls = pacc[y][x];
            if (cls < 0 || cls >= static_cast<long>(kClassPalette.size()))
                cls = 0;
            const auto& rgb = kClassPalette[static_cast<std::size_t>(cls)];
            racc[y][x][0] = rgb[0];
            racc[y][x][1] = rgb[1];
            racc[y][x][2] = rgb[2];
        }
    }
    cv::Mat f_img(H, W, CV_8UC3, forecast_rgb.data_ptr<uint8_t>());
    cv::Mat f_bgr;
    cv::cvtColor(f_img, f_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/forecast.png", f_bgr);
    cv::GaussianBlur(f_bgr, f_bgr, cv::Size(), 0.83);
    cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/forecast_raw.png", f_bgr);

    return 0;
}

#ifndef THOT_WAVELET_HPP
#define THOT_WAVELET_HPP
#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <optional>

namespace thot::signal::wavelet {
    inline std::pair<at::Tensor, at::Tensor> haar_filters(const at::Device& dev, at::ScalarType dtype) {
        const double s = std::sqrt(0.5);
        auto lo = torch::tensor({s, s}, torch::TensorOptions().dtype(dtype).device(dev));
        auto hi = torch::tensor({-s, s}, torch::TensorOptions().dtype(dtype).device(dev));
        return {lo, hi};
    }

    // Build conv1d kernel [outC=1,inC=1,k] from 1D vector; flip because conv is cross-correlation by default.
    inline at::Tensor make_conv1d_weight(const at::Tensor& f) {
        auto k = f.numel();
        auto w = f.flip({0}).reshape({1,1,(long)k}).contiguous();
        return w;
    }

    // Pad (reflect) to emulate “same” conv then stride=2 downsample.
    inline at::Tensor reflect_pad_1d(const at::Tensor& x, int padL, int padR) {
        namespace F = torch::nn::functional;
        return torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({padL, padR}).mode(torch::kReflect));
    }

    // x: [N] or [B,N]; returns (approx, detail) both downsampled by 2
    inline std::pair<at::Tensor, at::Tensor> dwt_once(const at::Tensor& x, const at::Tensor& lo, const at::Tensor& hi) {
        TORCH_CHECK(x.dim()==1 || (x.dim()==2 && x.size(0)>0),
                    "x must be [N] or [B,N]");
        const bool batched = (x.dim()==2);

        // [B,1,N]
        at::Tensor x3 = batched ? x.unsqueeze(1) : x.unsqueeze(0).unsqueeze(0);

        const int k = static_cast<int>(lo.numel());
        const int padL = (k - 1) / 2;
        const int padR = k / 2;

        x3 = reflect_pad_1d(x3, padL, padR);

        auto w_lo = make_conv1d_weight(lo).to(x.device(), x.scalar_type());
        auto w_hi = make_conv1d_weight(hi).to(x.device(), x.scalar_type());

        namespace F = torch::nn::functional;

        // No ambiguity + no ArrayRef dance.
        std::optional<at::Tensor> nobias = std::nullopt;
        c10::SmallVector<int64_t,1> stride_v{2}, pad_v{0}, dil_v{1};
        auto y_lo = at::conv1d(x3, w_lo, nobias, at::IntArrayRef(stride_v), at::IntArrayRef(pad_v), at::IntArrayRef(dil_v), 1);
        auto y_hi = at::conv1d(x3, w_hi, nobias, at::IntArrayRef(stride_v), at::IntArrayRef(pad_v), at::IntArrayRef(dil_v), 1);


        y_lo = y_lo.squeeze(1);
        y_hi = y_hi.squeeze(1);
        if (!batched) { y_lo = y_lo.squeeze(0); y_hi = y_hi.squeeze(0); }

        return std::pair<at::Tensor, at::Tensor>(y_lo, y_hi);
    }

    // Multi-level DWT. Returns final approx and all detail bands D1..DL (D1 = finest).
    struct DWTResult {
        at::Tensor final_approx;              // A_L
        std::vector<at::Tensor> details;      // {D1, D2, ..., DL}
        std::vector<at::Tensor> approximations; // {A1, A2, ..., AL}
    };

    inline DWTResult dwt(const at::Tensor& x, int levels) {
        TORCH_CHECK(levels >= 1, "levels must be >=1");
        auto dev = x.device();
        auto dtype = x.scalar_type();
        auto [lo, hi] = haar_filters(dev, dtype);

        DWTResult out;
        out.details.reserve(levels);
        out.approximations.reserve(levels);

        at::Tensor a = x;
        for (int l=0; l<levels; ++l) {
            auto [a_next, d] = dwt_once(a, lo, hi);
            out.details.push_back(d);
            out.approximations.push_back(a_next);
            a = a_next;
        }
        out.final_approx = a;
        return out;
    }

    // Optional: naive “upsample-and-hold” for plotting each level back to original length.
    // Repeats each sample 2^level times, for visual alignment only.
    inline at::Tensor upsample_for_plot_1d(const at::Tensor& y, int factor) {
        TORCH_CHECK(y.dim()==1, "upsample_for_plot_1d expects [M]");
        if (factor <= 1) return y.clone();
        // repeat_interleave along last dim
        return y.repeat_interleave(factor);
    }

    // Dump TSV with columns having potentially different lengths by writing separate files per series.
    // Simpler and avoids awkward resizing. Returns list of filenames written.
    inline std::vector<std::string>
    dump_for_gnuplot(const std::string& stem, const at::Tensor& x, const DWTResult& r) {
        std::vector<std::string> files;

        // Original
        {
            std::string path = stem + "_x.tsv";
            std::ofstream ofs(path);
            TORCH_CHECK(ofs.good(), "Failed to open ", path);
            auto h = x.detach().to(torch::kCPU).contiguous();
            const int64_t N = h.numel();
            for (int64_t i=0;i<N;++i) ofs << i << '\t' << h[i].item<double>() << '\n';
            files.push_back(path);
        }

        // Approximations A1..AL
        for (size_t l=0; l<r.approximations.size(); ++l) {
            std::string path = stem + "_A" + std::to_string(l+1) + ".tsv";
            std::ofstream ofs(path);
            TORCH_CHECK(ofs.good(), "Failed to open ", path);
            auto a = r.approximations[l].detach().to(torch::kCPU).contiguous();
            const int64_t M = a.numel();
            const int64_t stride = 1LL << (l+1); // relative to original spacing
            for (int64_t i=0;i<M;++i) ofs << (i*stride) << '\t' << a[i].item<double>() << '\n';
            files.push_back(path);
        }

        // Details D1..DL
        for (size_t l=0; l<r.details.size(); ++l) {
            std::string path = stem + "_D" + std::to_string(l+1) + ".tsv";
            std::ofstream ofs(path);
            TORCH_CHECK(ofs.good(), "Failed to open ", path);
            auto d = r.details[l].detach().to(torch::kCPU).contiguous();
            const int64_t M = d.numel();
            const int64_t stride = 1LL << (l+1);
            for (int64_t i=0;i<M;++i) ofs << (i*stride) << '\t' << d[i].item<double>() << '\n';
            files.push_back(path);
        }
        return files;
    }
}
#endif //THOT_WAVELET_HPP
// bk_cuda.cu
#include "bk_cuda.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

// ============================ Конфигурация ============================
// Верхняя граница числа вершин для GPU-версии.
// n = 10240 -> NW = 320, 320 * 32 = 10240 бит.
// Для увеличения лимита до 50000 нужно:
//   1) MAX_VERTICES = 50000, добавить case 1600 в switch(NW_used);
//   2) уменьшить MAX_DEPTH (иначе Frame не поместится в локальную память);
//   3) уменьшить maxCliques (буфер клик при NW=1600 занимает 6.4 КБ
//      на клику, т.е. 4M клик = 25.6 ГБ VRAM).
#define MAX_VERTICES 10240
#define MAX_DEPTH    20

// ============================ Проверка ошибок CUDA ============================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA ") + #call + ": " +     \
                                     cudaGetErrorString(_e));                  \
        }                                                                      \
    } while (0)

namespace {

/// RAII-обёртка над cudaMalloc/cudaFree.
/// Выделенная на GPU память всегда будет освобождена.
template <typename T>
struct DeviceBuffer {
    T* p = nullptr;
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t count) {
        CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    }
    void upload(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(p, host, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    void zero() { CUDA_CHECK(cudaMemset(p, 0, sizeof(T))); }
};

}  // namespace

// ============================ Ядро ============================
//
// Параллелизм: по одной нити на ориентированное ребро (v, c), v < c.
// Пара (v, c) — две наименьшие вершины клики размера >= 2, поэтому
// каждая такая клика обрабатывается ровно один раз (два уровня фиксации).
// Клики размера 1 (изолированные вершины) обрабатываются на хосте.
template <int NW>
__global__ void bkKernelT(
    const uint32_t* __restrict__ adjMask,
    int n,
    int nJobs,
    const int2* __restrict__ jobs,
    uint32_t* __restrict__ cliqueMasks,
    int* __restrict__ cliqueSizes,
    int* __restrict__ cliqueCount,
    int maxCliques) {

    struct Frame {
        uint32_t P[NW];      // кандидаты
        uint32_t X[NW];      // исключённые
        uint32_t cand[NW];   // P \ N(pivot) — перебираемые вершины
        int      r_size;     // размер текущей клики
    };

    Frame    stack[MAX_DEPTH];
    int      pushed_v[MAX_DEPTH];
    uint32_t R[NW];
    uint32_t PX[NW];

    const int stride = gridDim.x * blockDim.x;
    const int start  = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = start; j < nJobs; j += stride) {
        const int v = jobs[j].x;
        const int c = jobs[j].y;

        // R = {v, c}.
        for (int w = 0; w < NW; ++w) R[w] = 0;
        R[v >> 5] |= 1u << (v & 31);
        R[c >> 5] |= 1u << (c & 31);

        // ---- Корневой кадр ----
        // P = N(v) ∩ N(c) ∩ {u > c}, X = N(v) ∩ N(c) ∩ {u < c}.
        Frame& f0 = stack[0];
        const uint32_t* av = adjMask + static_cast<size_t>(v) * NW;
        const uint32_t* ac = adjMask + static_cast<size_t>(c) * NW;
        for (int w = 0; w < NW; ++w) {
            const uint32_t m = av[w] & ac[w];
            f0.P[w] = m;
            f0.X[w] = m;
        }
        const int cw = c >> 5, cb = c & 31;
        for (int w = 0; w < NW; ++w) {
            if (w < cw) {
                f0.P[w] = 0;
            } else if (w == cw) {
                f0.P[w] &= (cb == 31) ? 0u : ~((1u << (cb + 1)) - 1);
                f0.X[w] &= (cb == 0)  ? 0u :  ((1u << cb) - 1);
            } else {
                f0.X[w] = 0;
            }
        }
        f0.r_size  = 2;
        pushed_v[0] = c;

        // ---- Терминал ----
        bool pE = true, xE = true;
        for (int w = 0; w < NW; ++w) if (f0.P[w]) { pE = false; break; }
        for (int w = 0; w < NW; ++w) if (f0.X[w]) { xE = false; break; }
        if (pE && xE) {
            const int idx = atomicAdd(cliqueCount, 1);
            if (idx < maxCliques) {
                cliqueSizes[idx] = 2;
                for (int w = 0; w < NW; ++w)
                    cliqueMasks[static_cast<size_t>(idx) * NW + w] = R[w];
            }
            continue;
        }

        // ---- Пивот и кандидаты корневого кадра ----
        for (int w = 0; w < NW; ++w) PX[w] = f0.P[w] | f0.X[w];
        int pivot = -1, best = -1;
        for (int w = 0; w < NW; ++w) {
            uint32_t m = PX[w];
            while (m) {
                const int b = __ffs(m) - 1;
                m &= m - 1;
                const int u = w * 32 + b;
                const uint32_t* au = adjMask + static_cast<size_t>(u) * NW;
                int cnt = 0;
                for (int ww = 0; ww < NW; ++ww) cnt += __popc(au[ww] & f0.P[ww]);
                if (cnt > best) { best = cnt; pivot = u; }
            }
        }
        if (pivot < 0) {
            for (int w = 0; w < NW; ++w) f0.cand[w] = 0;
        } else {
            const uint32_t* ap = adjMask + static_cast<size_t>(pivot) * NW;
            for (int w = 0; w < NW; ++w) f0.cand[w] = f0.P[w] & ~ap[w];
        }

        // ---- Итеративный DFS ----
        int depth = 0;
        while (depth >= 0) {
            Frame& f = stack[depth];

            // Следующий кандидат: младший бит в cand ∩ P.
            int nextV = -1;
            for (int w = 0; w < NW; ++w) {
                const uint32_t m = f.cand[w] & f.P[w];
                if (m) { nextV = w * 32 + (__ffs(m) - 1); break; }
            }

            if (nextV < 0) {
                // Кадр исчерпан — pop.
                const int pv = pushed_v[depth];
                R[pv >> 5] &= ~(1u << (pv & 31));
                --depth;
                if (depth >= 0) {
                    Frame& par = stack[depth];
                    par.P[pv >> 5] &= ~(1u << (pv & 31));
                    par.X[pv >> 5] |=  (1u << (pv & 31));
                }
                continue;
            }

            if (depth + 1 >= MAX_DEPTH) {
                // Стек переполнен: молча пропускаем ветку, вершина уходит
                // в X, а не в новый кадр. Результат может быть неполным.
                f.P[nextV >> 5] &= ~(1u << (nextV & 31));
                f.X[nextV >> 5] |=  (1u << (nextV & 31));
                continue;
            }

            // Push: R ∪ {nextV}, новый кадр = (P ∩ N, X ∩ N).
            R[nextV >> 5] |= 1u << (nextV & 31);
            ++depth;
            Frame& nf = stack[depth];
            nf.r_size = f.r_size + 1;
            pushed_v[depth] = nextV;
            const uint32_t* an = adjMask + static_cast<size_t>(nextV) * NW;
            for (int w = 0; w < NW; ++w) {
                nf.P[w] = f.P[w] & an[w];
                nf.X[w] = f.X[w] & an[w];
            }

            bool pE2 = true, xE2 = true;
            for (int w = 0; w < NW; ++w) if (nf.P[w]) { pE2 = false; break; }
            for (int w = 0; w < NW; ++w) if (nf.X[w]) { xE2 = false; break; }
            if (pE2 && xE2) {
                const int idx = atomicAdd(cliqueCount, 1);
                if (idx < maxCliques) {
                    cliqueSizes[idx] = nf.r_size;
                    for (int w = 0; w < NW; ++w)
                        cliqueMasks[static_cast<size_t>(idx) * NW + w] = R[w];
                }
                R[nextV >> 5] &= ~(1u << (nextV & 31));
                --depth;
                f.P[nextV >> 5] &= ~(1u << (nextV & 31));
                f.X[nextV >> 5] |=  (1u << (nextV & 31));
                continue;
            }

            // Пиво нового кадра.
            for (int w = 0; w < NW; ++w) PX[w] = nf.P[w] | nf.X[w];
            int pv2 = -1, best2 = -1;
            for (int w = 0; w < NW; ++w) {
                uint32_t m = PX[w];
                while (m) {
                    const int b = __ffs(m) - 1;
                    m &= m - 1;
                    const int u = w * 32 + b;
                    const uint32_t* au = adjMask + static_cast<size_t>(u) * NW;
                    int cnt = 0;
                    for (int ww = 0; ww < NW; ++ww) cnt += __popc(au[ww] & nf.P[ww]);
                    if (cnt > best2) { best2 = cnt; pv2 = u; }
                }
            }
            if (pv2 < 0) {
                for (int w = 0; w < NW; ++w) nf.cand[w] = 0;
            } else {
                const uint32_t* ap = adjMask + static_cast<size_t>(pv2) * NW;
                for (int w = 0; w < NW; ++w) nf.cand[w] = nf.P[w] & ~ap[w];
            }
        }
    }
}

// ============================ Хост-функция ============================
CudaResult runBronKerboschCUDA(const std::vector<std::vector<int>>& adj,
                                int maxBlocks) {
    CudaResult out;
    const auto tStart = std::chrono::high_resolution_clock::now();

    const int n = static_cast<int>(adj.size());
    if (n == 0) { out.totalMs = 0.0; return out; }
    if (n > MAX_VERTICES) {
        throw std::runtime_error("n = " + std::to_string(n) +
                                 " > MAX_VERTICES = " + std::to_string(MAX_VERTICES));
    }

    // ---- Задания = ориентированные рёбра (v, c), v < c ----
    std::vector<int2> h_jobs;
    for (int v = 0; v < n; ++v) {
        for (int c : adj[v]) {
            if (c > v) h_jobs.push_back(make_int2(v, c));
        }
    }

    // ---- Изолированные вершины: клики размера 1 ----
    for (int v = 0; v < n; ++v) {
        if (adj[v].empty()) out.cliques.push_back({v});
    }

    if (h_jobs.empty()) {
        const auto tEnd = std::chrono::high_resolution_clock::now();
        out.totalMs  = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        out.kernelMs = 0.0;
        return out;
    }

    // ---- Ширина маски: NW = ceil(n / 32), округлённая до поддерживаемой ----
    const int nw = (n + 31) / 32;
    int NW_used;
    if      (nw <= 32)  NW_used = 32;
    else if (nw <= 64)  NW_used = 64;
    else if (nw <= 128) NW_used = 128;
    else if (nw <= 256) NW_used = 256;
    else if (nw <= 320) NW_used = 320;
    else
        throw std::runtime_error("n = " + std::to_string(n) +
                                 " требует NW > 320 (не поддерживается)");

    // ---- Битовая матрица смежности ----
    std::vector<uint32_t> h_adj(static_cast<size_t>(n) * NW_used, 0u);
    for (int u = 0; u < n; ++u) {
        for (int v : adj[u]) {
            h_adj[static_cast<size_t>(u) * NW_used + (v >> 5)] |= (1u << (v & 31));
        }
    }

    // ---- Буферы на устройстве (RAII) ----
    DeviceBuffer<uint32_t> d_adj, d_cm;
    DeviceBuffer<int2>     d_jobs;
    DeviceBuffer<int>      d_cs, d_cc;

    d_adj.allocate(h_adj.size());
    d_adj.upload(h_adj.data(), h_adj.size());
    d_jobs.allocate(h_jobs.size());
    d_jobs.upload(h_jobs.data(), h_jobs.size());

    // Максимум клик. 4M при NW=320 — это ~5.4 ГБ VRAM.
    const int maxCliques = 1 << 22;
    d_cm.allocate(static_cast<size_t>(maxCliques) * NW_used);
    d_cs.allocate(maxCliques);
    d_cc.allocate(1);
    d_cc.zero();

    const int blockSize = 128;

    #define LAUNCH(NWT)                                                          \
        do {                                                                     \
            int bpsm = 0;                                                        \
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(                       \
                &bpsm, bkKernelT<NWT>, blockSize, 0);                            \
            int smCount = 0;                                                     \
            cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0); \
            const int maxGrid = static_cast<int>(                                \
                (h_jobs.size() + blockSize - 1) / blockSize);                    \
            int gridSize = smCount * bpsm;                                       \
            if (maxBlocks > 0 && gridSize > maxBlocks) gridSize = maxBlocks;     \
            if (gridSize > maxGrid) gridSize = maxGrid;                          \
            if (gridSize < 1)       gridSize = 1;                                \
            bkKernelT<NWT><<<gridSize, blockSize>>>(                             \
                d_adj.p, n, static_cast<int>(h_jobs.size()), d_jobs.p,           \
                d_cm.p, d_cs.p, d_cc.p, maxCliques);                             \
            out.gridSize  = gridSize;                                            \
            out.blockSize = blockSize;                                           \
        } while (0)

    // ---- Запуск ядра (grid-stride) + замер времени ----
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));

    switch (NW_used) {
        case 32:  LAUNCH(32);  break;
        case 64:  LAUNCH(64);  break;
        case 128: LAUNCH(128); break;
        case 256: LAUNCH(256); break;
        case 320: LAUNCH(320); break;
    }
    #undef LAUNCH

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float kMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kMs, evStart, evStop));
    out.kernelMs = static_cast<double>(kMs);
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    CUDA_CHECK(cudaGetLastError());

    // ---- Чтение результата ----
    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_cc.p, sizeof(int), cudaMemcpyDeviceToHost));
    const int cnt = std::min(h_count, maxCliques);

    std::vector<uint32_t> h_masks(static_cast<size_t>(cnt) * NW_used, 0u);
    std::vector<int>      h_sizes(cnt, 0);
    if (cnt > 0) {
        CUDA_CHECK(cudaMemcpy(h_masks.data(), d_cm.p,
                              static_cast<size_t>(cnt) * NW_used * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sizes.data(), d_cs.p,
                              static_cast<size_t>(cnt) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    out.cliques.reserve(out.cliques.size() + cnt);
    for (int i = 0; i < cnt; ++i) {
        std::vector<int> cl;
        cl.reserve(h_sizes[i]);
        for (int w = 0; w < NW_used; ++w) {
            uint32_t m = h_masks[static_cast<size_t>(i) * NW_used + w];
            while (m) {
                const int b = __builtin_ffs(m) - 1;
                cl.push_back(w * 32 + b);
                m &= m - 1;
            }
        }
        std::sort(cl.begin(), cl.end());
        out.cliques.push_back(std::move(cl));
    }

    if (h_count > maxCliques) {
        throw std::runtime_error("Переполнение буфера клик: " +
                                 std::to_string(h_count) + " > " +
                                 std::to_string(maxCliques));
    }

    const auto tEnd = std::chrono::high_resolution_clock::now();
    out.totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    return out;
}

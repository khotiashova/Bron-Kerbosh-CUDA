// main.cpp
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bk_cuda.cuh"
#include "bk_sequential.h"
#include "graph_generator.h"

using namespace std;
using Clock = chrono::high_resolution_clock;

namespace {

/// Читает граф из файла: первая строка — n, далее пары "u v" (по ребру).
/// Некорректные строки (индексы вне [0, n), self-loops) пропускаются.
vector<vector<int>> loadGraphFromFile(const string& filename) {
    ifstream in(filename);
    if (!in) throw runtime_error("Не удалось открыть файл: " + filename);

    int n;
    if (!(in >> n) || n <= 0)
        throw runtime_error("Не удалось прочитать число вершин");

    vector<vector<int>> adj(n);
    int u, v;
    while (in >> u >> v) {
        if (u < 0 || u >= n || v < 0 || v >= n) continue;
        if (u == v) continue;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    return adj;
}

/// Приводит список клик к каноническому виду для сравнения:
/// сортирует вершины внутри клики, затем сортирует сами клики.
vector<vector<int>> canonicalize(vector<vector<int>> cliques) {
    for (auto& c : cliques) sort(c.begin(), c.end());
    sort(cliques.begin(), cliques.end());
    return cliques;
}

void printUsage(const char* prog) {
    cout <<
        "Использование:\n"
        "  " << prog << " --file <graph.txt> [--seed <N>]\n"
        "  " << prog << " --generate <N> <density_%> [--seed <N>]\n"
        "  " << prog << " --generate <N> --edges <M> [--seed <N>]\n"
        "  " << prog << " --gen-by-cliques <M> <density_%> [--seed <N>]\n\n"
        "Параметры:\n"
        "  --generate N density        граф G(N, density%) по Эрдёшу–Реньи\n"
        "  --generate N --edges M      ровно M рёбер, выбранных равномерно\n"
        "  --gen-by-cliques M density  граф с не менее M максимальных клик\n"
        "                              при плотности density (в процентах)\n"
        "  --file <path>               загрузить граф из файла\n"
        "  --save <path>               сохранить построенный граф в файл\n"
        "  --seed <N>                  seed генератора (по умолчанию 42);\n"
        "                              можно также указать позиционно после N density\n\n"
        "Флаги:\n"
        "  --cpu-only        запустить только CPU-версию\n"
        "  --gpu-only        запустить только CUDA-версию\n"
        "  --no-verify       не сверять результаты CPU и GPU\n"
        "  --gpu-blocks N    ограничить число CUDA-блоков (0 = авто)\n"
        "  --quiet           печатать минимум\n"
        "  -h, --help        показать эту справку\n";
}

}  // namespace

int main(int argc, char** argv) {
    string    filename;
    string    saveFilename;
    int       genN       = 0;
    int       gpuBlocks  = 0;  // 0 = auto
    double    genDensity = 0.0;
    long long genEdges   = -1;
    unsigned  genSeed    = 42;

    bool runCpu = true;
    bool runGpu = true;
    bool verify = true;
    bool quiet  = false;

    long long targetCliques = -1;
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            saveFilename = argv[++i];
        } else if (arg == "--generate" && i + 1 < argc) {
            genN = stoi(argv[++i]);
            // Опциональная плотность сразу после N.
            if (i + 1 < argc && argv[i + 1][0] != '-'
                             && string(argv[i + 1]) != "--edges"
                             && string(argv[i + 1]) != "--seed") {
                genDensity = stod(argv[++i]);
                // Опциональный позиционный seed сразу после плотности.
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    genSeed = static_cast<unsigned>(stoul(argv[++i]));
                }
            }
        } else if (arg == "--edges" && i + 1 < argc) {
            genEdges = stoll(argv[++i]);
            // Опциональный позиционный seed сразу после M.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                genSeed = static_cast<unsigned>(stoul(argv[++i]));
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            genSeed = static_cast<unsigned>(stoul(argv[++i]));
        } else if (arg == "--cpu-only")   runGpu = false;
        else if (arg == "--gpu-only")     runCpu = false;
        else if (arg == "--no-verify")    verify = false;
        else if (arg == "--quiet")        quiet  = true;
        else if (arg == "-h" || arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg == "--gpu-blocks" && i + 1 < argc) { gpuBlocks = stoi(argv[++i]); }
        else if (arg == "--gen-by-cliques" && i + 2 < argc) {
            targetCliques = stoll(argv[++i]);
            genDensity = stod(argv[++i]);
        }
        else {
            cerr << "Неизвестный аргумент: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // ---- Получение графа ----
    vector<vector<int>> adj;
    try {
        if (!filename.empty()) {
            adj = loadGraphFromFile(filename);
            if (!quiet) cout << "Загружен граф из файла: " << filename << "\n";
        } else if (genN > 0 && genEdges >= 0) {
            adj = GraphGenerator::generateWithEdges(genN, genEdges, genSeed);
            if (!quiet)
                cout << "Сгенерирован граф: n=" << genN
                     << ", рёбер=" << genEdges
                     << ", seed=" << genSeed << "\n";
        } else if (genN > 0) {
            adj = GraphGenerator::generateByDensity(genN, genDensity, genSeed);
            if (!quiet)
                cout << "Сгенерирован граф G(" << genN << ", плотность "
                     << genDensity << "%, seed=" << genSeed << ")\n";
        } else if (targetCliques >= 0) {
            adj = GraphGenerator::generateByMaxCliques(targetCliques, genDensity / 100.0, genSeed);
            if (!quiet) 
                cout << "Сгенерирован граф с ~" << targetCliques 
                     << " клик (плотность " << genDensity << "%)\n";
        } else {
            cerr << "Укажите --file <graph.txt> или --generate <N> [density|--edges M]\n";
            printUsage(argv[0]);
            return 1;
        }
    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }

    if (!saveFilename.empty()) {
        GraphGenerator::saveToFile(adj, saveFilename);
        if (!quiet) cout << "Граф сохранен в файл: " << saveFilename << "\n";
    }

    const int n = static_cast<int>(adj.size());
    size_t edges = 0;
    for (const auto& v : adj) edges += v.size();
    edges /= 2;

    if (!quiet)
        cout << "Вершин: " << n << ", рёбер: " << edges << "\n\n";

    vector<vector<int>> cpuCliques, gpuCliques;
    double cpuMs = 0.0, gpuKernelMs = 0.0, gpuTotalMs = 0.0;

    // ---- CPU ----
    if (runCpu) {
        try {
            BronKerboschSequential bk(n);
            for (int u = 0; u < n; ++u) {
                for (int v : adj[u]) {
                    if (u < v) bk.addEdge(u, v);
                }
            }
            bk.run();
            cpuCliques = bk.getMaximalCliques();
            cpuMs      = bk.getLastRunMs();

            cout << "[CPU] Найдено максимальных клик: " << cpuCliques.size()
                 << ", время алгоритма: " << fixed << setprecision(3)
                 << cpuMs << " мс\n";
        } catch (const exception& e) {
            cerr << "[CPU] Ошибка: " << e.what() << "\n";
            return 3;
        }
    }

    // ---- GPU ----
    if (runGpu) {
        try {
            CudaResult res = runBronKerboschCUDA(adj, gpuBlocks);
            gpuCliques  = std::move(res.cliques);
            gpuKernelMs = res.kernelMs;
            gpuTotalMs  = res.totalMs;

            cout << "[GPU] Найдено максимальных клик: " << gpuCliques.size()
                << ", время ядра: " << fixed << setprecision(3) << gpuKernelMs
                << " мс, полное время (с копиями): " << gpuTotalMs << " мс"
                << ", grid = " << res.gridSize << " x " << res.blockSize << "\n";
        } catch (const exception& e) {
            cerr << "[GPU] Ошибка: " << e.what() << "\n";
            return 4;
        }
    }

    // ---- Проверка ----
    if (verify && runCpu && runGpu) {
        auto a = canonicalize(cpuCliques);
        auto b = canonicalize(gpuCliques);
        if (a == b) {
            cout << "\nРезультаты CPU и GPU совпадают \u2713\n";
        } else {
            cout << "\nРезультаты НЕ совпадают \u2717  (CPU: " << a.size()
                 << ", GPU: " << b.size() << ")\n";
            size_t k = 0;
            for (; k < a.size() && k < b.size(); ++k) {
                if (a[k] != b[k]) break;
            }
            if (k < a.size() && k < b.size()) {
                cout << "Первое расхождение на позиции " << k << ":\n  CPU: ";
                for (int v : a[k]) cout << v << ' ';
                cout << "\n  GPU: ";
                for (int v : b[k]) cout << v << ' ';
                cout << "\n";
            }
            return 2;
        }
    }

    // ---- Ускорения ----
    if (runCpu && runGpu && cpuMs > 0.0 && gpuKernelMs > 0.0 && gpuTotalMs > 0.0) {
        cout << "\nУскорение (CPU / GPU ядро)         : "
             << fixed << setprecision(2) << (cpuMs / gpuKernelMs) << "x\n";
        cout << "Ускорение (CPU / GPU полное время) : "
             << (cpuMs / gpuTotalMs) << "x\n";
    }

    return 0;
}

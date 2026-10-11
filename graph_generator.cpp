#include "graph_generator.h"

#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

std::vector<std::vector<int>>
GraphGenerator::generateErdosRenyi(int n, double p, unsigned seed) {
    std::mt19937 gen(seed);
    std::bernoulli_distribution dist(p);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (dist(gen)) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    return adj;
}

std::vector<std::vector<int>>
GraphGenerator::generateByDensity(int n, double densityPercent, unsigned seed) {
    return generateErdosRenyi(n, densityPercent / 100.0, seed);
}

std::vector<std::vector<int>>
GraphGenerator::generateWithEdges(int n, long long m, unsigned seed) {
    const long long maxEdges = static_cast<long long>(n) * (n - 1) / 2;
    if (m < 0 || m > maxEdges) {
        throw std::runtime_error("m=" + std::to_string(m) +
                                 " вне диапазона [0, " + std::to_string(maxEdges) + "]");
    }

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    std::vector<std::vector<int>> adj(n);
    std::set<std::pair<int, int>> seen;

    long long added = 0;
    while (added < m) {
        int u = dist(gen), v = dist(gen);
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        if (!seen.emplace(u, v).second) continue;
        adj[u].push_back(v);
        adj[v].push_back(u);
        ++added;
    }
    return adj;
}

void GraphGenerator::saveToFile(const std::vector<std::vector<int>>& adj,
                                const std::string& filename) {
    std::ofstream out(filename);
    const int n = static_cast<int>(adj.size());
    out << n << "\n";
    for (int i = 0; i < n; ++i) {
        for (int j : adj[i]) {
            if (i < j) out << i << ' ' << j << "\n";
        }
    }
}

#include "graph_generator.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
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

std::vector<std::set<int>>
MaxCliqueTracker::bronKerboschOnSet(
    const std::set<int>& S,
    const std::vector<std::vector<int>>& adj)
{
    std::vector<std::set<int>> out;
    std::set<int> R, P = S, X;

    std::function<void(std::set<int>&, std::set<int>&, std::set<int>&)> bk =
        [&](std::set<int>& R_, std::set<int>& P_, std::set<int>& X_) {
            if (P_.empty() && X_.empty()) {
                out.push_back(R_);
                return;
            }

            // Пивот — вершина из P ∪ X с максимумом соседей в P.
            int pivot = -1, best = -1;
            auto scan = [&](const std::set<int>& s) {
                for (int u : s) {
                    int d = 0;
                    for (int w : adj[u]) if (P_.count(w)) ++d;
                    if (d > best) { best = d; pivot = u; }
                }
            };
            scan(P_); scan(X_);

            // Кандидаты — P без соседей пивота.
            std::vector<int> cand;
            for (int u : P_) {
                if (pivot < 0 ||
                    !std::binary_search(adj[pivot].begin(),
                                        adj[pivot].end(), u))
                    cand.push_back(u);
            }

            for (int u : cand) {
                R_.insert(u);
                std::set<int> nP, nX;
                for (int w : P_)
                    if (std::binary_search(adj[u].begin(), adj[u].end(), w))
                        nP.insert(w);
                for (int w : X_)
                    if (std::binary_search(adj[u].begin(), adj[u].end(), w))
                        nX.insert(w);
                bk(R_, nP, nX);
                R_.erase(u);
                P_.erase(u);
                X_.insert(u);
            }
        };

    bk(R, P, X);
    return out;
}

void MaxCliqueTracker::insertClique(std::set<int> C) {
    auto [it, ok] = cliques_.insert(std::move(C));
    if (!ok) return;  // такая клика уже есть — по логике не должно
    for (int w : *it) index_[w].insert(&*it);
}

void MaxCliqueTracker::addVertex() {
    const int v = static_cast<int>(adj_.size());
    adj_.push_back({});
    insertClique(std::set<int>{v});
}

void MaxCliqueTracker::addEdge(int u, int v) {
    // Порядок в adj_ сохраняется:
    //   v — новый максимум для adj_[u];
    //   u меньше всех уже добавленных соседей v в adj_[v]
    //   (рёбра в adj_[v] добавлялись только с меньшими индексами).
    adj_[u].push_back(v);
    adj_[v].push_back(u);

    // N_uv = N(u) ∩ N(v).
    std::vector<int> N_uv;
    std::set_intersection(
        adj_[u].begin(), adj_[u].end(),
        adj_[v].begin(), adj_[v].end(),
        std::back_inserter(N_uv));
    const std::set<int> N_uv_set(N_uv.begin(), N_uv.end());

    // 1) Новые клики: BK на G[N_uv] + {u, v}.
    auto sub = bronKerboschOnSet(N_uv_set, adj_);

    // 2) Уничтожаемые клики — только те, что содержат u или v.
    std::vector<const std::set<int>*> removed;

    // 2a) C э u, C\{u} \in N(v).
    if (auto it_u = index_.find(u); it_u != index_.end()) {
        for (const auto* Cp : it_u->second) {
            const auto& C = *Cp;
            bool ok = true;
            for (int w : C) {
                if (w == u) continue;
                if (!std::binary_search(adj_[v].begin(),
                                        adj_[v].end(), w)) {
                    ok = false; break;
                }
            }
            if (ok) removed.push_back(Cp);
        }
    }
    // 2b) C э v, C\{v} \in N(u).
    if (auto it_v = index_.find(v); it_v != index_.end()) {
        for (const auto* Cp : it_v->second) {
            const auto& C = *Cp;
            bool ok = true;
            for (int w : C) {
                if (w == v) continue;
                if (!std::binary_search(adj_[u].begin(),
                                        adj_[u].end(), w)) {
                    ok = false; break;
                }
            }
            if (ok) removed.push_back(Cp);
        }
    }

    // 3) Удаляем: сначала из index_, потом из cliques_.
    for (const auto* Cp : removed) {
        for (int w : *Cp) {
            auto it_idx = index_.find(w);
            if (it_idx == index_.end()) continue;
            it_idx->second.erase(Cp);
            if (it_idx->second.empty()) index_.erase(it_idx);
        }
        cliques_.erase(cliques_.find(*Cp));
    }

    // 4) Вставляем новые.
    for (auto& C : sub) {
        C.insert(u);
        C.insert(v);
        insertClique(std::move(C));
    }
}

std::vector<std::vector<int>>
GraphGenerator::generateByEdges(long long m, double p, unsigned seed) {
    std::mt19937 gen(seed);
    std::bernoulli_distribution dist(p);

    std::vector<std::vector<int>> adj;
    long long currentNumEdges = 0;

    while (currentNumEdges < m) {
        const int v = static_cast<int>(adj.size());
        std::vector<int> neighbors;
        for (int u = 0; u < v && currentNumEdges < m; ++u) {
            if (!dist(gen)) continue;
            adj[u].push_back(v);
            neighbors.push_back(u);
            ++currentNumEdges;
        }
        adj.push_back(std::move(neighbors));
    }
    return adj;
}

std::vector<std::vector<int>>
GraphGenerator::generateByMaxCliques(long long m, double p, unsigned seed) {
    if (m == 0) return {};

    std::mt19937 gen(seed);
    std::bernoulli_distribution dist(p);

    MaxCliqueTracker tracker;
    tracker.addVertex();  // одна вершина — одна клика

    int current_v = 0;    // индекс последней добавленной вершины
    int u = 0;            // очередной кандидат в соседи для current_v

    while (tracker.count() < m) {
        if (u >= current_v) {
            // Все u для текущей вершины перебраны — добавляем новую.
            // +1 клика гарантирует завершение, даже если p = 0.
            tracker.addVertex();
            ++current_v;
            u = 0;
            continue;
        }
        if (dist(gen)) tracker.addEdge(u, current_v);
        ++u;
    }

    return tracker.graph();
}

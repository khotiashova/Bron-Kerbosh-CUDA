#include "bk_sequential.h"

#include <algorithm>
#include <chrono>
#include <iterator>

using std::vector;

namespace {

/// Пересечение двух отсортированных векторов (a ∩ b).
vector<int> intersect_(const vector<int>& a, const vector<int>& b) {
    vector<int> r;
    r.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(r));
    return r;
}

/// Разность двух отсортированных векторов (a \ b).
vector<int> difference_(const vector<int>& a, const vector<int>& b) {
    vector<int> r;
    r.reserve(a.size());
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(r));
    return r;
}

/// Объединение двух отсортированных векторов (a ∪ b).
vector<int> union_(const vector<int>& a, const vector<int>& b) {
    vector<int> r;
    r.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(r));
    return r;
}

/// |a ∩ b| без построения пересечения (O(|a| + |b|)).
size_t countIntersect(const vector<int>& a, const vector<int>& b) {
    size_t i = 0, j = 0, cnt = 0;
    while (i < a.size() && j < b.size()) {
        if      (a[i] == b[j]) { ++cnt; ++i; ++j; }
        else if (a[i] <  b[j]) ++i;
        else                   ++j;
    }
    return cnt;
}

}  // namespace

BronKerboschSequential::BronKerboschSequential(int n)
    : n_(n), adj_(n) {}

void BronKerboschSequential::addEdge(int u, int v) {
    adj_[u].push_back(v);
    adj_[v].push_back(u);
}

void BronKerboschSequential::run() {
    maximal_cliques_.clear();

    // Списки смежности должны быть отсортированы для set_intersection.
    for (auto& a : adj_) std::sort(a.begin(), a.end());

    const auto t0 = std::chrono::high_resolution_clock::now();

    // На старте P = {0, 1, ..., n-1}, R = X = ∅.
    vector<int> R, P(n_), X;
    for (int i = 0; i < n_; ++i) P[i] = i;

    bronKerbosch(std::move(R), std::move(P), std::move(X));

    const auto t1 = std::chrono::high_resolution_clock::now();
    lastRunMs_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::vector<std::vector<int>>
BronKerboschSequential::getMaximalCliques() const {
    return maximal_cliques_;
}

void BronKerboschSequential::bronKerbosch(vector<int> R,
                                          vector<int> P,
                                          vector<int> X) {
    // База рекурсии: клика максимальна, если расширять её больше нечем.
    if (P.empty() && X.empty()) {
        maximal_cliques_.push_back(std::move(R));
        return;
    }

    // Пивот: u ∈ P ∪ X с максимумом |N(u) ∩ P|. Перебираем только
    // P \ N(pivot) — это радикально сокращает ветвление на плотных графах.
    // Корректность сохраняется: любая клика, содержащая вершину из
    // P ∩ N(pivot), будет найдена из ветки этой вершины.
    const vector<int> PX = union_(P, X);
    int pivot = -1;
    size_t best = 0;
    for (int u : PX) {
        const size_t cnt = countIntersect(adj_[u], P);
        if (pivot < 0 || cnt > best) { best = cnt; pivot = u; }
    }

    // candidates = P \ N(pivot).
    const vector<int> candidates = difference_(P, adj_[pivot]);

    for (int v : candidates) {
        // R ∪ {v} с сохранением порядка.
        vector<int> new_R = R;
        new_R.insert(std::lower_bound(new_R.begin(), new_R.end(), v), v);

        vector<int> new_P = intersect_(P, adj_[v]);
        vector<int> new_X = intersect_(X, adj_[v]);

        bronKerbosch(std::move(new_R), std::move(new_P), std::move(new_X));

        // P := P \ {v}, X := X ∪ {v} — с сохранением порядка.
        P.erase(std::lower_bound(P.begin(), P.end(), v));
        X.insert(std::lower_bound(X.begin(), X.end(), v), v);
    }
}

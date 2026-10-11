#pragma once
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

/**
 * @brief Инкрементальный трекер максимальных клик.
 *
 * Правила обновления при добавлении ребра (u, v):
 *   N_uv = N(u) ∩ N(v).
 *  - Новые клики    = { {u,v} ∪ C : C ∈ MaxCliques(G[N_uv]) }.
 *  - Уничтожаются   = { C : u ∈ C и C\{u} \in N(v) }
 *                    ∪ { C : v ∈ C и C\{v} \in N(u) }.
 *  Фактически это алгоритм Брона-Кербоша
 */
class MaxCliqueTracker {
public:
    /// Списки смежности графа (отсортированы по возрастанию).
    const std::vector<std::vector<int>>& graph() const {
        return adj_;
    }

    /// Текущее число максимальных клик.
    long long count() const {
        return static_cast<long long>(cliques_.size());
    }

    /// Добавить изолированную вершину v = adj_.size(). count += 1.
    void addVertex();

    /**
     * @brief Добавить ребро (u, v).
     *
     * Обе вершины уже должны существовать в графе. Инкрементально
     * обновляет cliques_ и index_.
     */
    void addEdge(int u, int v);

private:
    /**
     * @brief Брон–Кербош с пивотированием на индуцированном G[S].
     *
     * @param S   Множество вершин подграфа.
     * @param adj Полный граф (списки соседей отсортированы).
     * @return Все максимальные клики G[S]. Для пустого S —
     *         {{}} (одна «пустая» клика).
     */
    static std::vector<std::set<int>>
    bronKerboschOnSet(const std::set<int>& S,
                      const std::vector<std::vector<int>>& adj);

    /// Вставить клику в cliques_ и обновить index_.
    void insertClique(std::set<int> C);

    std::vector<std::vector<int>> adj_;      // списки соседей, по возрастанию
    std::set<std::set<int>>      cliques_;   // все максимальные клики
    // Индекс: вершина -> клики, содержащие её. 
    // (оптимизация - при добавлении ребра проверяем не все клики)
    std::unordered_map<int, std::set<const std::set<int>*>> index_;
};

/**
 * @brief Генератор тестовых неориентированных графов.
 *
 * Все методы детерминированы по seed: одинаковые параметры → одинаковый
 * граф. Петли и кратные рёбра не создаются.
 */
class GraphGenerator {
public:
    /**
     * @brief Граф G(n, p) по модели Эрдёша — Реньи: каждое из C(n,2)
     *        рёбер независимо присутствует с вероятностью p.
     * @param n Число вершин, n > 0.
     * @param p Вероятность ребра, 0.0 <= p <= 1.0.
     * @param seed Seed генератора.
     * @return Список смежности размера n.
     */
    static std::vector<std::vector<int>>
    generateErdosRenyi(int n, double p, unsigned seed = 42);

    /**
     * @brief Эквивалент generateErdosRenyi(n, densityPercent / 100).
     * @param densityPercent Плотность в процентах, 0..100.
     */
    static std::vector<std::vector<int>>
    generateByDensity(int n, double densityPercent, unsigned seed = 42);

    /**
     * @brief Ровно m различных рёбер, выбранных равномерно из C(n,2).
     * @param m Число рёбер, 0 <= m <= C(n,2).
     * @throws std::runtime_error если m вне диапазона.
     */
    static std::vector<std::vector<int>>
    generateWithEdges(int n, long long m, unsigned seed = 42);

    /**
     * @brief Сохраняет граф в файл: первая строка — n, далее пары "u v"
     *        для каждого ребра (u < v).
     */
    static void saveToFile(const std::vector<std::vector<int>>& adj,
                           const std::string& filename);

    /**
     * @brief Генерирует граф, набирая хотя бы m рёбер.
     *
     * Растём по одной вершине; каждое ребро к уже существующей вершине
     * проводим с вероятностью p (модель Эрдёша–Реньи).
     *
     * @param m    Целевое (нижнее) число рёбер, m >= 0.
     * @param p    Вероятность ребра, 0 <= p <= 1.
     * @param seed Зерно генератора.
     */
    static std::vector<std::vector<int>>
    generateByEdges(long long m, double p, unsigned seed = 42);

    /**
     * @brief Генерирует граф ровно с m максимальными кликами.
     *
     * Растём по одной вершине; каждое ребро к уже существующей вершине
     * проводим с вероятностью p (модель Эрдёша–Реньи). 
     * Критерий остановки — число максимальных клик. Может давать
     * больше, чем m, но не значительно.
     *
     * @param m    Число максимальных клик, m >= 0.
     * @param p    Вероятность ребра, 0 <= p <= 1.
     * @param seed Зерно генератора.
     */
    static std::vector<std::vector<int>>
    generateByMaxCliques(long long m, double p, unsigned seed = 42);

};

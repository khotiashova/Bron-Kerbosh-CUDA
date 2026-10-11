#pragma once
#include <string>
#include <vector>

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
};

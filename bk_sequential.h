#pragma once
#include <vector>

/**
 * @brief Последовательная реализация алгоритма Брона-Кербоша
 *        с выбором опорной вершины (pivoting, Tomita et al.).
 *
 * Класс хранит неориентированный граф в виде отсортированных списков
 * смежности и позволяет найти все максимальные клики вызовом run().
 *
 * Сложность и число клик в общем случае экспоненциальны по n;
 * пивот сокращает ветвление на плотных графах.
 *
 * Петли и кратные рёбра не поддерживаются.
 */
class BronKerboschSequential {
public:
    /**
     * @brief Создаёт объект для графа с n вершинами (0..n-1), без рёбер.
     * @param n Число вершин, n >= 0.
     */
    explicit BronKerboschSequential(int n);

    /**
     * @brief Добавляет неориентированное ребро (u, v).
     * @param u Первая вершина, 0 <= u < n.
     * @param v Вторая вершина, 0 <= v < n.
     * @note Петли и кратные рёбра не поддерживаются.
     */
    void addEdge(int u, int v);

    /**
     * @brief Запускает поиск всех максимальных клик.
     * @note Повторный вызов перезаписывает предыдущий результат.
     */
    void run();

    /**
     * @brief Возвращает список всех найденных максимальных клик.
     * @return Клики, каждая — отсортированный по возрастанию вектор вершин.
     */
    std::vector<std::vector<int>> getMaximalCliques() const;

    /// Время последнего успешного run() в миллисекундах.
    double getLastRunMs() const { return lastRunMs_; }

private:
    int n_;
    std::vector<std::vector<int>> adj_;
    std::vector<std::vector<int>> maximal_cliques_;
    double lastRunMs_ = 0.0;

    // Рекурсивный обход. R, P, X — отсортированные векторы:
    //   R — текущая клика;
    //   P — кандидаты (смежные со всеми из R и ещё не обработанные);
    //   X — исключённые (смежные со всеми из R и уже обработанные).
    void bronKerbosch(std::vector<int> R,
                      std::vector<int> P,
                      std::vector<int> X);
};

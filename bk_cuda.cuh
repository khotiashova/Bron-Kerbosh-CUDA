#pragma once
#include <vector>

/// Результат работы CUDA-версии Bron–Kerbosch.
struct CudaResult {
    std::vector<std::vector<int>> cliques;  ///< Все максимальные клики.
    double kernelMs  = 0.0;                 ///< Чистое время ядра (cudaEvent), мс.
    double totalMs   = 0.0;                 ///< Полное время (malloc, H2D, ядро, D2H, декод), мс.
    int    gridSize  = 0;                   ///< Сколько CUDA-блоков реально запущено.
    int    blockSize = 0;                   ///< Размер блока (потоков на блок).
};

/**
 * @brief Находит все максимальные клики графа на GPU.
 *
 * @param adj        Список смежности (0..n-1).
 * @param maxBlocks  Максимальное число CUDA-блоков. 0 = автоматически (по occupancy).
 *                   Используется для экспериментов по сильной масштабируемости.
 * @return CudaResult с кликами, временами и параметрами запуска.
 *
 * @throws std::runtime_error при n > MAX_VERTICES, переполнении буфера клик,
 *         ошибке CUDA или иной внутренней ошибке.
 */
CudaResult runBronKerboschCUDA(const std::vector<std::vector<int>>& adj,
                               int maxBlocks = 0);

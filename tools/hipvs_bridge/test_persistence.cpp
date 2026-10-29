/* Manual GPU-machine test for the GPU snapshot bridge round trip.
 *
 *   test_persistence write /tmp/snapshot.bin   # build + export graph
 *   test_persistence read  /tmp/snapshot.bin   # rebuild from (graph, vectors)
 *
 * The vectors are regenerated deterministically on both sides, exactly like the
 * seekdb snapshot stores the graph next to the vectors the caller already owns. */
#include "seekdb_cuvs_bridge.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void make_vectors(std::vector<float> &vectors, long count, long dimension)
{
  vectors.resize(static_cast<size_t>(count) * dimension);
  for (long row = 0; row < count; ++row) {
    for (long column = 0; column < dimension; ++column) {
      vectors[row * dimension + column] = std::sin(float(row * 31 + column * 17));
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 3) return 2;
  const long count = 10000;
  const long dimension = 32;
  std::vector<float> vectors;
  make_vectors(vectors, count, dimension);
  int rc = 0;

  if (std::strcmp(argv[1], "write") == 0) {
    void *index = seekdb_cuvs_build(vectors.data(), count, dimension);
    if (index == nullptr) return 3;
    const long graph_bytes = seekdb_cuvs_graph_bytes(index);
    if (graph_bytes <= 0) { seekdb_cuvs_free(index); return 4; }
    /* The graph degree is not fixed by the bridge; derive it from the byte size. */
    if (graph_bytes % (count * static_cast<long>(sizeof(unsigned int))) != 0) {
      seekdb_cuvs_free(index);
      return 5;
    }
    const long degree = graph_bytes / (count * static_cast<long>(sizeof(unsigned int)));
    std::vector<unsigned int> graph(static_cast<size_t>(graph_bytes) / sizeof(unsigned int));
    if (seekdb_cuvs_export_graph(index, graph.data(), graph_bytes) != 0) {
      seekdb_cuvs_free(index);
      return 6;
    }
    seekdb_cuvs_free(index);
    FILE *file = std::fopen(argv[2], "wb");
    if (file == nullptr) return 7;
    const long header[3] = {count, dimension, degree};
    const bool ok = std::fwrite(header, sizeof(header), 1, file) == 1 &&
                    std::fwrite(graph.data(), 1, static_cast<size_t>(graph_bytes), file) ==
                        static_cast<size_t>(graph_bytes);
    std::fclose(file);
    if (!ok) return 8;
    std::printf("write: n=%ld dim=%ld degree=%ld graph_bytes=%ld\n",
                count, dimension, degree, graph_bytes);
    return 0;
  }

  FILE *file = std::fopen(argv[2], "rb");
  if (file == nullptr) return 9;
  long header[3] = {0, 0, 0};
  if (std::fread(header, sizeof(header), 1, file) != 1) { std::fclose(file); return 10; }
  const long n = header[0], dim = header[1], degree = header[2];
  if (n != count || dim != dimension || degree <= 0) { std::fclose(file); return 11; }
  const long graph_bytes = n * degree * static_cast<long>(sizeof(unsigned int));
  std::vector<unsigned int> graph(static_cast<size_t>(graph_bytes) / sizeof(unsigned int));
  const bool read_ok = std::fread(graph.data(), 1, static_cast<size_t>(graph_bytes), file) ==
                       static_cast<size_t>(graph_bytes);
  std::fclose(file);
  if (!read_ok) return 12;

  /* Shape mismatches have to be rejected instead of building a broken index. */
  void *invalid = seekdb_cuvs_build_from_graph(vectors.data(), count - 1, dimension,
                                               graph.data(), degree);
  if (invalid != nullptr) { seekdb_cuvs_free(invalid); return 13; }
  invalid = seekdb_cuvs_build_from_graph(vectors.data(), count, dimension + 1,
                                         graph.data(), degree);
  if (invalid != nullptr) { seekdb_cuvs_free(invalid); return 14; }

  void *index = seekdb_cuvs_build_from_graph(vectors.data(), n, dim, graph.data(), degree);
  if (index == nullptr) return 15;
  unsigned int ids[10];
  float distances[10];
  rc = seekdb_cuvs_search(index, vectors.data(), 1, 10, ids, distances);
  seekdb_cuvs_free(index);
  if (rc != 0 || ids[0] != 0 || std::fabs(distances[0]) > 0.0001f) return 16;
  std::printf("read: n=%ld top1=%u distance=%g\n", n, ids[0], distances[0]);
  return 0;
}

#ifndef SEEKDB_CUVS_BRIDGE_H
#define SEEKDB_CUVS_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif
/* One-shot: build a CAGRA index over base (n x dim, row-major f32) on the GPU and
 * search queries (nq x dim), writing top-k uint32 neighbor ids to out_ids
 * (length nq*topk). Returns 0 on success, non-zero on error. */
int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids);

/* Build-once / search-many (used by seekdb's ob_vsag_adaptor GPU path).
 *  - seekdb_cuvs_build: builds a CAGRA index on the GPU from base (n x dim,
 *    row-major f32) and returns an opaque handle (NULL on error). The dataset is
 *    uploaded once through pinned host staging; `base` may be freed after this
 *    returns.
 *  - seekdb_cuvs_search: runs top-k search for nq queries (row-major f32, dim
 *    cols), writing uint32 row offsets to out_ids[nq*topk] and (if non-NULL) L2
 *    distances to out_dist[nq*topk]. Returns 0 on success.
 *  - seekdb_cuvs_free: releases the handle.
 * All ROCm/cuVS state lives inside the .so; seekdb only ever sees these C symbols. */
void* seekdb_cuvs_build(const float* base, long n, long dim);
int   seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                         unsigned int* out_ids, float* out_dist);
void  seekdb_cuvs_free(void* handle);

/* GPU snapshot support (one pass of large block transfers).
 *  - seekdb_cuvs_graph_bytes: size in bytes of the index's row-major [n, degree]
 *    uint32 kNN graph (<= 0 on error).
 *  - seekdb_cuvs_export_graph: copy that graph device->host through pinned
 *    staging into host_out (capacity bytes); the vectors are not copied because
 *    the caller already owns them on the host.
 *  - seekdb_cuvs_build_from_graph: rebuild an index from an exported graph plus
 *    its dataset without re-running graph construction. Returns NULL on error. */
long  seekdb_cuvs_graph_bytes(void* handle);
int   seekdb_cuvs_export_graph(void* handle, void* host_out, long capacity);
void* seekdb_cuvs_build_from_graph(const float* vectors, long n, long dim,
                                   const unsigned int* graph, long graph_degree);
#ifdef __cplusplus
}
#endif
#endif

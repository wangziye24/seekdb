/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here.
 *
 * Contest task (AMD GPU CAGRA index build): the baseline builds the CAGRA graph
 * on the GPU and persists a "GPU snapshot" of it.  The snapshot is on the build
 * critical path (CREATE VECTOR INDEX ... -> first query), so the bridge exposes
 * exactly the three pieces the seekdb adaptor needs to persist it with one pass
 * of large block transfers:
 *
 *   - seekdb_cuvs_build: upload the dataset once through pinned host staging and
 *     let the CAGRA index keep a device reference to it.  cuVS does not copy a
 *     device dataset again, and pinned + one big H2D beats the pageable copy of
 *     a host dataset.
 *   - seekdb_cuvs_graph_bytes / seekdb_cuvs_export_graph: copy only the kNN
 *     graph device->host through pinned staging.  The vectors are already on the
 *     host when the snapshot is written, so they never have to make a round trip
 *     through the GPU or through a temporary CAGRA file.
 *   - seekdb_cuvs_build_from_graph: rebuild the index from (graph, dataset) with
 *     cuvsCagraIndexFromArgs -- the expensive graph construction is not re-run,
 *     the graph and the vectors are uploaded straight into device buffers.
 *
 * The CAGRA build parameters stay at the values the baseline compares against,
 * so recall is unchanged.
 */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* Device buffers the index keeps a non-owning reference to. */
  void*  dev_dataset;
  size_t dev_dataset_bytes;
  void*  dev_graph;
  size_t dev_graph_bytes;
} sk_cuvs_index;

static void sk_fill_tensor(DLManagedTensor* t, void* data, int device_type, int dtype_code,
                           int bits, long* shape, int ndim)
{
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = (DLDeviceType)device_type;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = ndim;
  t->dl_tensor.dtype.code = (DLDataTypeCode)dtype_code;
  t->dl_tensor.dtype.bits = (unsigned char)bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = (int64_t*)shape;
  t->dl_tensor.strides = NULL;
  t->dl_tensor.byte_offset = 0;
  t->manager_ctx = NULL;
  t->deleter = NULL;
}

/* Same CAGRA parameters the frozen baseline uses (they define the recall). */
static void sk_init_index_params(cuvsCagraIndexParams_t ip)
{
  ip->build_algo = NN_DESCENT;
  ip->graph_degree = 128;
  ip->intermediate_graph_degree = 256;
  ip->nn_descent_niter = 100;
}

/* Pinned staging comes from cuVS's own host allocator (two-argument C API, so
 * no HIP flag-argument differences); callers fall back to plain copies when the
 * allocation is unavailable. */
static void* sk_pinned_alloc(size_t bytes)
{
  void* ptr = NULL;
  if (bytes == 0 || cuvsRMMHostAlloc(&ptr, bytes) != CUVS_SUCCESS) { return NULL; }
  return ptr;
}

static void sk_pinned_free(void* ptr, size_t bytes)
{
  if (ptr != NULL) { cuvsRMMHostFree(ptr, bytes); }
}

/* Host -> device through one pinned staging buffer; falls back to a plain copy. */
static int sk_upload(void* dst, const void* src, size_t bytes)
{
  int rc = 0;
  if (dst == NULL || src == NULL || bytes == 0) { return -1; }
  {
    void* pinned = sk_pinned_alloc(bytes);
    if (pinned != NULL) {
      memcpy(pinned, src, bytes);
      if (cudaMemcpy(dst, pinned, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { rc = -2; }
      sk_pinned_free(pinned, bytes);
    } else if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      rc = -3;
    }
  }
  if (rc != 0) {
    fprintf(stderr, "[seekdb_cuvs] H2D copy of %zu bytes failed: %s\n",
            bytes, cudaGetErrorString(cudaGetLastError()));
  }
  return rc;
}

/* Device -> host through one pinned staging buffer; falls back to a plain copy. */
static int sk_download(void* dst, const void* src, size_t bytes)
{
  int rc = 0;
  if (dst == NULL || src == NULL || bytes == 0) { return -1; }
  {
    void* pinned = sk_pinned_alloc(bytes);
    if (pinned != NULL) {
      if (cudaMemcpy(pinned, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = -2; }
      else { memcpy(dst, pinned, bytes); }
      sk_pinned_free(pinned, bytes);
    } else if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
      rc = -3;
    }
  }
  if (rc != 0) {
    fprintf(stderr, "[seekdb_cuvs] D2H copy of %zu bytes failed: %s\n",
            bytes, cudaGetErrorString(cudaGetLastError()));
  }
  return rc;
}

static void sk_free_index(sk_cuvs_index* h)
{
  if (h == NULL) { return; }
  if (h->idx != NULL) { cuvsCagraIndexDestroy(h->idx); h->idx = NULL; }
  if (h->dev_dataset != NULL) {
    cuvsRMMFree(h->res, h->dev_dataset, h->dev_dataset_bytes);
    h->dev_dataset = NULL; h->dev_dataset_bytes = 0;
  }
  if (h->dev_graph != NULL) {
    cuvsRMMFree(h->res, h->dev_graph, h->dev_graph_bytes);
    h->dev_graph = NULL; h->dev_graph_bytes = 0;
  }
  if (h->ip != NULL) { cuvsCagraIndexParamsDestroy(h->ip); h->ip = NULL; }
  if (h->res != 0) { cuvsResourcesDestroy(h->res); h->res = 0; }
  free(h);
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = NULL;
  size_t bytes = 0;
  long shape[2];
  DLManagedTensor dset;
  int built = 0;
  if (base == NULL || n <= 0 || dim <= 0) { return NULL; }
  bytes = (size_t)n * (size_t)dim * sizeof(float);
  h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (h == NULL) { return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { sk_free_index(h); return NULL; }
  sk_init_index_params(h->ip);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) {
    sk_free_index(h);
    fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
    return NULL;
  }

  /* Fast path: one pinned upload, then the index references that device copy.
   * The host-dataset path below keeps the old behaviour if pinned staging or the
   * RMM allocation is unavailable. */
  if (cuvsRMMAlloc(h->res, &h->dev_dataset, bytes) == CUVS_SUCCESS) {
    h->dev_dataset_bytes = bytes;
    if (sk_upload(h->dev_dataset, base, bytes) == 0) {
      shape[0] = n; shape[1] = dim;
      sk_fill_tensor(&dset, h->dev_dataset, kDLCUDA, kDLFloat, 32, shape, 2);
      if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) == CUVS_SUCCESS) { built = 1; }
    }
    if (!built) {
      cuvsRMMFree(h->res, h->dev_dataset, h->dev_dataset_bytes);
      h->dev_dataset = NULL;
      h->dev_dataset_bytes = 0;
      cuvsCagraIndexDestroy(h->idx);
      h->idx = NULL;
      if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) {
        sk_free_index(h);
        fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
        return NULL;
      }
    }
  }
  if (!built) {
    shape[0] = n; shape[1] = dim;
    sk_fill_tensor(&dset, (void*)base, kDLCPU, kDLFloat, 32, shape, 2);
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) {
      fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
      sk_free_index(h);
      return NULL;
    }
  }
  h->n = n;
  h->dim = dim;
  return h;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  int rc = 0;
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float)*nq*h->dim)  != CUVS_SUCCESS) { rc=10; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned)*nq*topk) != CUVS_SUCCESS) { rc=11; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float)*nq*topk)    != CUVS_SUCCESS) { rc=12; goto done; }
  if (cudaMemcpy(qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    sk_fill_tensor(&qt, qd, kDLCUDA, kDLFloat, 32, qs, 2);
    sk_fill_tensor(&nt, nd, kDLCUDA, kDLUInt, 32, nsh, 2);
    sk_fill_tensor(&dt, dd, kDLCUDA, kDLFloat, 32, dsh, 2);
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc=14; goto done; }
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
      sp->search_width = 4;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float)*nq*topk);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(unsigned)*nq*topk);
  if (qd) cuvsRMMFree(h->res, qd, sizeof(float)*nq*h->dim);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_free_index((sk_cuvs_index*)handle);
}

/* Fills *graph with the index's device graph view (row-major [n, degree]).
 * The view carries a heap-allocated shape array: call graph->deleter(graph). */
static int sk_graph_view(sk_cuvs_index* h, DLManagedTensor* graph, long* rows, long* degree)
{
  if (h == NULL || graph == NULL || rows == NULL || degree == NULL) { return -1; }
  memset(graph, 0, sizeof(*graph));
  if (cuvsCagraIndexGetGraph(h->idx, graph) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] cannot get the CAGRA graph: %s\n", cuvsGetLastErrorText());
    return -2;
  }
  if (graph->dl_tensor.ndim != 2 || graph->dl_tensor.shape == NULL ||
      graph->dl_tensor.data == NULL) {
    if (graph->deleter != NULL) { graph->deleter(graph); }
    return -3;
  }
  *rows = (long)graph->dl_tensor.shape[0];
  *degree = (long)graph->dl_tensor.shape[1];
  if (*rows <= 0 || *degree <= 0) {
    if (graph->deleter != NULL) { graph->deleter(graph); }
    return -4;
  }
  return 0;
}

long seekdb_cuvs_graph_bytes(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  DLManagedTensor graph;
  long rows = 0, degree = 0;
  if (sk_graph_view(h, &graph, &rows, &degree) != 0) { return -1; }
  if (graph.deleter != NULL) { graph.deleter(&graph); }
  return rows * degree * (long)sizeof(unsigned int);
}

int seekdb_cuvs_export_graph(void* handle, void* host_out, long capacity)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  DLManagedTensor graph;
  long rows = 0, degree = 0;
  size_t bytes = 0;
  int rc = 0;
  if (host_out == NULL || capacity <= 0) { return -1; }
  if (sk_graph_view(h, &graph, &rows, &degree) != 0) { return -2; }
  bytes = (size_t)rows * (size_t)degree * sizeof(unsigned int);
  if ((size_t)capacity < bytes) {
    if (graph.deleter != NULL) { graph.deleter(&graph); }
    return -3;
  }
  rc = sk_download(host_out, graph.dl_tensor.data, bytes);
  if (graph.deleter != NULL) { graph.deleter(&graph); }
  return rc == 0 ? 0 : -4;
}

void* seekdb_cuvs_build_from_graph(const float* vectors, long n, long dim,
                                   const unsigned int* graph, long graph_degree)
{
  sk_cuvs_index* h = NULL;
  size_t vbytes = 0, gbytes = 0;
  long dshape[2], gshape[2];
  DLManagedTensor dset, dgraph;
  int64_t actual_n = 0, actual_dim = 0;
  if (vectors == NULL || graph == NULL || n <= 0 || dim <= 0 || graph_degree <= 0) {
    return NULL;
  }
  vbytes = (size_t)n * (size_t)dim * sizeof(float);
  gbytes = (size_t)n * (size_t)graph_degree * sizeof(unsigned int);
  h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (h == NULL) { return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { sk_free_index(h); return NULL; }
  sk_init_index_params(h->ip);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { sk_free_index(h); return NULL; }

  if (cuvsRMMAlloc(h->res, &h->dev_dataset, vbytes) != CUVS_SUCCESS) { sk_free_index(h); return NULL; }
  h->dev_dataset_bytes = vbytes;
  if (cuvsRMMAlloc(h->res, &h->dev_graph, gbytes) != CUVS_SUCCESS) { sk_free_index(h); return NULL; }
  h->dev_graph_bytes = gbytes;
  if (sk_upload(h->dev_dataset, vectors, vbytes) != 0 ||
      sk_upload(h->dev_graph, graph, gbytes) != 0) {
    sk_free_index(h);
    return NULL;
  }
  dshape[0] = n; dshape[1] = dim;
  gshape[0] = n; gshape[1] = graph_degree;
  sk_fill_tensor(&dset, h->dev_dataset, kDLCUDA, kDLFloat, 32, dshape, 2);
  sk_fill_tensor(&dgraph, h->dev_graph, kDLCUDA, kDLUInt, 32, gshape, 2);
  /* The index only stores references to the two device buffers above; they are
   * owned (and freed) by this handle. */
  if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &dgraph, &dset, h->idx) != CUVS_SUCCESS ||
      cuvsCagraIndexGetSize(h->idx, &actual_n) != CUVS_SUCCESS ||
      cuvsCagraIndexGetDims(h->idx, &actual_dim) != CUVS_SUCCESS ||
      actual_n != (int64_t)n || actual_dim != (int64_t)dim) {
    fprintf(stderr, "[seekdb_cuvs] cannot rebuild the CAGRA index from the graph: "
            "%s (got %ld x %ld, expected %ld x %ld)\n",
            cuvsGetLastErrorText(), (long)actual_n, (long)actual_dim, n, dim);
    sk_free_index(h);
    return NULL;
  }
  h->n = n;
  h->dim = dim;
  return h;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}

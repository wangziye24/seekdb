# GPU CAGRA snapshot persistence

Status: **implemented**. A `lib=cuvs` index persists the CAGRA kNN graph as a
"GPU snapshot" in the vector-index snapshot table, and a restart reloads that
graph instead of building it again. This document describes the stream format and
why the write/reload path is shaped the way it is.

## Where it sits in the timing window

The build benchmark starts at `CREATE VECTOR INDEX` and ends when the first fixed
query returns a legal result. The snapshot write happens on that path, so the
serializer must not add a second device->host pass, a temporary CAGRA file, or a
duplicate copy of the dataset. The screenshot below is the shape the serializer
optimizes for:

```
INSERT/DDL:  rows --add_index--> host buffers (vectors + external ids)
             legacy fallback:  VSAG/hnswlib also builds a CPU graph (not used for lib=cuvs)
DDL:         build CAGRA on GPU over the buffered vectors
             serialize:  graph (device->host, pinned) + ids + vectors  -> snapshot stream
first query: serve top-k from the GPU index
restart:     deserialize: read ids + vectors + graph
             cuvsCagraIndexFromArgs(graph, vectors)  -> same graph, no graph construction
```

The build parameters (`NN_DESCENT`, `graph_degree=128`,
`intermediate_graph_degree=256`, `nn_descent_niter=100`) are unchanged, the graph
and the vectors are bit-identical to what was built, and the metric is still L2,
so Recall is the same as the graph that was built.

## Snapshot stream format (`OBCUVS03`)

Little-endian, straight after the 8-byte magic:

| field | type | meaning |
|---|---|---|
| `graph_bytes` | `uint64` | byte size of the kNN graph matrix; `0` = graph was not exported |
| `dim` | `int32` | vector dimension |
| `reserved` | `int32` | must be 0 |
| `n` | `int64` | number of vectors |
| ids | `int64[n]` | CAGRA row offset -> external vid |
| vectors | `float32[n*dim]` | row-major vectors (the same buffer the adaptor uses) |
| graph | `uint32[n*degree]` | row-major CAGRA graph, only when `graph_bytes > 0` |

`degree = graph_bytes / (n * 4)`. The vectors are stored once: the previous
format persisted a full CAGRA file (graph **and** a second copy of the vectors)
through a temporary file and then copied those bytes into the snapshot stream, so
the dataset was written twice and made a device->host round trip on the build
path.

## Write path

`ob_cuvs_serialize` (in `src/oblib/lib/vector/ob_vsag_adaptor.cpp`):

1. Builds the CAGRA index on a 32MB pthread if it is stale (same as before).
2. `seekdb_cuvs_graph_bytes` / `seekdb_cuvs_export_graph` copy only the graph
   device->host through one pinned staging buffer.
3. Writes magic/header, ids, the host-resident vectors and the graph into the
   snapshot stream (the stream itself is buffered and persisted as LOB chunks by
   `ObVectorSerializer`).

## Reload path

`ob_cuvs_deserialize`:

1. Reads ids, vectors (they become the adaptor's host buffer, so the CPU fallback
   keeps working) and the graph.
2. `seekdb_cuvs_build_from_graph` uploads graph and vectors through pinned
   staging and wraps them with `cuvsCagraIndexFromArgs` on a 32MB pthread -- the
   expensive graph construction is not re-run.
3. Falls back to a real GPU build from the vectors if the graph is absent
   (`graph_bytes == 0`) or the rebuild fails. That keeps a corrupt/older snapshot
   correct at the cost of build time; it never persists or serves a CPU result in
   place of the GPU index.

## Bridge side

`tools/hipvs_bridge/seekdb_cuvs_bridge.c`:

- Pinned staging uses cuVS's own host allocator (`cuvsRMMHostAlloc`, a two-argument
  C API) and falls back to plain pageable copies when it is unavailable.
- `seekdb_cuvs_build` uploads the dataset once through a pinned staging buffer and
  passes a device `DLManagedTensor` to `cuvsCagraBuild`. The index keeps a
  non-owning reference to that device buffer (cuVS does not copy a device dataset
  again) and the handle frees it after `cuvsCagraIndexDestroy`. If pinned staging
  or the RMM allocation is unavailable it falls back to the host-dataset path.
- `seekdb_cuvs_export_graph` uses `cuvsCagraIndexGetGraph` and a pinned D2H copy.
- `seekdb_cuvs_build_from_graph` uses `cuvsCagraIndexFromArgs`; the graph and
  dataset device buffers are owned by the handle because the index only stores
  references to them.

## Verification

- `tools/hipvs_bridge/build.sh` compiles the bridge against `/opt/hipvs`; the
  seekdb tree compiles with `-DOB_BUILD_CUVS=ON`.
- The stream format and the reload path can be exercised without a GPU with a
  host-side test double that implements the same 6-symbol C ABI (brute-force
  search, synthetic graph): build -> snapshot -> restart -> reload -> identical
  query results. The test double is a local testing asset, not part of the build
  or the runtime.
- On a GPU machine, `validation/run_ddl_persistence_smoke.sh` covers the real
  path: snapshot write on DDL, reload after a container restart (no `cuvs_build`
  trace, `cuvs_serve` present), deletes/inserts after the reload.

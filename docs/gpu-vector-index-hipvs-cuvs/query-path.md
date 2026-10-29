# Query path: reusable search scratch

Status: **implemented**. The frozen baseline allocated, for **every single-row
query**, three device buffers (`cuRMMAlloc` + `cuRMMFree`), one
`cuvsCagraSearchParams` object, and used pageable copies for the query upload and
the result download. The query benchmark (`nq=4096` single-row SQL queries, Top-10
and Top-100) pays that overhead 4096 times per dataset, so it is worth removing it.

## What changed (bridge only)

`tools/hipvs_bridge/seekdb_cuvs_bridge.c`:

- the index handle now owns the search scratch: query/neighbor/distance device
  buffers, one pinned host staging buffer and one `cuvsCagraSearchParams`;
- `seekdb_cuvs_search` grows that scratch on demand (Top-10 and Top-100 share it)
  and reuses it for every call instead of allocating/freeing per query;
- the search params object is created once and keeps the same
  `itopk_size = max(512, 4*topk)` / `search_width = 4` settings the baseline used;
- the query and the results go through pinned host staging when available, with the
  original pageable copy as fallback.

The actual search is still exactly one `cuvsCagraSearch` call on the
organizer-fixed hipVS/cuVS CAGRA implementation, with the same params and the same
FP32 inputs/outputs. Only the surrounding buffer management changed, which the
contest rules allow ("查询必须使用组织方固定版本的 hipVS/cuVS CAGRA 搜索实现；
允许调整其支持的参数并优化外围流程"). Results are identical by construction.

## Why it is safe

- The adaptor serialises searches on one index (`ob_cuvs_try_search` holds the
  registry mutex across the bridge call), so a per-handle scratch buffer is not
  shared concurrently; the batch entry point is serialised the same way.
- Buffers are only ever grown, and every copy is bounded by the sizes derived from
  `nq`, `dim` and `topk`, so changing `topk` between query sets (Top-10 vs Top-100)
  just grows the scratch once.
- Pinned staging is optional: when it cannot be allocated the code falls back to
  the baseline's pageable `cudaMemcpy`, so the first query cannot fail because of it.
- Device memory added per index: a few KB (one query row plus the top-k results),
  far below the 1.25x VRAM limit.

## Verification

- Compiles against the organizer-pinned hipVS headers
  (commit `f96a663d0a4441d6f5344bd3ef552b71ae7b370b`) with
  `gcc -std=gnu11 -O2 -Wall -Wextra -D__HIP_PLATFORM_AMD__` (0 warnings) and exports
  the same 7-symbol C ABI;
- Real-GPU verification on the AMD Radeon Cloud instance (`lib=cuvs` index build,
  query, restart/reload) is the same procedure as the snapshot validation and is
  documented in the contest submission notes.

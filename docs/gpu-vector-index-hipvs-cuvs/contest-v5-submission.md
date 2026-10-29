# AMD 赛题（600072）参赛提交说明 — V5 基线上的低风险优化

## 1. 提交版本

| 项 | 值 |
|---|---|
| 冻结基线 tag | `amd-cagra-contest-baseline-v5` |
| 冻结基线 commit | `634888d9b037153fa143d2c913ba4b25503e1d62` |
| 本分支 | `amd-cagra-submit-v5`（该 commit 的后代） |

本提交只动「GPU 快照持久化 IO」与「CPU→GPU 传输 / 查询外围缓冲管理」，
**CAGRA 建图参数、距离度量、FP32 语义、`lib=vsag` CPU 路径均未改动**。

## 2. 改动

1. **GPU 快照 IO（`src/oblib/lib/vector/ob_vsag_adaptor.cpp`）**
   快照格式 `OBCUVS03`：只持久化「kNN 图（device→host）+ 外部 id + host 上已有的
   向量」，不再经由 CAGRA 临时文件、不再把向量写两遍；重载用
   `cuvsCagraIndexFromArgs` 从「图+向量」装配索引，不重跑建图。详见
   [`snapshot-persistence.md`](snapshot-persistence.md)。
2. **GPU 建图 / 传输（`tools/hipvs_bridge/`）**
   数据集通过 pinned 暂存一次性 H2D，并让 CAGRA 索引直接引用该显存；导出图、
   从图重建索引都走 pinned 拷贝与 `cuvsCagraIndexFromArgs`。
3. **查询外围流程（`tools/hipvs_bridge/`）**
   每个索引的查询 scratch（query/top-k 显存缓冲 + pinned 暂存 + search params）
   只创建一次、逐次复用，不再每条查询做 3 次显存分配/释放与参数对象创建。
   搜索仍是一次固定版本 `cuvsCagraSearch` 调用，参数与输出与基线一致。
   详见 [`query-path.md`](query-path.md)。

## 3. 统一编译方式（与赛题要求一致）

```bash
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh <输出目录>/libseekdb_cuvs_bridge.so

cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOB_ENABLE_UNITY=ON -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=<输出目录>/libseekdb_cuvs_bridge.so
cmake --build build_release --target seekdb
```

## 4. 测试方法

与赛题规定一致：`LOAD DATA` 导入后，从
`CREATE VECTOR INDEX ... WITH (distance=l2, type=hnsw, lib=cuvs, m=16,
ef_construction=200, ef_search=64)` 开始计时，到第一条固定查询返回合法结果；
查询使用 `... ORDER BY l2_distance(v, <query>) APPROXIMATE LIMIT <k>`。
每轮使用全新 observer 与全新数据目录（无复用 CAGRA 图/索引文件/进程内缓存）。
仓库内验证资产：`validation/run_gpu_smoke.sh`、`validation/run_ddl_persistence_smoke.sh`、
`tools/hipvs_bridge/test_persistence.cpp`（bridge 级快照写/读）。

## 5. 预期效果

- 构建：窗口内省掉约 1～1.5GB 的搬运（重复向量写入、CAGRA 临时文件往返、
  pageable dataset D2H）与一次 pageable H2D；
- 查询：单行查询省掉 3 次显存分配/释放、参数对象创建与 2 次 pageable 拷贝；
- 图与向量与基线逐字节一致 → Recall 不变，查询结果逐字节一致。

## 6. 硬门槛自检

| # | 门槛 | 说明 |
|---|---|---|
| 1 | 能编译 | bridge 用固定 hipVS 头文件编译通过（0 warning）；seekdb 用上方固定命令编译通过 |
| 2 | 结果正确 | 搜索调用与输出语义不变；快照重载失败时回退为 GPU 重建，不用 CPU 结果顶替 |
| 3 | Recall | CAGRA 参数、图、向量均未变 → 与基线一致 |
| 4 | 真实 GPU | 建图与查询仍全部由 hipVS/cuVS CAGRA 完成 |
| 5 | 无回归 | `lib=vsag` 路径零改动；查询路径只是复用缓冲区 |
| 6 | 资源 | 每索引额外显存只有查询 scratch 的几 KB |

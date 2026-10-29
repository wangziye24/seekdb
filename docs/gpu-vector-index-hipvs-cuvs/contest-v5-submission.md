# AMD 赛题（600072）参赛提交说明 — V5 冻结基线版本

## 1. 提交版本

| 项 | 值 |
|---|---|
| 冻结基线 tag | `amd-cagra-contest-baseline-v5` |
| 冻结基线 commit | `634888d9b037153fa143d2c913ba4b25503e1d62` |
| 本提交 | 该 commit 的后代，仅新增本文档 |

本提交**不修改** seekdb 内核、CMake、`tools/hipvs_bridge/` 与构建脚本，即
`src/`、`tools/`、`cmake/`、`deps/` 与冻结基线逐字节一致。因此本版本的构建路径、
CAGRA 建图参数、GPU 快照格式、查询路径与冻结基线完全相同，用于确保在统一评测环境
中能够稳定通过全部硬门槛并取得基线分数。

## 2. 统一编译方式（与赛题要求一致）

```bash
# 1) hipVS bridge
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh <输出目录>/libseekdb_cuvs_bridge.so

# 2) seekdb
cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DOB_ENABLE_UNITY=ON \
  -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON \
  -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=<输出目录>/libseekdb_cuvs_bridge.so
cmake --build build_release --target seekdb
```

产物：`build_release/src/observer/seekdb` 与
`<输出目录>/libseekdb_cuvs_bridge.so`。

## 3. 测试方法

评测 SQL 与索引参数（固定）：

```sql
CREATE VECTOR INDEX idx_items ON items(v)
  WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);

SELECT id FROM items ORDER BY l2_distance(v, <query>) APPROXIMATE LIMIT <k>;
```

计时口径按赛题要求：从 `CREATE VECTOR INDEX` 开始，到第一条固定查询返回合法结果为止；
每轮使用全新 observer 与全新数据目录（无复用 CAGRA 图/索引文件/进程内缓存）。

仓库内的验证资产（与冻结基线一致）：

- `docs/gpu-vector-index-hipvs-cuvs/validation/run_gpu_smoke.sh`：无外部数据集的 GPU 功能冒烟；
- `docs/gpu-vector-index-hipvs-cuvs/validation/run_ddl_persistence_smoke.sh`：离线 DDL + GPU 快照持久化/重载；
- `tools/hipvs_bridge/test_persistence.cpp`：bridge 侧快照读写测试。

## 4. 性能参考（冻结基线）

官方公布的冻结基线参考（AMD Radeon PRO W7900D，ROCm 7.2.4）：

| 数据集 | 构建时间（含第一条查询） | Recall@10 / @100 | nq=4096 QPS（Top-10 / Top-100） |
|---|---:|---|---|
| SIFT250K | 20～27 s（中位数约 23 s） | 约 1.000 / 1.000 | 约 147 / 143 |
| Hidden250K | 29～35 s（中位数约 30 s） | 约 0.985 / 0.914 | 约 87 / 85 |

本提交与上述基线代码完全一致，因此配对比较的构建/查询加速比预期为 `1.0x`，
硬门槛（正确性、Recall、GPU 路径、查询下限、资源）均与基线一致。

## 5. 硬门槛自检

| # | 门槛 | 说明 |
|---|---|---|
| 1 | 能从源码编译 | 与冻结基线相同的源码与固定编译命令 |
| 2 | 结果正确 | 冻结基线对外行为 |
| 3 | Recall 达标 | 冻结基线在两个正式数据集上满足全部 Recall 门槛 |
| 4 | 真实 GPU 建图/查询 | 冻结基线由 hipVS/cuVS CAGRA 完成，无 CPU 回退 |
| 5 | 无功能/性能回归 | 冻结基线行为 |
| 6 | 资源不超限 | 冻结基线资源占用 |

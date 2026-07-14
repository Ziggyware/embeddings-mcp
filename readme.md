# embeddings-mcp

**Vector embedding and retrieval as a Model Context Protocol service.**

`embeddings-mcp` exposes a local, self-hosted vector database over stdio via the [Model Context Protocol](https://modelcontextprotocol.io). It embeds text with an ONNX BERT-family encoder, indexes the resulting vectors in an exact-search flat index, and persists state through a write-ahead log so that any MCP-capable client — agent, IDE, or orchestration layer — can add, search, compose, and delete passages without standing up an external vector database.

| | |
|---|---|
| **Author** | [Ziggyware](https://github.com/Ziggyware) |
| **Repository** | [Ziggyware/embeddings-mcp](https://github.com/Ziggyware/embeddings-mcp) |
| **Runtime** | .NET 8.0 |
| **Transport** | MCP over stdio |
| **Inference** | ONNX Runtime (CPU) |

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Architecture](#architecture)
3. [Embedding Pipeline](#embedding-pipeline)
4. [Index & Similarity Search](#index--similarity-search)
5. [Compositional Search](#compositional-search)
6. [Persistence Model](#persistence-model)
7. [Concurrency Model](#concurrency-model)
8. [Installation & Build](#installation--build)
9. [Configuration](#configuration)
10. [Tool Reference](#tool-reference)
11. [On-Disk Format](#on-disk-format)
12. [Performance Characteristics](#performance-characteristics)
13. [Known Gaps & Unverified Components](#known-gaps--unverified-components)
14. [License](#license)

---

## Design Philosophy

`embeddings-mcp` is built on a small set of deliberate constraints rather than a feature checklist:

- **Exact search, not approximate.** At the scale this service targets, brute-force cosine similarity over a flat index is both simpler and more correct than an ANN index (HNSW, IVF, etc.). There is no recall/latency tradeoff to reason about, because there is no approximation.
- **Vectors as unit vectors.** Every embedding — passage or query — is L2-normalized at creation time. This collapses cosine similarity to a single dot product per comparison, removing a division from the hot path without changing the ranking.
- **Durability without a database.** The index lives as two flat files (a binary vector store and a JSON metadata sidecar) plus a write-ahead log. Correctness does not depend on compaction ever running — the WAL is replayed on every load.
- **The protocol boundary is the tool surface.** Every mutation and query the service supports is a first-class MCP tool with an explicit, documented contract (idempotency, side effects, read/write classification) rather than a generic RPC passthrough.

---

## Architecture

```
                         ┌─────────────────────────────┐
                         │   MCP Client (stdio)         │
                         └──────────────┬────────────────┘
                                        │  JSON-RPC over stdio
                         ┌──────────────▼────────────────┐
                         │        EmbedTools               │
                         │  (MCP tool surface / contracts) │
                         └───────┬─────────────┬──────────┘
                                 │             │
                    ┌────────────▼──┐   ┌───────▼────────────┐
                    │  OnnxEmbedder  │   │     FlatIndex        │
                    │  BERT encoder  │   │  in-memory vectors,   │
                    │  mean-pool +   │   │  RWLock-guarded,      │
                    │  L2 normalize  │   │  exact top-K search   │
                    └────────────────┘   └───────┬────────────┘
                                                 │
                                         ┌────────▼────────────┐
                                         │     IndexStore        │
                                         │  base snapshot (.vec/ │
                                         │  .meta.json) + WAL     │
                                         └────────────────────────┘
```

Four components, each with a single responsibility:

- **`OnnxEmbedder`** — owns the ONNX Runtime `InferenceSession` and `BertTokenizer`, and is the only component that produces vectors.
- **`FlatIndex`** — an in-memory `Dictionary<string, Entry>` guarded by a `ReaderWriterLockSlim`, offering add/remove/get/search over the current working set.
- **`IndexStore`** — persistence. Writes a base snapshot and an append-only WAL; reconstructs the working set on startup by replaying base + WAL.
- **`EmbedTools`** — the MCP tool surface. Every public tool method embeds (if needed), mutates or queries `FlatIndex`, and persists via `IndexStore` in the same call.

---

## Embedding Pipeline

Text enters the system through a BERT-family ONNX encoder (`input_ids`, `attention_mask`, `token_type_ids` → `last_hidden_state`). The service does not use the `[CLS]` token representation; it computes **attention-mask-weighted mean pooling** over the token axis:

```
v_d = ( Σ_t mask_t · h_{t,d} ) / ( Σ_t mask_t )      for each dimension d ∈ [0, 768)
```

where `h_t` is the hidden state at token position `t` and `mask_t ∈ {0, 1}` excludes padding tokens from the sum. The pooled vector is then L2-normalized:

```
v̂ = v / ‖v‖₂
```

Normalization is performed in double-precision accumulation (`sumSq` as `double`) before the final cast back to `float`, avoiding catastrophic cancellation across 768-dimensional sums on longer sequences.

**Asymmetric query/passage encoding.** Passages are embedded as-is. Queries are prefixed with the instruction string `"Represent this sentence for searching relevant passages: "` before tokenization. This is the standard asymmetric bi-encoder convention for retrieval-tuned BERT variants — the query and passage towers see different input distributions by design, and the prefix is what steers the query embedding toward the passage embedding space at inference time. Batch passage embedding (`EmbedPassagesBatch`) applies the identical pooling and normalization math per row, vectorized with `System.Numerics.Vector<float>` SIMD lanes over the pooling accumulation and division.

The model and tokenizer vocabulary are supplied at process start as file paths — the service ships no bundled weights and makes no assumption about which specific checkpoint is loaded, only that it is a standard BERT-architecture ONNX export exposing a `last_hidden_state` output and a compatible WordPiece vocabulary.

---

## Index & Similarity Search

Because every stored and query vector is unit-normalized, cosine similarity reduces algebraically to a dot product:

```
cos(a, b) = (a · b) / (‖a‖‖b‖) = a · b       when ‖a‖ = ‖b‖ = 1
```

`FlatIndex.Search` computes this dot product (via `System.Numerics.Tensors.TensorPrimitives.Dot`, itself hardware-vectorized) against every entry in the working set and retains the top-K by score using a bounded `PriorityQueue<Hit, float>` min-heap of size K:

- Below K entries seen: insert unconditionally.
- At capacity: compare the incoming score against the heap's minimum; replace only if larger.

This bounds the ranking step to `O(n log K)` comparisons rather than an `O(n log n)` full sort, on top of the `O(n·d)` dot-product cost that dominates for any realistic `d = 768`.

---

## Compositional Search

`embed_compose_search` performs vector arithmetic directly in embedding space — the same operation that produces results like *king − man + woman ≈ queen* in word-vector models, generalized to sentence embeddings and to arbitrary signed weights:

```
q = Σ_i  w_i · v_i          (v_i from an existing indexed id, or freshly embedded text)
q̂ = q / ‖q‖₂
```

Terms may carry **negative** weight, which subtracts a concept from the composed query rather than adding it — e.g. weighting an id at `-1.0` pushes the search away from that passage's semantic content. By default (`excludeSourceIds = true`), any id used as a term is excluded from its own results, since a term vector is trivially near-collinear with itself and would otherwise dominate the top-K.

---

## Persistence Model

State durability is split into two layers that are reconciled on load, not on write:

1. **Base snapshot** — `index.vec` (raw `float32` vectors, fixed-width, magic-tagged, dimension-tagged) and `index.meta.json` (id/text/source per entry, positionally aligned to the vector file). Produced by `IndexStore.Save()`.
2. **Write-ahead log** — `index.wal`, an append-only binary log of `Upsert` and `Delete` operations, written synchronously on every mutating tool call.

On startup, `IndexStore.Load` reconstructs the working set by reading the base snapshot into a dictionary keyed by id, then replaying the WAL in order — later WAL records overwrite earlier snapshot entries. **The WAL is not an optional acceleration structure; it is required for correctness.** A snapshot alone reflects only the state as of the last `Compact()` or `Save()`; the WAL carries everything since. `embed_compact` folds the WAL into a fresh snapshot and truncates it — purely a startup-time and disk-usage optimization, never a correctness requirement, since `Load()` transparently replays base + WAL regardless of WAL length.

All snapshot writes are atomic: content is written to a temp file in the same directory and then `File.Move`'d over the destination, so a crash mid-write cannot leave a torn `index.vec` or `index.meta.json`.

Dimension mismatches are treated as a hard startup failure rather than silently truncated or padded: if `index.vec` was built against a different embedding dimension than the model currently configured, `Load()` throws rather than producing vectors that are structurally incompatible with the loaded model.

---

## Concurrency Model

`FlatIndex` uses a `ReaderWriterLockSlim` around the backing dictionary: any number of concurrent reads (`Search`, `Get`, `Count`) proceed without contention, while writes (`Add`, `Remove`, `LoadSnapshot`) take an exclusive lock. Search operates on a point-in-time snapshot (`ById.Values.ToArray()`) taken under the read lock, so ranking never observes a torn write.

Two search strategies are selected by working-set size, with a threshold of **2,000 entries**:

- **Sequential** (`< 2000`): single-threaded scan into one bounded heap. Below this size, thread dispatch overhead exceeds the cost of the scan itself.
- **Parallel** (`≥ 2000`): `Parallel.For` partitions the entry array across worker threads, each maintaining its own local bounded heap (`localInit`/`localFinally` avoids any shared mutable state during the scan), followed by a sequential merge of the per-partition heaps into one final top-K heap.

The write-ahead log itself is serialized through a dedicated `lock (WalLock)` distinct from the index's `ReaderWriterLockSlim`, since WAL appends are a filesystem operation with its own ordering requirement (each record must be fully written before the next begins) independent of in-memory index state.

---

## Installation & Build

**Prerequisites:** .NET 8.0 SDK, an ONNX-exported BERT-family sentence encoder, and its matching WordPiece vocabulary file.

```bash
git clone https://github.com/Ziggyware/embeddings-mcp.git
cd embeddings-mcp
dotnet build -c Release
```

The project targets `net8.0`, builds as an executable (`OutputType=Exe`), and produces the assembly `embed-retrieval.exe` / `embed-retrieval`. If a `model/` directory exists alongside the project at build time, its contents are copied to the output directory automatically (`CopyToOutputDirectory=PreserveNewest`); this is a convenience for co-locating weights, not a requirement — model and vocabulary paths are always supplied explicitly at runtime.

---

## Configuration

The service takes exactly three positional arguments — there is no config file, environment-variable fallback, or interactive setup:

```bash
embed-retrieval <EMBED_MODEL_PATH> <EMBED_VOCAB_PATH> <EMBEDDING_ROOT>
```

| Argument | Description |
|---|---|
| `EMBED_MODEL_PATH` | Path to the ONNX-exported encoder (`.onnx`). Must expose `input_ids`, `attention_mask`, `token_type_ids` inputs and a `last_hidden_state` output. |
| `EMBED_VOCAB_PATH` | Path to the WordPiece vocabulary consumed by `BertTokenizer`. |
| `EMBEDDING_ROOT` | Directory where `index.vec`, `index.meta.json`, and `index.wal` are read from and written to. |

Missing or malformed arguments fail fast: fewer than three arguments throws before the host starts; a missing model or vocab file throws `FileNotFoundException` from `OnnxEmbedder.Init`; a missing embedding root throws `McpException` from `IndexStore.Load`.

As an MCP stdio server, `embed-retrieval` is typically launched by an MCP-aware client (agent runtime, IDE extension, orchestrator) which supplies these three arguments as part of its server-launch configuration, rather than invoked interactively.

---

## Tool Reference

All tools are exposed under the `embed-retrieval` MCP server. Read-only tools (`embed_search`, `embed_compose_search`) take no exclusive lock and never touch the WAL; mutating tools (`embed_add`, `embed_add_batch`, `embed_delete`, `embed_compact`) auto-persist within the same call.

| Tool | R/W | Destructive | Idempotent | Description |
|---|:---:|:---:|:---:|---|
| `embed_add` | write | no | yes | Embed one passage and upsert it by id. Re-adding an existing id overwrites, without versioning. |
| `embed_add_batch` | write | no | yes | Embed and upsert many passages in a single batched inference call — preferred over repeated `embed_add` for bulk indexing. |
| `embed_search` | read | no | yes | Embed the query with the query-side instruction prefix, return the top-K nearest passages by cosine similarity. |
| `embed_compose_search` | read | no | yes | Build a synthetic query as a weighted sum of existing ids and/or fresh text, renormalize, and search. Supports negative weights. |
| `embed_delete` | write | yes | yes | Remove a passage by id. Deleting a nonexistent id returns `removed: false` rather than throwing. |
| `embed_compact` | write | no | yes | Fold the WAL into the base snapshot and truncate it. Never required for correctness. |

### `embed_add`
```
Add(id: string, text: string, source?: string) → { added, total }
```

### `embed_add_batch`
```
AddBatch(itemsJson: [{ id, text, source? }]) → { addedCount, total }
```

### `embed_search`
```
Search(query: string, topK: int = 5) → { hits: [{ Id, Score, Text, Source }] }
```

### `embed_compose_search`
```
ComposeSearch(
  termsJson: [{ id?, text?, weight }],   // exactly one of id/text per term
  topK: int = 5,
  excludeSourceIds: bool = true
) → { hits: [{ Id, Score, Text, Source }], composedFrom: int }
```

### `embed_delete`
```
Delete(id: string) → { removed, total }
```

### `embed_compact`
```
Compact() → { compacted, total }
```

---

## On-Disk Format

**`index.vec`** (little-endian binary):

| Field | Type | Notes |
|---|---|---|
| Magic | `int32` | `0x45424958` ("EBIX") — validated on load; mismatch throws `InvalidDataException`. |
| Dim | `int32` | Embedding dimensionality. Must equal the currently loaded model's `Dim`; mismatch throws with an explicit rebuild instruction. |
| Count | `int32` | Number of entries. |
| Vectors | `float32[Count][Dim]` | Row-major, positionally aligned with `index.meta.json`. |

**`index.meta.json`** — a JSON array of `{ Id, Text, Source }`, positionally aligned to the vector rows above.

**`index.wal`** — an append-only sequence of variable-length records:

| Opcode | Layout |
|---|---|
| `1` (Upsert) | `byte op` · length-prefixed UTF-8 `id` · `float32[Dim]` vector · length-prefixed UTF-8 `text` (or `-1` for null) · length-prefixed UTF-8 `source` (or `-1` for null) |
| `2` (Delete) | `byte op` · length-prefixed UTF-8 `id` |

An unrecognized opcode encountered during replay throws `InvalidDataException`, treating a corrupt or truncated WAL as a hard failure rather than silently skipping the remainder.

---

## Performance Characteristics

- **Search cost:** `O(n·d)` for the dot-product scan (`n` = index size, `d` = 768), plus `O(n log K)` for top-K maintenance via bounded heap. There is no index-build amortization to reason about — every search is exact and reflects the current state precisely, with no rebuild or staleness window.
- **Parallel crossover at n = 2000:** below this, single-thread scan; at or above, `Parallel.For` with per-partition local heaps and a final sequential merge, trading a fixed dispatch/merge overhead for near-linear core scaling on the dominant `O(n·d)` term.
- **Batch embedding throughput:** `EmbedPassagesBatch` pads all sequences in a batch to the batch's own max token length (not a fixed `MaxSeqLen`), and vectorizes the mean-pooling accumulation with `System.Numerics.Vector<float>`, amortizing ONNX Runtime session overhead across the batch rather than per item.
- **Memory:** the entire working set is memory-resident (`Dictionary<string, Entry>`); there is no paging or memory-mapped fallback. Capacity planning is `n × (d × 4 bytes + text + overhead)`.

---

## Known Gaps & Unverified Components

In the interest of not asserting more than the source establishes:

- **`Program.cs` is a reconstructed stdio bootstrap**, not verified against a reference `Program.cs` from the original project — the source material available for this document included the tool, index, storage, and embedder files but not the actual host entry point. It mirrors the standard `ModelContextProtocol` C# SDK stdio pattern (`AddMcpServer().WithStdioServerTransport().WithToolsFromAssembly()`), but the server name/version string and the tool-registration mechanism (explicit registration vs. assembly scan) should be reconciled against the real entry point before this is treated as authoritative.
- **No embedded model ships with the repository.** The service is architecture-compatible with any ONNX BERT-family encoder exposing the expected input/output tensor names and a 768-dimension `last_hidden_state`; it does not bundle or pin a specific checkpoint.
- **No authentication or transport-level access control** is implemented — the trust boundary is whatever process spawns the stdio server, consistent with the MCP stdio transport model generally.

---

## License

See the repository's [LICENSE](https://github.com/Ziggyware/embeddings-mcp) file for terms.

---

<p align="center"><sub>embeddings-mcp — built and maintained by <a href="https://github.com/Ziggyware">Ziggyware</a></sub></p>

// Exact brute-force cosine search. No HNSW: scoped to <10k vectors per stated design
// constraint. [rule 5 failure mode]: this file does NOT self-limit or warn if vector
// count grows past that scope — it will keep working, just with linearly growing
// per-query latency (O(n·d)), silently. If you later need >10k-100k vectors, this is
// the file to replace with an ANN structure; nothing else in the design depends on
// FlatIndex internals beyond the public methods below.
internal static class FlatIndex
{
    public sealed record Entry(string Id, float[] Vector, string Text, string Source);

    static readonly Dictionary<string, Entry> ById = new();
    static readonly ReaderWriterLockSlim Gate = new(); // mirrors Workspace.Gate's role:
        // writers (Add/Remove) exclusive, readers (Search) concurrent among themselves.

    public static void Add(string id, float[] vector, string text, string source)
    {
        if (vector.Length != OnnxEmbedder.Dim)
            throw new ArgumentException(
                $"vector dim {vector.Length} != expected {OnnxEmbedder.Dim}");
        Gate.EnterWriteLock();
        try { ById[id] = new Entry(id, vector, text, source); }
        finally { Gate.ExitWriteLock(); }
    }

    public static bool Remove(string id)
    {
        Gate.EnterWriteLock();
        try { return ById.Remove(id); }
        finally { Gate.ExitWriteLock(); }
    }

    public static int Count { get { Gate.EnterReadLock(); try { return ById.Count; } finally { Gate.ExitReadLock(); } } }

    public sealed record Hit(string Id, float Score, string Text, string Source);

    // topK via full scan + sort. [derived]: O(n log n) dominated by the sort, not the
    // scan; at n<10k this is sub-millisecond on typical hardware [heuristic, confidence:
    // low — not benchmarked in this environment, mark [unverified — check before relying
    // on this] if latency is load-bearing for your use case].
    public static Hit[] Search(float[] queryVec, int topK)
    {
        if (queryVec.Length != OnnxEmbedder.Dim)
            throw new ArgumentException(
                $"query vector dim {queryVec.Length} != expected {OnnxEmbedder.Dim}");
        Gate.EnterReadLock();
        try
        {
            return ById.Values
                .Select(e => new Hit(e.Id, Dot(queryVec, e.Vector), e.Text, e.Source))
                .OrderByDescending(h => h.Score)
                .Take(topK)
                .ToArray();
        }
        finally { Gate.ExitReadLock(); }
    }

    static float Dot(float[] a, float[] b)
    {
        // Both inputs assumed pre-normalized (embedder guarantees this) — no defensive
        // renormalization here. [rule 3 assumption surfacing]: if you ever feed this
        // vectors from a source OTHER than OnnxEmbedder, this silently returns a
        // meaningless "cosine" for unnormalized inputs. No check enforces normalization.
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    // Snapshot for persistence — IndexStore reads this, doesn't touch Dictionary directly.
    public static Entry[] Snapshot()
    {
        Gate.EnterReadLock();
        try { return ById.Values.ToArray(); }
        finally { Gate.ExitReadLock(); }
    }

    public static void LoadSnapshot(IEnumerable<Entry> entries)
    {
        Gate.EnterWriteLock();
        try { ById.Clear(); foreach (var e in entries) ById[e.Id] = e; }
        finally { Gate.ExitWriteLock(); }
    }
}
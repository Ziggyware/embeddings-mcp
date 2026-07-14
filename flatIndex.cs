using System.Numerics.Tensors;
using System.Collections.Concurrent;

internal static class FlatIndex
{
    public sealed record Entry(string Id, float[] Vector, string Text, string Source);
    public sealed record Hit(string Id, float Score, string Text, string Source);

    static readonly Dictionary<string, Entry> ById = new();
    static readonly ReaderWriterLockSlim Gate = new();

    const int ParallelThreshold = 2000;

    public static void Add(string id, float[] vector, string text, string source)
    {
        if (vector.Length != OnnxEmbedder.Dim)
            throw new ArgumentException($"vector dim {vector.Length} != expected {OnnxEmbedder.Dim}");
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

    public static Entry Get(string id)
    {
        Gate.EnterReadLock();
        try { return ById.TryGetValue(id, out var e) ? e : null; }
        finally { Gate.ExitReadLock(); }
    }

    public static int Count { get { Gate.EnterReadLock(); try { return ById.Count; } finally { Gate.ExitReadLock(); } } }

    public static Hit[] Search(float[] queryVec, int topK)
    {
        if (queryVec.Length != OnnxEmbedder.Dim)
            throw new ArgumentException($"query vector dim {queryVec.Length} != expected {OnnxEmbedder.Dim}");
        if (topK <= 0) throw new ArgumentException("topK must be > 0");

        Entry[] snapshot;
        Gate.EnterReadLock();
        try { snapshot = ById.Values.ToArray(); }
        finally { Gate.ExitReadLock(); }

        if (snapshot.Length == 0) return Array.Empty<Hit>();
        return snapshot.Length >= ParallelThreshold
            ? SearchParallel(queryVec, snapshot, topK)
            : SearchSequential(queryVec, snapshot, topK);
    }

    static Hit[] SearchSequential(float[] q, Entry[] entries, int topK)
    {
        var heap = new PriorityQueue<Hit, float>(topK + 1);
        foreach (var e in entries)
            Consider(heap, new Hit(e.Id, Dot(q, e.Vector), e.Text, e.Source), topK);
        return Drain(heap);
    }

    static Hit[] SearchParallel(float[] q, Entry[] entries, int topK)
    {
        var partitionHeaps = new ConcurrentBag<PriorityQueue<Hit, float>>();

        Parallel.For(0, entries.Length,
            localInit: () => new PriorityQueue<Hit, float>(topK + 1),
            body: (i, _, localHeap) =>
            {
                var e = entries[i];
                Consider(localHeap, new Hit(e.Id, Dot(q, e.Vector), e.Text, e.Source), topK);
                return localHeap;
            },
            localFinally: partitionHeaps.Add);

        var merged = new PriorityQueue<Hit, float>(topK + 1);
        foreach (var h in partitionHeaps)
            while (h.Count > 0) Consider(merged, h.Dequeue(), topK);
        return Drain(merged);
    }

    static void Consider(PriorityQueue<Hit, float> heap, Hit h, int topK)
    {
        if (heap.Count < topK) heap.Enqueue(h, h.Score);
        else if (h.Score > heap.Peek().Score) { heap.Dequeue(); heap.Enqueue(h, h.Score); }
    }

    static Hit[] Drain(PriorityQueue<Hit, float> heap)
    {
        var result = new Hit[heap.Count];
        for (int i = result.Length - 1; i >= 0; i--) result[i] = heap.Dequeue();
        return result;
    }

    static float Dot(float[] a, float[] b) =>
        TensorPrimitives.Dot((ReadOnlySpan<float>)a, b);

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
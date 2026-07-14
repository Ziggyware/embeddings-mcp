using ModelContextProtocol;
using ModelContextProtocol.Server;
using System.Text.Json;

[McpServerToolType]
internal static class EmbedTools
{
    static readonly JsonSerializerOptions J = new() { WriteIndented = false };
    static readonly JsonSerializerOptions JIn = new() { PropertyNameCaseInsensitive = true };
    static string Ser(object o) => JsonSerializer.Serialize(o, J);

    static string Guarded(Func<string> body)
    {
        try { return body(); }
        catch (McpException) { throw; }
        catch (Exception e) { throw new McpException(e.Message); }
    }

    public sealed record AddItem(string id, string text, string source);
    public sealed record ComposeTerm(string id, string text, double weight);

    [McpServerTool(Name = "embed_add", Title = "Add Passage", ReadOnly = false, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Embed one text passage and add it to the index under the given id. Idempotent: " +
        "re-adding an existing id overwrites it (no duplicate/versioning). Auto-persists " +
        "after the call. For many items at once, use AddBatch instead.")]
    public static string Add(string id, string text, string source = null) => Guarded(() =>
    {
        var vec = OnnxEmbedder.EmbedPassage(text);
        var entry = new FlatIndex.Entry(id, vec, text, source);
        FlatIndex.Add(id, vec, text, source);
        IndexStore.AppendUpsert(entry);
        return Ser(new { added = id, total = FlatIndex.Count });
    });

    [McpServerTool(Name = "embed_add_batch", Title = "Add Passages (Batch)", ReadOnly = false, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Embed and add multiple passages in one call. itemsJson: array of {id, text, " +
        "source?}. Prefer this over repeated Add calls when indexing more than a " +
        "handful of items.")]
    public static string AddBatch(string itemsJson) => Guarded(() =>
    {
        var items = JsonSerializer.Deserialize<AddItem[]>(itemsJson, JIn)
            ?? throw new McpException("itemsJson did not parse");
        var texts = items.Select(i => i.text).ToArray();
        var vecs = OnnxEmbedder.EmbedPassagesBatch(texts);
        var entries = new FlatIndex.Entry[items.Length];
        for (int i = 0; i < items.Length; i++)
        {
            entries[i] = new FlatIndex.Entry(items[i].id, vecs[i], items[i].text, items[i].source);
            FlatIndex.Add(items[i].id, vecs[i], items[i].text, items[i].source);
        }
        IndexStore.AppendUpsertBatch(entries);
        return Ser(new { addedCount = items.Length, total = FlatIndex.Count });
    });

    [McpServerTool(Name = "embed_search", Title = "Search Index", ReadOnly = true, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Embed the query with the query-side instruction prefix and return the topK " +
        "nearest passages by cosine similarity. Exact search at the current design scale.")]
    public static string Search(string query, int topK = 5) => Guarded(() =>
    {
        if (topK <= 0) throw new McpException("topK must be > 0");
        var qvec = OnnxEmbedder.EmbedQuery(query);
        var hits = FlatIndex.Search(qvec, topK);
        return Ser(new { hits = hits.Select(h => new { h.Id, h.Score, h.Text, h.Source }) });
    });

    [McpServerTool(Name = "embed_compose_search", Title = "Compositional Search", ReadOnly = true, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Build a synthetic query as a weighted sum of existing indexed ids and/or fresh " +
        "text, renormalize, then top-K search. termsJson: array of {id?, text?, weight} — " +
        "exactly one of id/text per term, weight can be negative to subtract a concept. " +
        "excludeSourceIds (default true) drops any term-id from its own results.")]
    public static string ComposeSearch(string termsJson, int topK = 5, bool excludeSourceIds = true) => Guarded(() =>
    {
        var terms = JsonSerializer.Deserialize<ComposeTerm[]>(termsJson, JIn)
            ?? throw new McpException("termsJson did not parse");
        if (terms.Length == 0) throw new McpException("at least one term required");

        var acc = new float[OnnxEmbedder.Dim];
        var excludeIds = new HashSet<string>();
        foreach (var t in terms)
        {
            float[] v;
            if (!string.IsNullOrEmpty(t.id))
            {
                var entry = FlatIndex.Get(t.id) ?? throw new McpException($"unknown id: {t.id}");
                v = entry.Vector;
                excludeIds.Add(t.id);
            }
            else if (!string.IsNullOrEmpty(t.text))
            {
                v = OnnxEmbedder.EmbedQuery(t.text);
            }
            else throw new McpException("each term needs id or text, not neither/both-empty");

            var w = (float)t.weight;
            for (int d = 0; d < acc.Length; d++) acc[d] += w * v[d];
        }
        OnnxEmbedder.Normalize(acc);

        var fetch = topK + (excludeSourceIds ? excludeIds.Count : 0);
        var hits = FlatIndex.Search(acc, fetch);
        if (excludeSourceIds)
            hits = hits.Where(h => !excludeIds.Contains(h.Id)).Take(topK).ToArray();

        return Ser(new { hits = hits.Select(h => new { h.Id, h.Score, h.Text, h.Source }), composedFrom = terms.Length });
    });

    [McpServerTool(Name = "embed_delete", Title = "Delete Passage", ReadOnly = false, Destructive = true, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Remove a passage by id. Idempotent: deleting a nonexistent id returns removed:false " +
        "rather than throwing. Auto-persists after the call.")]
    public static string Delete(string id) => Guarded(() =>
    {
        var removed = FlatIndex.Remove(id);
        if (removed) IndexStore.AppendDelete(id);
        return Ser(new { removed, total = FlatIndex.Count });
    });

    [McpServerTool(Name = "embed_compact", Title = "Compact Index", ReadOnly = false, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Fold the write-ahead log into the base index files and truncate the WAL. Not " +
        "required for correctness — Load() replays base+WAL transparently — only for " +
        "keeping startup Load() time and disk usage bounded after many adds/deletes.")]
    public static string Compact() => Guarded(() =>
    {
        IndexStore.Compact();
        return Ser(new { compacted = true, total = FlatIndex.Count });
    });
}
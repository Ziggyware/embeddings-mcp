
using ModelContextProtocol;
using ModelContextProtocol.Server;
using System.Text.Json;

[McpServerToolType]
internal static class EmbedTools
{
    static readonly JsonSerializerOptions J = new() { WriteIndented = false };
    static readonly JsonSerializerOptions JIn = new() { PropertyNameCaseInsensitive = true };
    static string Ser(object o) => JsonSerializer.Serialize(o, J);

    // Single exception boundary, mirrors reference project's FsTools.Guarded — reimplemented
    // standalone since this is a separate project (confirmed: new standalone server).
    static string Guarded(Func<string> body)
    {
        try { return body(); }
        catch (McpException) { throw; }
        catch (Exception e) { throw new McpException(e.Message); }
    }

    public sealed record AddItem(string id, string text, string source);

    [McpServerTool(Name = "embed_add", Title = "Add Passage", ReadOnly = false, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Embed one text passage and add it to the index under the given id. Idempotent: " +
        "re-adding an existing id overwrites it (no duplicate/versioning). Auto-persists " +
        "after the call. For many items at once, use AddBatch instead — one Save() per " +
        "batch, not one per item.")]
    public static string Add(string id, string text, string source = null) => Guarded(() =>
    {
        var vec = OnnxEmbedder.EmbedPassage(text);
        FlatIndex.Add(id, vec, text, source);
        IndexStore.Save();
        return Ser(new { added = id, total = FlatIndex.Count });
    });

    [McpServerTool(Name = "embed_add_batch", Title = "Add Passages (Batch)", ReadOnly = false, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Embed and add multiple passages in one call, one Save() at the end rather than " +
        "per-item. itemsJson: array of {id, text, source?}. Prefer this over repeated " +
        "Add calls when indexing more than a handful of items — avoids redundant full-" +
        "index rewrites.")]
    public static string AddBatch(string itemsJson) => Guarded(() =>
    {
        var items = JsonSerializer.Deserialize<AddItem[]>(itemsJson, JIn)
            ?? throw new McpException("itemsJson did not parse");
        // Uses EmbedPassagesBatch, which — per Part 1's disclosed note — is a per-item
        // loop under the hood, not a padded-tensor batch. Batching here saves Save()
        // calls, not inference time. [restating that caveat, not re-deriving it]
        var texts = items.Select(i => i.text).ToArray();
        var vecs = OnnxEmbedder.EmbedPassagesBatch(texts);
        for (int i = 0; i < items.Length; i++)
            FlatIndex.Add(items[i].id, vecs[i], items[i].text, items[i].source);
        IndexStore.Save();
        return Ser(new { addedCount = items.Length, total = FlatIndex.Count });
    });

    [McpServerTool(Name = "embed_search", Title = "Search Index", ReadOnly = true, Destructive = false, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Embed the query with the query-side instruction prefix (asymmetric from passage " +
        "embedding — see Add) and return the topK nearest passages by cosine similarity. " +
        "Exact search, not approximate, at the current <10k-vector design scale.")]
    public static string Search(string query, int topK = 5) => Guarded(() =>
    {
        if (topK <= 0) throw new McpException("topK must be > 0");
        var qvec = OnnxEmbedder.EmbedQuery(query);
        var hits = FlatIndex.Search(qvec, topK);
        return Ser(new { hits = hits.Select(h => new { h.Id, h.Score, h.Text, h.Source }) });
    });

    [McpServerTool(Name = "embed_delete", Title = "Delete Passage", ReadOnly = false, Destructive = true, Idempotent = true, OpenWorld = false),
     System.ComponentModel.Description(
        "Remove a passage by id. Idempotent: deleting a nonexistent id returns removed:false " +
        "rather than throwing. Auto-persists after the call.")]
    public static string Delete(string id) => Guarded(() =>
    {
        var removed = FlatIndex.Remove(id);
        if (removed) IndexStore.Save();
        return Ser(new { removed, total = FlatIndex.Count });
    });
}
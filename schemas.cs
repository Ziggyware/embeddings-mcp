
// ── Add, AddBatch (EmbedTools) ──
public sealed record EmbedAddResult(string added, int total);
public sealed record EmbedAddBatchResult(int addedCount, int total);

// ── Search (EmbedTools) ──
public sealed record SearchHit(string Id, float Score, string Text, string Source);
public sealed record SearchResult(SearchHit[] hits);

// ── Delete (EmbedTools) ──
public sealed record EmbedDeleteResult(bool removed, int total);

// ── Stats (EmbedTools) ──
public sealed record EmbedStatsResult(int count, int dim);

// ── Save, Reload (EmbedTools) ──
public sealed record EmbedSaveResult(bool saved);
public sealed record EmbedReloadResult(bool reloaded, int total);
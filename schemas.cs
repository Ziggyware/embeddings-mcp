public sealed record EmbedAddResult(string added, int total);
public sealed record EmbedAddBatchResult(int addedCount, int total);

public sealed record SearchHit(string Id, float Score, string Text, string Source);
public sealed record SearchResult(SearchHit[] hits);
public sealed record ComposeSearchResult(SearchHit[] hits, int composedFrom);

public sealed record EmbedDeleteResult(bool removed, int total);

public sealed record EmbedStatsResult(int count, int dim);

public sealed record EmbedSaveResult(bool saved);
public sealed record EmbedReloadResult(bool reloaded, int total);
public sealed record EmbedCompactResult(bool compacted, int total);
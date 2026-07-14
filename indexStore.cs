
using ModelContextProtocol;
using System.Text.Json;

// Persistence: binary vector blob + JSON sidecar for text/metadata, atomic write via
// temp-file-then-rename (same pattern as the reference project's FsTools.AtomicWrite,
// reimplemented here standalone since this is a separate project, not shared code).
internal static class IndexStore
{
    // Magic + dim in the header so a dimension mismatch (e.g. you swap embedding models
    // later) fails loudly at load rather than silently misreading bytes as garbage floats.
    // [rule 5 failure mode]: without this header check, a stale index built under a
    // different model/dim would load as corrupted noise with NO error — worst kind of
    // failure, wrong answers with no signal. This guards specifically against that.
    const int Magic = 0x45424958; // "EBIX"

    internal static string Root = Directory.GetCurrentDirectory();
    static string VecPath => Path.Combine(Root, "index.vec");
    static string MetaPath => Path.Combine(Root, "index.meta.json");

    sealed record MetaEntry(string Id, string Text, string Source);

    public static void Save()
    {
        var entries = FlatIndex.Snapshot();

        using (var ms = new MemoryStream())
        using (var bw = new BinaryWriter(ms))
        {
            bw.Write(Magic);
            bw.Write(OnnxEmbedder.Dim);
            bw.Write(entries.Length);
            foreach (var e in entries)
                foreach (var f in e.Vector)
                    bw.Write(f);
            AtomicWrite(VecPath, ms.ToArray());
        }

        var meta = entries.Select(e => new MetaEntry(e.Id, e.Text, e.Source)).ToArray();
        AtomicWrite(MetaPath, JsonSerializer.SerializeToUtf8Bytes(meta));
        // Two files, two AtomicWrites: NOT a single atomic transaction across both.
        // [rule 5 failure mode]: a crash between the two writes leaves vectors and
        // metadata out of sync (e.g. new vec blob, stale meta or vice versa) — Load()
        // below only checks internal consistency (magic/dim/count within index.vec),
        // it does NOT cross-validate entry count against index.meta.json's array length.
        // Adding that cross-check is a small follow-up, not done here silently:
        // [rule 18] "want a combined-file format instead, to make this genuinely atomic?"
    }

    public static void Load(string embed_root)
    {
        Root = embed_root ?? throw new McpException("Embedding root directory not configured");

        if (!File.Exists(VecPath) || !File.Exists(MetaPath))
        {
            FlatIndex.LoadSnapshot(Array.Empty<FlatIndex.Entry>());
            return; // fresh start, not an error — no index yet is a valid initial state
        }

        using var fs = File.OpenRead(VecPath);
        using var br = new BinaryReader(fs);
        var magic = br.ReadInt32();
        if (magic != Magic)
            throw new InvalidDataException($"{VecPath}: bad magic, not an index.vec file");
        var dim = br.ReadInt32();
        if (dim != OnnxEmbedder.Dim)
            throw new InvalidDataException(
                $"{VecPath}: index built with dim={dim}, current model dim={OnnxEmbedder.Dim}. " +
                "Rebuild the index against the current model — vectors are not compatible across dims.");
        var count = br.ReadInt32();

        var meta = JsonSerializer.Deserialize<MetaEntry[]>(File.ReadAllBytes(MetaPath))
            ?? throw new InvalidDataException($"{MetaPath}: failed to parse");
        if (meta.Length != count)
            throw new InvalidDataException(
                $"vector count {count} != metadata count {meta.Length} — files out of sync, " +
                "see Save()'s two-file caveat; rebuild the index.");

        var entries = new FlatIndex.Entry[count];
        for (int i = 0; i < count; i++)
        {
            var vec = new float[dim];
            for (int d = 0; d < dim; d++) vec[d] = br.ReadSingle();
            entries[i] = new FlatIndex.Entry(meta[i].Id, vec, meta[i].Text, meta[i].Source);
        }
        FlatIndex.LoadSnapshot(entries);
    }

    static void AtomicWrite(string path, byte[] content)
    {
        var tmp = Path.Combine(Path.GetDirectoryName(path)!, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        File.WriteAllBytes(tmp, content);
        File.Move(tmp, path, overwrite: true); // rename is atomic on same filesystem —
            // NOT atomic if tmp and dest are on different volumes/mounts. [rule 5, edge
            // case] same caveat as any AtomicWrite pattern; irrelevant unless Root spans
            // mount boundaries, flagging rather than assuming your deployment.
    }
}
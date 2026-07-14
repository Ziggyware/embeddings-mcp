using ModelContextProtocol;
using System.Text.Json;

internal static class IndexStore
{
    const int Magic = 0x45424958;
    const byte OpUpsert = 1, OpDelete = 2;

    internal static string Root = Directory.GetCurrentDirectory();
    static string VecPath => Path.Combine(Root, "index.vec");
    static string MetaPath => Path.Combine(Root, "index.meta.json");
    static string WalPath => Path.Combine(Root, "index.wal");
    static readonly object WalLock = new();

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
    }

    public static void AppendUpsert(FlatIndex.Entry e)
    {
        lock (WalLock)
        {
            using var fs = new FileStream(WalPath, FileMode.Append, FileAccess.Write, FileShare.None);
            using var bw = new BinaryWriter(fs);
            WriteUpsertRecord(bw, e);
        }
    }

    public static void AppendUpsertBatch(IEnumerable<FlatIndex.Entry> entries)
    {
        lock (WalLock)
        {
            using var fs = new FileStream(WalPath, FileMode.Append, FileAccess.Write, FileShare.None);
            using var bw = new BinaryWriter(fs);
            foreach (var e in entries) WriteUpsertRecord(bw, e);
        }
    }

    public static void AppendDelete(string id)
    {
        lock (WalLock)
        {
            using var fs = new FileStream(WalPath, FileMode.Append, FileAccess.Write, FileShare.None);
            using var bw = new BinaryWriter(fs);
            bw.Write(OpDelete);
            WriteStr(bw, id);
        }
    }

    static void WriteUpsertRecord(BinaryWriter bw, FlatIndex.Entry e)
    {
        bw.Write(OpUpsert);
        WriteStr(bw, e.Id);
        foreach (var f in e.Vector) bw.Write(f);
        WriteStr(bw, e.Text);
        WriteStr(bw, e.Source);
    }

    static void WriteStr(BinaryWriter bw, string s)
    {
        if (s is null) { bw.Write(-1); return; }
        var bytes = System.Text.Encoding.UTF8.GetBytes(s);
        bw.Write(bytes.Length);
        bw.Write(bytes);
    }

    static string ReadStr(BinaryReader br)
    {
        var len = br.ReadInt32();
        return len < 0 ? null : System.Text.Encoding.UTF8.GetString(br.ReadBytes(len));
    }

    public static void Compact()
    {
        lock (WalLock)
        {
            Save();
            if (File.Exists(WalPath)) File.Delete(WalPath);
        }
    }

    public static void Load(string embed_root)
    {
        Root = embed_root ?? throw new McpException("Embedding root directory not configured");
        var working = new Dictionary<string, FlatIndex.Entry>();

        if (File.Exists(VecPath) && File.Exists(MetaPath))
        {
            using var fs = File.OpenRead(VecPath);
            using var br = new BinaryReader(fs);
            var magic = br.ReadInt32();
            if (magic != Magic) throw new InvalidDataException($"{VecPath}: bad magic, not an index.vec file");
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
                    $"vector count {count} != metadata count {meta.Length} — files out of sync, rebuild the index.");
            for (int i = 0; i < count; i++)
            {
                var vec = new float[dim];
                for (int d = 0; d < dim; d++) vec[d] = br.ReadSingle();
                working[meta[i].Id] = new FlatIndex.Entry(meta[i].Id, vec, meta[i].Text, meta[i].Source);
            }
        }

        if (File.Exists(WalPath))
        {
            using var fs = File.OpenRead(WalPath);
            using var br = new BinaryReader(fs);
            while (fs.Position < fs.Length)
            {
                var op = br.ReadByte();
                var id = ReadStr(br);
                if (op == OpUpsert)
                {
                    var vec = new float[OnnxEmbedder.Dim];
                    for (int d = 0; d < OnnxEmbedder.Dim; d++) vec[d] = br.ReadSingle();
                    var text = ReadStr(br);
                    var source = ReadStr(br);
                    working[id] = new FlatIndex.Entry(id, vec, text, source);
                }
                else if (op == OpDelete) working.Remove(id);
                else throw new InvalidDataException($"{WalPath}: unknown WAL opcode {op} — file corrupt or truncated mid-record");
            }
        }
        FlatIndex.LoadSnapshot(working.Values);
    }

    static void AtomicWrite(string path, byte[] content)
    {
        var tmp = Path.Combine(Path.GetDirectoryName(path)!, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        File.WriteAllBytes(tmp, content);
        File.Move(tmp, path, overwrite: true);
    }
}
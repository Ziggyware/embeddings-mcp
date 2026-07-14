using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System.Numerics;

internal static class OnnxEmbedder
{
    public const int Dim = 768;
    const int MaxSeqLen = 512;
    const string QueryPrefix = "Represent this sentence for searching relevant passages: ";

    static InferenceSession _session;
    static BertTokenizer _tok;
    static bool _initialized;

    public static void Init(string modelPath, string vocabPath)
    {
        if (_initialized)
            throw new InvalidOperationException("OnnxEmbedder.Init already called — re-init not supported.");

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"embedding model not found at {modelPath}.");
        if (!File.Exists(vocabPath))
            throw new FileNotFoundException($"tokenizer vocab not found at {vocabPath}.");

        var opts = new SessionOptions
        {
            InterOpNumThreads = 1,
            IntraOpNumThreads = Environment.ProcessorCount
        };
        _session = new InferenceSession(modelPath, opts);

        using var vocabStream = File.OpenRead(vocabPath);
        _tok = BertTokenizer.Create(vocabStream, new BertOptions { LowerCaseBeforeTokenization = true });

        _initialized = true;
    }

    static void EnsureInitialized()
    {
        if (!_initialized)
            throw new InvalidOperationException(
                "OnnxEmbedder.Init(modelPath, vocabPath) must be called before embedding.");
    }

    public static float[] EmbedQuery(string text) => EmbedOne(QueryPrefix + text);
    public static float[] EmbedPassage(string text) => EmbedOne(text);

    public static float[][] EmbedPassagesBatch(string[] texts)
    {
        EnsureInitialized();
        int b = texts.Length;
        if (b == 0) return Array.Empty<float[]>();

        var tokenized = new IReadOnlyList<int>[b];
        int maxLen = 0;
        for (int i = 0; i < b; i++)
        {
            tokenized[i] = _tok.EncodeToIds(texts[i], MaxSeqLen, addSpecialTokens: true, out _, out _);
            maxLen = Math.Max(maxLen, tokenized[i].Count);
        }

        var ids = new long[b * maxLen];
        var mask = new long[b * maxLen];
        var types = new long[b * maxLen];

        for (int i = 0; i < b; i++)
        {
            var row = tokenized[i];
            int off = i * maxLen;
            for (int t = 0; t < row.Count; t++) { ids[off + t] = row[t]; mask[off + t] = 1; }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(ids, new[] { b, maxLen })),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(mask, new[] { b, maxLen })),
            NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(types, new[] { b, maxLen }))
        };

        using var results = _session.Run(inputs);
        var hidden = ((DenseTensor<float>)results.First(r => r.Name == "last_hidden_state").AsTensor<float>()).Buffer.Span;

        var outVecs = new float[b][];
        int vw = Vector<float>.Count;

        for (int i = 0; i < b; i++)
        {
            var pooled = new float[Dim];
            float maskSum = 0;
            int rowBase = i * maxLen * Dim;
            for (int t = 0; t < maxLen; t++)
            {
                if (mask[i * maxLen + t] == 0) continue;
                maskSum += 1;
                int off = rowBase + t * Dim;
                int d = 0;
                for (; d <= Dim - vw; d += vw)
                    (new Vector<float>(pooled, d) + new Vector<float>(hidden.Slice(off + d, vw))).CopyTo(pooled, d);
                for (; d < Dim; d++) pooled[d] += hidden[off + d];
            }
            if (maskSum > 0)
            {
                var vDiv = new Vector<float>(maskSum);
                int d = 0;
                for (; d <= Dim - vw; d += vw)
                    (new Vector<float>(pooled, d) / vDiv).CopyTo(pooled, d);
                for (; d < Dim; d++) pooled[d] /= maskSum;
            }
            Normalize(pooled);
            outVecs[i] = pooled;
        }
        return outVecs;
    }

    static float[] EmbedOne(string text)
    {
        EnsureInitialized();
        var encoding = _tok!.EncodeToIds(text, MaxSeqLen, addSpecialTokens: true, out _, out _);

        int n = encoding.Count;
        var inputIds = new long[n];
        var attnMask = new long[n];
        var typeIds = new long[n];
        for (int i = 0; i < n; i++) { inputIds[i] = encoding[i]; attnMask[i] = 1; }

        var inputTensor = new DenseTensor<long>(inputIds, new[] { 1, n });
        var maskTensor = new DenseTensor<long>(attnMask, new[] { 1, n });
        var typeTensor = new DenseTensor<long>(typeIds, new[] { 1, n });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor),
            NamedOnnxValue.CreateFromTensor("token_type_ids", typeTensor)
        };

        using var results = _session!.Run(inputs);
        var lastHidden = results.First(r => r.Name == "last_hidden_state").AsTensor<float>();

        var pooled = new float[Dim];
        float maskSum = 0;
        for (int t = 0; t < n; t++)
        {
            if (attnMask[t] == 0) continue;
            maskSum += 1;
            for (int d = 0; d < Dim; d++) pooled[d] += lastHidden[0, t, d];
        }
        if (maskSum > 0)
            for (int d = 0; d < Dim; d++) pooled[d] /= maskSum;

        Normalize(pooled);
        return pooled;
    }

    internal static void Normalize(float[] v)
    {
        double sumSq = 0;
        for (int i = 0; i < v.Length; i++) sumSq += (double)v[i] * v[i];
        var norm = (float)Math.Sqrt(sumSq);
        if (norm < 1e-12f) return;
        for (int i = 0; i < v.Length; i++) v[i] /= norm;
    }
}
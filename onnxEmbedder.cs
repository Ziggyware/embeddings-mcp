using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

// bge-base-en-v1.5: 768d, mean-pooled (mask-aware), L2-normalized. Query side carries an
// instruction prefix; passage side does not.
// [empirical — model card convention; pooling method corrected from CLS to mean per
// bge-base-en-v1.5-ONNX's own documented usage, still worth re-verifying against your
// actual downloaded export if results look degenerate]
internal static class OnnxEmbedder
{
    public const int Dim = 768;
    const int MaxSeqLen = 512;
    const string QueryPrefix = "Represent this sentence for searching relevant passages: ";

    static InferenceSession _session;
    static BertTokenizer _tok;
    static bool _initialized;

    // Explicit init, no static ctor. [rule 5 failure mode]: caller MUST call this before
    // any Embed* call, or Embed* throws a clear InvalidOperationException rather than the
    // opaque "static ctor already ran with null paths" failure this replaces. Called once
    // from Program.cs after paths are known (env vars, CLI args, wherever they come from) —
    // not gated to any one config source, that decision stays outside this file.
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

    public static float[][] EmbedPassagesBatch(string[] texts) =>
        texts.Select(EmbedPassage).ToArray();
    // Still a per-item loop, not a true batch — same disclosed gap as before, unchanged
    // by this refactor. [rule 18, restated not re-derived]

    static float[] EmbedOne(string text)
    {
        EnsureInitialized();
        var encoding = _tok!.EncodeToIds(text, MaxSeqLen, addSpecialTokens: true, out _, out _);
        // Silent truncation past MaxSeqLen — unchanged caveat, restated not re-derived.

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
        // Input/output tensor names still [unverified] against your actual downloaded
        // export — unchanged by this refactor, same remedy as before (InputMetadata.Keys).

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

    static void Normalize(float[] v)
    {
        double sumSq = 0;
        for (int i = 0; i < v.Length; i++) sumSq += (double)v[i] * v[i];
        var norm = (float)Math.Sqrt(sumSq);
        if (norm < 1e-12f) return;
        for (int i = 0; i < v.Length; i++) v[i] /= norm;
    }
}
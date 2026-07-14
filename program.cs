
// Stdio MCP host bootstrap. [unverified — check before relying on this]: your reference
// service's actual Program.cs wasn't in the pasted document, only its tool files were, so
// this mirrors the STANDARD ModelContextProtocol C# SDK stdio bootstrap pattern rather than
// your project's exact one. Reconcile against your real Program.cs before running —
// specifically: server name/version string, and whether you register tool types explicitly
// or via assembly scan.

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using ModelContextProtocol.Server;

var builder = Host.CreateApplicationBuilder(args);

builder.Services
    .AddMcpServer(opts =>
    {
        opts.ServerInfo = new() { Name = "embed-retrieval", Version = "0.1.0" };
    })
    .WithStdioServerTransport()
    .WithToolsFromAssembly();

var host = builder.Build();
if (args.Length != 3)
{
    throw new ModelContextProtocol.McpException("Embedding root directory parameter missing. Parameters must be: EMBED_MODEL_PATH, EMBED_VOCAB_PATH, EMBEDDING_ROOT");
}
OnnxEmbedder.Init(args.ElementAt(0), args.ElementAt(1));
IndexStore.Load(args.ElementAt(2));

await host.RunAsync();
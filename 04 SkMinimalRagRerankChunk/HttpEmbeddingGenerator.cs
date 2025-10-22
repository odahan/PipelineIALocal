// HttpEmbeddingGenerator.cs

using System.Net.Http.Json;
using Microsoft.Extensions.AI;

namespace SkMinimalRagRerankChunk;

public sealed class HttpEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;

    private sealed record EmbedReq(string text);
    private sealed record EmbedRes(string model, float[] embedding);

    public HttpEmbeddingGenerator(HttpClient http, string baseUrl)
    {
        _http = http;
        _baseUrl = baseUrl.TrimEnd('/');
    }

    // Méthode officielle : on génère pour une séquence d'entrées et on renvoie un GeneratedEmbeddings<Embedding<float>>
    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var list = new List<Embedding<float>>();

        foreach (var v in values)
        {
            if (string.IsNullOrWhiteSpace(v))
                throw new ArgumentException("Embedding: texte vide ou blanc.");

            using var res = await _http.PostAsJsonAsync($"{_baseUrl}/embed", new EmbedReq(v), cancellationToken);
            if (!res.IsSuccessStatusCode)
            {
                var err = await res.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException($"Embeddings POST failed {(int)res.StatusCode} {res.ReasonPhrase}. Body: {err}");
            }

            var obj = await res.Content.ReadFromJsonAsync<EmbedRes>(cancellationToken: cancellationToken)
                      ?? throw new InvalidOperationException("Réponse /embed vide");

            list.Add(new Embedding<float>(obj.embedding));
        }


        var gens = new GeneratedEmbeddings<Embedding<float>>(list);
        return gens;
    }

    // Contrat de l’interface non générique parente (optionnel ici)
    public object? GetService(Type serviceType, object? key) => null;

    public void Dispose() { }
}
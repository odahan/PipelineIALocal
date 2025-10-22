using System.Net.Http.Json;
using System.Text;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

// IAIService.Attributes

namespace SkMinimalRagRerankChunk;

public sealed class OllamaChatService : IChatCompletionService
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;
    private readonly string _model;
    private readonly IReadOnlyDictionary<string, object> _attributes;

    private sealed record GenReq(string model, string prompt, bool stream);
    private sealed record GenRes(string response);

    public OllamaChatService(
        HttpClient http,
        string baseUrl,
        string model,
        IReadOnlyDictionary<string, object>? attributes = null)
    {
        _http = http;
        _baseUrl = baseUrl.TrimEnd('/');
        _model = model;
        _attributes = attributes ?? new Dictionary<string, object>();
    }

    public IReadOnlyDictionary<string, object> Attributes => _attributes;

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chat,
        PromptExecutionSettings? settings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = BuildPromptFromHistory(chat);

        using var res = await _http.PostAsJsonAsync(
            $"{_baseUrl}/api/generate",
            new GenReq(_model, prompt, stream: false),
            cancellationToken);

        res.EnsureSuccessStatusCode();

        var obj = await res.Content.ReadFromJsonAsync<GenRes>(cancellationToken: cancellationToken)
                  ?? throw new InvalidOperationException("Réponse vide d'Ollama.");

        var msg = new ChatMessageContent(AuthorRole.Assistant, obj.response?.Trim() ?? "");
        return new[] { msg };
    }

    public IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chat,
        PromptExecutionSettings? settings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
        => throw new NotImplementedException("Streaming non implémenté pour ce POC.");

    private static string BuildPromptFromHistory(ChatHistory chat)
    {
        var sb = new StringBuilder();

        foreach (var m in chat)
        {
            // On normalise simplement le label de rôle fourni par SK
            var role = (m.Role.Label ?? "user").ToLowerInvariant();
            if (role != "system" && role != "user" && role != "assistant")
                role = "user"; // valeur par défaut sûre

            sb.Append(role).Append(": ").AppendLine(m.Content ?? "");
        }

        // Forcer la prise de parole du modèle
        sb.Append("assistant: ");
        return sb.ToString();
    }
}
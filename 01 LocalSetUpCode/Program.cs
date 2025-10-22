// Program.cs — Top-level statements (.NET 9)
// RAG local minimal : Embeddings (HTTP) -> Qdrant -> Ollama
// Variante avec IDs de points en UUID (string) pour Qdrant.

/* LICENCE
   ──────────────────────────────────────────────────────────────────────────
   Conditions d’utilisation – Code de démonstration (Initiation Semantic Kernel)
   ──────────────────────────────────────────────────────────────────────────

   © 2025 Olivier Dahan & E-Naxos – Tous droits réservés.

   1) Objet et périmètre
      Ce code est fourni « tel quel », sans aucune garantie, à des fins
      exclusivement pédagogiques, en accompagnement de la vidéo
      « Pipeline IA Local » publiée sur :
      https://www.youtube.com/@e-naxosConsulting

   2) Propriété intellectuelle
      La propriété du code et des documents associés appartient à
      Olivier Dahan & E-Naxos. 
      Toute utilisation hors du cadre pédagogique personnel, toute
      intégration dans un produit/service, tout usage professionnel
      ou commercial, toute mise en production, toute modification,
      adaptation, publication ou redistribution, en tout ou partie,
      sont interdits sans autorisation écrite préalable de l’auteur.

   3) Licence applicable aux documents fournis (code + supports)
      L’ensemble des documents fournis est régi par la licence :
      Creative Commons Attribution – NonCommercial – NoDerivatives 4.0 International
      (CC BY-NC-ND 4.0)

      Texte officiel : https://creativecommons.org/licenses/by-nc-nd/4.0/
      Effet pratique (résumé non contractuel) :
        • Vous pouvez télécharger et partager le contenu tel quel,
          avec attribution, sans usage commercial, et sans modification.
        • Aucune création d’œuvre dérivée n’est autorisée.
        • Toute autre utilisation ou publication nécessite l’accord écrit
          préalable de l’éditeur (Olivier Dahan & E-Naxos).

   4) Exclusion de garantie et limitation de responsabilité
      CE CONTENU EST FOURNI « EN L’ÉTAT », SANS AUCUNE GARANTIE EXPRESSE
      OU IMPLICITE, Y COMPRIS, SANS S’Y LIMITER, LES GARANTIES DE QUALITÉ
      MARCHANDE, D’ADÉQUATION À UN USAGE PARTICULIER ET D’ABSENCE
      DE CONTREFAÇON. EN AUCUN CAS L’AUTEUR/L’ÉDITEUR NE SAURAIT ÊTRE
      TENU RESPONSABLE DE DOMMAGES DIRECTS OU INDIRECTS, SPÉCIAUX,
      ACCESSOIRES OU CONSÉCUTIFS, PERTES DE DONNÉES OU D’EXPLOITATION,
      DÉCOULANT DE L’UTILISATION OU DE L’IMPOSSIBILITÉ D’UTILISER CE CODE,
      MÊME SI LA POSSIBILITÉ DE TELS DOMMAGES A ÉTÉ SIGNALÉE.

   5) Tolérance d’usage
      Autorisé : consultation, exécution locale et étude à titre d’exemple,
      à des fins personnelles d’apprentissage, strictement dans le cadre
      de la série précitée.
      Interdit : tout autre usage (notamment professionnel/commercial),
      toute redistribution ou hébergement public (dépôts Git, gists,
      packages, sites, etc.) sans accord écrit préalable.

   ──────────────────────────────────────────────────────────────────────────
*/


using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

// -------------------- CONFIG --------------------
const string OllamaBase = "http://localhost:11434";
const string OllamaModel = "llama3";                 // alias sûr dans ton env
const string EmbedBase = "http://localhost:5000";   // serveur d'embeddings local (/embed)
const string QdrantBase = "http://localhost:6333";   // Qdrant (Docker)
const string Collection = "docs_demo";

// HttpClient partagé
var http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };

Console.OutputEncoding = Encoding.UTF8;
Console.WriteLine("=== RAG local minimal (UUID) : Embeddings -> Qdrant -> Ollama ===\n");

// 0) Sanity checks
await SanityCheckAsync();

// 1) Détecter la dimension d'embedding
var testVec = await GetEmbeddingAsync("texte de test pour dimension");
int dim = testVec.Length;
Console.WriteLine($"[Embed] Dimension détectée = {dim}");

// --- Vider + recréer la collection ici ---
await ResetCollectionAsync(Collection, dim);

// 2) Créer/valider la collection Qdrant
await EnsureCollectionAsync(Collection, dim);

// 3) Ingestion de quelques documents — IDs = UUID (string)
var docs = new (string id, string text)[]
{
    (Guid.NewGuid().ToString(), "Le RAG combine une base vectorielle et un modèle génératif."),
    (Guid.NewGuid().ToString(), ".NET 9 apporte des améliorations de performance et du tooling modernisé."),
    (Guid.NewGuid().ToString(), "Qdrant est une base de données vectorielle performante et simple à déployer.")
};
await IngestAsync(Collection, docs);

// 4) Question -> embedding -> search -> prompt -> génération
var question = "Explique-moi brièvement ce qu'est le RAG et le rôle de Qdrant.";
var queryVec = await GetEmbeddingAsync(question);
var topK = await SearchAsync(Collection, queryVec, k: 3);

Console.WriteLine("\n[TopK] Passages retenus :");
foreach (var (text, score) in topK)
    Console.WriteLine($" - (score {score:F3}) {text}");

var prompt = BuildPrompt(topK, question);
var answer = await GenerateAsync(prompt);

Console.WriteLine("\n[LLM] Réponse :\n");
Console.WriteLine(answer);
Console.WriteLine("\n=== Terminé ===");

// -------------------- Fonctions locales --------------------

// Vérifie rapidement que les trois services externes répondent (Embeddings, Qdrant, Ollama).
async Task SanityCheckAsync()
{
    try
    {
        var r1 = await http.GetAsync($"{EmbedBase}/ready");
        Console.WriteLine(r1.IsSuccessStatusCode ? "[Check] Embeddings OK" : "[Check] Embeddings: /ready non OK");

        var r2 = await http.GetAsync($"{QdrantBase}/");
        Console.WriteLine(r2.IsSuccessStatusCode ? "[Check] Qdrant OK" : "[Check] Qdrant: racine non OK");

        var mini = new OllamaGenerateRequest(OllamaModel, "ping", false);
        var r3 = await http.PostAsJsonAsync($"{OllamaBase}/api/generate", mini);
        Console.WriteLine(r3.IsSuccessStatusCode ? "[Check] Ollama OK" : "[Check] Ollama: /api/generate non OK");
    }
    catch (Exception ex)
    {
        Console.WriteLine("[Check] Avertissement: " + ex.Message);
    }
}

// Retourne le vecteur d'embedding pour un texte donné via le service local d'embeddings.
async Task<float[]> GetEmbeddingAsync(string text)
{
    var res = await http.PostAsJsonAsync($"{EmbedBase}/embed", new EmbedRequest(text));
    res.EnsureSuccessStatusCode();
    var obj = await res.Content.ReadFromJsonAsync<EmbedResponse>();
    if (obj?.embedding == null || obj.embedding.Length == 0)
        throw new InvalidOperationException("Embedding vide.");
    return obj.embedding;
}

// Crée la collection Qdrant si nécessaire (distance cosinus), sinon ne fait rien.
async Task EnsureCollectionAsync(string name, int size)
{
    var body = new QdrantCreateCollectionRequest(new Vectors(size, "Cosine"));
    var res = await http.PutAsJsonAsync($"{QdrantBase}/collections/{name}", body);

    if (res.IsSuccessStatusCode)
    {
        Console.WriteLine($"[Qdrant] Collection '{name}' créée (size={size}).");
        return;
    }

    var txt = await res.Content.ReadAsStringAsync();
    if ((int)res.StatusCode == 409 || txt.Contains("already exists", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine($"[Qdrant] Collection '{name}' déjà existante.");
        return;
    }

    res.EnsureSuccessStatusCode();
}

// Vectorise et insère des documents dans Qdrant (IDs en UUID string).
async Task IngestAsync(string name, (string id, string text)[] docs)
{
    var points = new List<Point>();
    foreach (var (id, text) in docs)
    {
        var vec = await GetEmbeddingAsync(text);  // 384 floats attendus
        var payload = new Dictionary<string, object> { ["text"] = text };
        points.Add(new Point(id, vec, payload));  // id = UUID (string)
    }

    var upsert = new QdrantUpsertRequest(points.ToArray(), wait: true);
    var res = await http.PutAsJsonAsync($"{QdrantBase}/collections/{name}/points", upsert);
    res.EnsureSuccessStatusCode();
    Console.WriteLine($"[Qdrant] Ingesté {points.Count} document(s).");
}

// Recherche les k points les plus proches dans Qdrant et renvoie texte + score.
async Task<List<(string text, double score)>> SearchAsync(string name, float[] vector, int k)
{
    var req = new QdrantSearchRequest(vector, k, with_payload: true);
    var res = await http.PostAsJsonAsync($"{QdrantBase}/collections/{name}/points/search", req);
    res.EnsureSuccessStatusCode();

    var json = await res.Content.ReadAsStringAsync();
    using var doc = JsonDocument.Parse(json);
    var results = new List<(string, double)>();

    if (doc.RootElement.TryGetProperty("result", out var arr) && arr.ValueKind == JsonValueKind.Array)
    {
        foreach (var item in arr.EnumerateArray())
        {
            double score = item.GetProperty("score").GetDouble();
            string text = item.GetProperty("payload").GetProperty("text").GetString() ?? "";
            results.Add((text, score));
        }
    }
    return results;
}

// Appelle Ollama pour générer une réponse non-streamée à partir d'un prompt.
async Task<string> GenerateAsync(string prompt)
{
    var req = new OllamaGenerateRequest(OllamaModel, prompt, stream: false);
    var res = await http.PostAsJsonAsync($"{OllamaBase}/api/generate", req);
    res.EnsureSuccessStatusCode();
    var obj = await res.Content.ReadFromJsonAsync<OllamaGenerateResponse>();
    return obj?.response?.Trim() ?? "(réponse vide)";
}

// Construit un prompt concis en français à partir des passages retrouvés et de la question.
string BuildPrompt(List<(string text, double score)> topK, string question)
{
    var sb = new StringBuilder();
    sb.AppendLine("Tu es un assistant technique. Réponds en français clair, concis et exact.");
    sb.AppendLine("Contexte (passages pertinents) :");
    foreach (var (text, _) in topK)
        sb.AppendLine($"- {text}");
    sb.AppendLine();
    sb.AppendLine($"Question : {question}");
    sb.AppendLine("Réponse (appuie-toi sur le contexte fourni, évite les spéculations) :");
    return sb.ToString();
}

// Supprime puis recrée la collection Qdrant avec la dimension fournie.
async Task ResetCollectionAsync(string name, int size)
{
    using var httpLocal = new HttpClient();

    // 1) Supprimer la collection si elle existe (ignore 404)
    var del = await httpLocal.DeleteAsync($"{QdrantBase}/collections/{name}");
    if (del.IsSuccessStatusCode)
    {
        Console.WriteLine($"[Qdrant] Collection '{name}' supprimée.");
    }
    else if ((int)del.StatusCode != 404)
    {
        del.EnsureSuccessStatusCode();
    }

    // 2) Recréer la collection avec la bonne dimension
    var body = new { vectors = new { size, distance = "Cosine" } };
    var create = await httpLocal.PutAsJsonAsync($"{QdrantBase}/collections/{name}", body);
    create.EnsureSuccessStatusCode();
    Console.WriteLine($"[Qdrant] Collection '{name}' recréée (size={size}).");
}


// -------------------- Types de transport JSON --------------------

// Requête envoyée au service d'embeddings.
public record EmbedRequest(string text);

// Réponse du service d'embeddings contenant le modèle et le vecteur.
public record EmbedResponse(string model, float[] embedding);

// Requête au point de terminaison /api/generate d'Ollama.
public record OllamaGenerateRequest(string model, string prompt, bool stream = false);

// Réponse texte renvoyée par Ollama après génération.
public record OllamaGenerateResponse(string response);

// Corps de création d'une collection Qdrant (configuration des vecteurs).
public record QdrantCreateCollectionRequest(Vectors vectors);

// Spécification des vecteurs (dimension et métrique de distance).
public record Vectors(int size, string distance);

// Point Qdrant: identifiant UUID (string), vecteur et charge utile (payload).
public record Point(string id, float[] vector, Dictionary<string, object> payload);

// Requête d'upsert de points dans Qdrant, avec option wait pour application synchrone.
public record QdrantUpsertRequest(Point[] points, bool wait = true);

// Requête de recherche vectorielle dans Qdrant (vecteur requête, nombre de résultats, payload).
public record QdrantSearchRequest(float[] vector, int limit, bool with_payload);

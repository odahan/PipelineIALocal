// Program.cs — Top-level statements (.NET 9)
// SK minimal : Embeddings (IEmbeddingGenerator) -> Qdrant (REST) -> Chat (IChatCompletionService/Ollama)

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

using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using SkMinimalRag;

// ------------ CONFIG ------------
// URLs des services locaux (adapter si nécessaire)
const string OllamaBase = "http://localhost:11434"; // Endpoint Ollama (chat/génération)
const string EmbedBase = "http://localhost:5000";   // Serveur d'embeddings (endpoint /embed)
const string QdrantBase = "http://localhost:6333";  // Qdrant (Docker par défaut)
const string Collection = "docs_demo";              // Nom de collection Qdrant

// --- Sélection du modèle LLM ---
// Priorité : argument --model <nom> > env OLLAMA_MODEL > défaut "llama3".
string? modelFromArgs = null;
for (int i = 0; i + 1 < args.Length; i++)
{
    if (args[i].Equals("--model", StringComparison.OrdinalIgnoreCase))
    {
        modelFromArgs = args[i + 1];
        break;
    }
}

// Si pas de modèle en argument, demande à l'utilisateur
// Attention : les modèles doivent être préalablement installés dans Ollama.
if (modelFromArgs == null || string.IsNullOrWhiteSpace(modelFromArgs))
{
    Console.WriteLine("Sélectionnez un modèle en tapant son numéro :");
    Console.WriteLine(" 1. qwen3");
    Console.WriteLine(" 2. llama3 (défaut)");   
    Console.WriteLine(" 3. gpt-oss");
    Console.Write("Votre choix (1-3) : ");
    var choice = Console.ReadLine();
    modelFromArgs = choice switch
    {
        "1" => "qwen3",
        "3" => "gpt-oss",
        _ => "llama3", // Par défaut
    };

}


var OllamaModel = !string.IsNullOrWhiteSpace(modelFromArgs)
    ? modelFromArgs!
    : (Environment.GetEnvironmentVariable("OLLAMA_MODEL") ?? "llama3");


Console.OutputEncoding = Encoding.UTF8; // Affichage UTF-8 dans la console
Console.WriteLine($"[Config] Modèle Ollama = {OllamaModel}");

// ------------ Kernel & services ------------
// Construction d'un Kernel SK minimal + DI de HttpClient, embeddings et chat
var builder = Kernel.CreateBuilder();
builder.Services.AddSingleton(new HttpClient());

// Embeddings via Microsoft.Extensions.AI (implémentation HTTP custom)
builder.Services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(
    sp => new HttpEmbeddingGenerator(sp.GetRequiredService<HttpClient>(), EmbedBase));

// Chat via SK (Ollama) avec le modèle choisi dynamiquement
builder.Services.AddSingleton<IChatCompletionService>(
    sp => new OllamaChatService(sp.GetRequiredService<HttpClient>(), OllamaBase, OllamaModel));

var kernel = builder.Build();

var embed = kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
var chat = kernel.GetRequiredService<IChatCompletionService>();

// ------------ Sanity checks ------------
// Vérifie la disponibilité des 3 services (Embeddings, Qdrant, Ollama)
await SanityCheckAsync();

// 1) Détection de la dimension d'embedding du service
var probe = await embed.GenerateAsync(new[] { "probe" });
var dim = probe[0].Vector.Length; // Embedding<float>.Vector : ReadOnlyMemory<float>
Console.WriteLine($"[Embed] Dimension détectée = {dim}");

// --- Vider (reset) la collection ici pour un run reproductible ---
await ResetCollectionAsync(Collection, dim);

// 2) S'assurer que la collection existe (idempotent)
await EnsureCollectionAsync(Collection, dim);

// 3) Ingestion de quelques documents de démonstration (ids = UUID)
var docs = new (string id, string text)[]
{
    (Guid.NewGuid().ToString(), "Le RAG combine une base vectorielle et un modèle génératif."),
    (Guid.NewGuid().ToString(), ".NET 9 apporte des améliorations de performance et du tooling modernisé."),
    (Guid.NewGuid().ToString(), "Qdrant est une base de données vectorielle performante et simple à déployer.")
};
await IngestAsync(Collection, docs);

// 4) Pipeline RAG : question -> embedding -> recherche Qdrant -> prompt -> chat
var question = "Explique-moi brièvement ce qu'est le RAG et le rôle de Qdrant.";
var q = await embed.GenerateAsync(new[] { question });
var qv = q[0].Vector.ToArray();

var topK = await SearchAsync(Collection, qv, 3);
Console.WriteLine("\n[TopK] Passages retenus :");
foreach (var (text, score) in topK)
    Console.WriteLine($" - (score {score:F3}) {text}");

// Construction d'un historique de chat minimal (system + user) avec le contexte récupéré
var history = new ChatHistory();
history.AddSystemMessage("Tu es un assistant technique francophone. Style: clair, professionnel, concis, sans anglicismes ni tutoiement.");
history.AddUserMessage($"Contexte :\n- {string.Join("\n- ", topK.Select(t => t.text))}\n\nQuestion : {question}\nRéponds en t'appuyant prioritairement sur le contexte.");

// Appel du service de chat (Ollama via SK) et affichage de la réponse
var messages = await chat.GetChatMessageContentsAsync(history);
Console.WriteLine("\n[SK] Réponse :\n" + messages[0].Content);

// ============ Fonctions locales (REST Qdrant) ============

// Vérifie la santé des services externes (Embeddings, Qdrant, Ollama) et logue l'état.
async Task SanityCheckAsync()
{
    using var http = new HttpClient();
    try
    {
        var r1 = await http.GetAsync($"{EmbedBase}/ready");
        Console.WriteLine(r1.IsSuccessStatusCode ? "[Check] Embeddings OK" : "[Check] Embeddings: /ready KO");

        var r2 = await http.GetAsync($"{QdrantBase}/");
        Console.WriteLine(r2.IsSuccessStatusCode ? "[Check] Qdrant OK" : "[Check] Qdrant KO");

        var mini = new { model = OllamaModel, prompt = "ping", stream = false };
        var r3 = await http.PostAsJsonAsync($"{OllamaBase}/api/generate", mini);
        Console.WriteLine(r3.IsSuccessStatusCode ? "[Check] Ollama OK" : "[Check] Ollama KO");
    }
    catch (Exception ex) { Console.WriteLine("[Check] " + ex.Message); }
}

// Crée la collection Qdrant si absente (idempotent). Utilise la distance Cosine et la dimension fournie.
async Task EnsureCollectionAsync(string name, int size)
{
    using var http = new HttpClient();
    var body = new { vectors = new { size, distance = "Cosine" } };
    var res = await http.PutAsJsonAsync($"{QdrantBase}/collections/{name}", body);
    if (res.IsSuccessStatusCode) { Console.WriteLine($"[Qdrant] Collection '{name}' créée."); return; }

    var txt = await res.Content.ReadAsStringAsync();
    if ((int)res.StatusCode == 409 || txt.Contains("already exists", StringComparison.OrdinalIgnoreCase))
        Console.WriteLine($"[Qdrant] Collection '{name}' déjà existante.");
    else
        res.EnsureSuccessStatusCode();
}

// Ingestion en batch: génère les embeddings, construit les points et upsert dans Qdrant (payload: text).
async Task IngestAsync(string name, (string id, string text)[] docs)
{
    using var http = new HttpClient();

    // Génération des embeddings en batch
    var gens = await embed.GenerateAsync(docs.Select(d => d.text));
    var vectors = gens.Select(e => e.Vector.ToArray()).ToArray();

    // Prépare le payload des points (id, vecteur, métadonnées)
    var points = new List<object>(docs.Length);
    for (int i = 0; i < docs.Length; i++)
        points.Add(new { id = docs[i].id, vector = vectors[i], payload = new Dictionary<string, object> { ["text"] = docs[i].text } });

    // Upsert dans Qdrant (wait=true pour attendre la complétion)
    var upsert = new { points, wait = true };
    var res = await http.PutAsJsonAsync($"{QdrantBase}/collections/{name}/points", upsert);
    res.EnsureSuccessStatusCode();
    Console.WriteLine($"[Qdrant] Ingesté {points.Count} document(s).");
}

// Recherche des k voisins les plus proches dans Qdrant et retourne (texte, score) pour chaque item.
async Task<List<(string text, double score)>> SearchAsync(string name, float[] vector, int k)
{
    using var http = new HttpClient();
    var req = new { vector, limit = k, with_payload = true };
    var res = await http.PostAsJsonAsync($"{QdrantBase}/collections/{name}/points/search", req);
    res.EnsureSuccessStatusCode();

    // Parsing minimal de la réponse JSON (texte + score)
    var json = await res.Content.ReadAsStringAsync();
    using var doc = JsonDocument.Parse(json);
    var list = new List<(string, double)>();
    foreach (var item in doc.RootElement.GetProperty("result").EnumerateArray())
    {
        var score = item.GetProperty("score").GetDouble();
        var text = item.GetProperty("payload").GetProperty("text").GetString() ?? "";
        list.Add((text, score));
    }
    return list;
}

// Réinitialise une collection : supprime si présente (ignore 404) puis recrée avec la dimension donnée.
async Task ResetCollectionAsync(string name, int size)
{
    using var http = new HttpClient();

    // 1) Supprimer la collection si elle existe (ignore 404)
    var del = await http.DeleteAsync($"{QdrantBase}/collections/{name}");
    if (del.IsSuccessStatusCode)
        Console.WriteLine($"[Qdrant] Collection '{name}' supprimée.");
    else if ((int)del.StatusCode != 404)
        del.EnsureSuccessStatusCode();

    // 2) Recréer la collection avec la bonne dimension
    var body = new { vectors = new { size, distance = "Cosine" } };
    var create = await http.PutAsJsonAsync($"{QdrantBase}/collections/{name}", body);
    create.EnsureSuccessStatusCode();
    Console.WriteLine($"[Qdrant] Collection '{name}' recréée (size={size}).");
}
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

if (modelFromArgs == null || string.IsNullOrWhiteSpace(modelFromArgs))
{
    Console.WriteLine("Sélectionnez un modèle en tapant son numéro :");
    Console.WriteLine(" 1. qwen2.5");
    Console.WriteLine(" 2. llama3 (défaut)");
    Console.WriteLine(" 3. mistral");
    Console.Write("Votre choix (1-3) : ");
    var choice = Console.ReadLine();
    modelFromArgs = choice switch
    {
        "1" => "qwen2.5",
        "3" => "mistral",
        _ => "llama3", // Par défaut
    };

}


// --- Options Rerank ---
bool useRerank = args.Any(a => a.Equals("--rerank", StringComparison.OrdinalIgnoreCase));
int kVec = 30;   // candidats récupérés en vectoriel
int rTop = 5;    // combien on garde après rerank

for (int i = 0; i + 1 < args.Length; i++)
{
    if (args[i].Equals("--k", StringComparison.OrdinalIgnoreCase) && int.TryParse(args[i + 1], out var k)) kVec = k;
    if (args[i].Equals("--r", StringComparison.OrdinalIgnoreCase) && int.TryParse(args[i + 1], out var r)) rTop = r;
}

if (!args.Any(a => a.Equals("--rerank", StringComparison.OrdinalIgnoreCase)))
{
    Console.WriteLine("Choix du Reranking 0=non, 1=oui (défaut 0) : ");
    var rr = Console.ReadLine();
    if (rr == "1") useRerank = true;
    if (useRerank)
    {
        Console.WriteLine($"Valeur de k (default{kVec}) : ");
        var kk = Console.ReadLine();
        if (int.TryParse(kk, out var k)) kVec = k;
        Console.WriteLine($"Valeur de r (default{rTop}) : ");
        var rr2 = Console.ReadLine();
        if (int.TryParse(rr2, out var r)) rTop = r;
    }
}


Console.WriteLine($"[Config] Rerank={(useRerank ? "ON" : "OFF")} | k={kVec} | r={rTop}");


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
    (Guid.NewGuid().ToString(), ".NET 9 apporte des améliorations de performance et du tooling modernisé."),
    (Guid.NewGuid().ToString(), "Le RAG associe recherche sémantique et LLM pour produire des réponses sourcées."),
    (Guid.NewGuid().ToString(), "Les embeddings transforment les textes en vecteurs comparables par similarité."),
    (Guid.NewGuid().ToString(), "La normalisation L2 stabilise la similarité cosinus entre vecteurs."),
    (Guid.NewGuid().ToString(), "Qdrant s’appuie sur l’index HNSW pour accélérer la recherche de voisins."),
    (Guid.NewGuid().ToString(), "Cosine, produit scalaire et distance euclidienne sont des métriques usuelles en recherche vectorielle."),
    (Guid.NewGuid().ToString(), "Un bon chunking améliore le rappel : 300–500 tokens avec chevauchement modéré."),
    (Guid.NewGuid().ToString(), "Les métadonnées (payload) permettent de filtrer par langue, catégorie ou date."),
    (Guid.NewGuid().ToString(), "Le RAG combine une base vectorielle et un modèle génératif."),
    (Guid.NewGuid().ToString(), "Le reranking re-score le top-K avec un cross-encoder plus précis que le bi-encodeur."),
    (Guid.NewGuid().ToString(), "Une hybrid search combine BM25 (lexical) et vecteurs pour plus de robustesse."),
    (Guid.NewGuid().ToString(), "Le top-K initial doit être assez large pour laisser travailler le reranker."),
    (Guid.NewGuid().ToString(), "Semantic Kernel orchestre l’embedding, la recherche et la génération dans un même pipeline."),
    (Guid.NewGuid().ToString(), "Ollama exécute des modèles LLM en local via une API HTTP unifiée."),
    (Guid.NewGuid().ToString(), "La quantisation Q4/Q5 réduit la VRAM au prix d’un impact qualité modéré."),
    (Guid.NewGuid().ToString(), "ONNX Runtime accélère l’inférence sur CPU et GPU (DirectML ou CUDA)."),
    (Guid.NewGuid().ToString(), "Docker isole les services (embeddings, reranker, Qdrant) et simplifie le déploiement."),
    (Guid.NewGuid().ToString(), "Un micro-service d’embeddings expose /ready et /embed pour la supervision et l’appel."),
    (Guid.NewGuid().ToString(), "Qdrant est une base de données vectorielle performante et simple à déployer."),
    (Guid.NewGuid().ToString(), "E5 multilingue est robuste pour la recherche de passages en plusieurs langues."),
    (Guid.NewGuid().ToString(), "BGE-M3 offre d’excellentes performances multilingues en retrieval."),
    (Guid.NewGuid().ToString(), "Limiter la longueur des passages réduit la latence du reranker sans trop perdre en précision."),
    (Guid.NewGuid().ToString(), "Le cross-encoder lit la requête et le passage ensemble pour noter la pertinence."),
    (Guid.NewGuid().ToString(), "Un prompt clair doit rappeler au LLM d’appuyer sa réponse sur le contexte fourni."),
    (Guid.NewGuid().ToString(), "Le warm-up du modèle raccourcit la latence de la première requête."),
    (Guid.NewGuid().ToString(), "Qdrant supporte des filtres complexes et des index de payload pour accélérer les requêtes."),
    (Guid.NewGuid().ToString(), "La commande upsert ajoute ou remplace des points par identifiant unique."),
    (Guid.NewGuid().ToString(), "Avec la distance cosinus, la normalisation des embeddings est fortement recommandée."),
    (Guid.NewGuid().ToString(), "Dans un diviseur chargé, R2 en parallèle avec RL abaisse la tension de sortie."),
    (Guid.NewGuid().ToString(), "Un filtre RC passe-bas a une fréquence de coupure fc = 1/(2πRC)."),
    (Guid.NewGuid().ToString(), "Qwen 2.5 produit souvent un français naturel et suit bien les instructions."),
    (Guid.NewGuid().ToString(), "Mistral 7B est rapide et correct en français, adapté aux environnements contraints."),
    (Guid.NewGuid().ToString(), "Llama 3 est efficace mais peut adopter un style plus anglicisé en français."),
    (Guid.NewGuid().ToString(), "Un catalogue produit contient titres, attributs, prix et descriptions normalisées."),
    (Guid.NewGuid().ToString(), "Une FAQ regroupe des questions fréquentes et des réponses synthétiques et stables."),
    (Guid.NewGuid().ToString(), "L’harmonisation des unités (cm vs mm) évite des confusions lors de l’indexation."),
    (Guid.NewGuid().ToString(), "Le mapping de synonymes améliore la récupération lexicale sur des requêtes variées."),
    (Guid.NewGuid().ToString(), "Le score de similarité décroît à mesure que les textes s’éloignent sémantiquement."),
    (Guid.NewGuid().ToString(), "En RAG, mieux vaut quelques passages très pertinents qu’un contexte trop verbeux."),
    (Guid.NewGuid().ToString(), "La Révolution française débute en 1789 et transforme durablement les institutions."),
    (Guid.NewGuid().ToString(), "Le protocole HTTP définit des méthodes comme GET, POST, PUT et DELETE."),
    (Guid.NewGuid().ToString(), "Le JSON est un format d’échange texte simple, clé-valeur, largement supporté."),
    (Guid.NewGuid().ToString(), "Un moteur asynchrone triphasé est robuste et économique en maintenance."),
    (Guid.NewGuid().ToString(), "Pour une TV, le HDR et 120 Hz améliorent la dynamique et la fluidité des jeux."),
    (Guid.NewGuid().ToString(), "Une perceuse 18 V brushless fournit un couple élevé pour le perçage exigeant."),
    (Guid.NewGuid().ToString(), "Un aspirateur sans sac facilite la maintenance et limite les consommables."),
};

await IngestAsync(Collection, docs);

// 4) Pipeline RAG : question -> embedding -> recherche Qdrant -> prompt -> chat
var question = "Explique-moi brièvement ce qu'est le RAG et le rôle de Qdrant.";
var q = await embed.GenerateAsync(new[] { question });
var qv = q[0].Vector.ToArray();

// Récupère LARGE pour donner de la matière au reranker sauf si pas de rerank alors rTop utilisé
var topK = await SearchAsync(Collection, qv, useRerank ? kVec : rTop);

// Affichage des K meilleurs vectoriels (diagnostic)
Console.WriteLine($"\n[Vectoriel] {kVec} candidats (top quelques-uns) :");
foreach (var (text, score) in topK.Take(Math.Min(5, topK.Count)))
    Console.WriteLine($" - (score {score:F3}) {text}");

// rerank si demandé
List<(string text, double score)> picked;
if (useRerank)
{
    picked = await RerankAsync(question, topK.Select(t => t.text), topN: rTop);
    Console.WriteLine("\n[Rerank] Passages retenus :");
    foreach (var (text, score) in picked)
        Console.WriteLine($" - (score {score:F3}) {text}");
}
else
{
    picked = topK.Take(rTop).ToList();
    Console.WriteLine("\n[TopK] Passages retenus (sans rerank) :");
    foreach (var (text, score) in picked)
        Console.WriteLine($" - (score {score:F3}) {text}");
}

// Construction d'un historique de chat minimal (system + user) avec le contexte retenu
var history = new ChatHistory();
history.AddSystemMessage("Tu es un assistant technique francophone. Style: clair, professionnel, concis, sans anglicismes ni tutoiement.");
history.AddUserMessage($"Contexte :\n- {string.Join("\n- ", picked.Select(t => t.text))}\n\nQuestion : {question}\nRéponds en t'appuyant prioritairement sur le contexte.");


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

// Rerank une liste de candidats via le service /rerank local (Ollama).
async Task<List<(string text, double score)>> RerankAsync(string query, IEnumerable<string> candidates, int topN = 5)
{
    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
    var req = new RerankRequest(query, candidates.ToArray(), topN);
    var res = await http.PostAsJsonAsync("http://localhost:5001/rerank", req);
    res.EnsureSuccessStatusCode();

    var obj = await res.Content.ReadFromJsonAsync<RerankResponse>()
              ?? throw new InvalidOperationException("Réponse /rerank vide");
    return obj.items.Select(i => (i.text, i.score)).ToList();
}

record RerankRequest(string query, string[] candidates, int top_n = 5);
record RerankItem(string text, double score);
record RerankResponse(string model, RerankItem[] items);



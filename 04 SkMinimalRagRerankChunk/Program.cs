// Program.cs — entrypoint classique (classe Program)
// Pipeline local : Embeddings (HTTP) -> Qdrant (REST) -> Rerank (HTTP) -> Chat (Ollama via SK)
// + Ingestion PDF/DOCX (chunking+overlap), dédup SHA256 (manifest), reset optionnel, prune, préfixes embeddings, sources.

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

using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;


namespace SkMinimalRagRerankChunk
{
    public class Program
    {
        // ----------- CONFIG GLOBALE -----------
        private const string OllamaBase = "http://localhost:11434";
        private const string EmbedBase = "http://localhost:5000";
        private const string QdrantBase = "http://localhost:6333";
        private const string Collection = "docs_demo";

        // Dossier Data fixe + manifest de dédup (hash -> chemin)
        private const string DataFolder = @"D:\VideoCoursSK\EtapeLocale\04 SkMinimalRagRerankChunk\Data";
        private const string ManifestPath = @"D:\VideoCoursSK\EtapeLocale\04 SkMinimalRagRerankChunk\ingest-manifest.json";

        // Profil de préfixes embeddings (utile pour E5/BGE). Pour MiniLM, laisser vide.
        private record EmbeddingProfile(string Model, string QueryPrefix, string DocumentPrefix);

        // Types utilitaires pour le code
        private record SearchHit(string Text, double Score, string DocId, string DocName, int Page, int ChunkIndex);
        private record RerankRequest(string Query, string[] Candidates, int Top_n = 5);
        private record RerankItem(string Text, double Score);
        private record RerankResponse(string Model, RerankItem[] Items);
        private record IngestManifest(Dictionary<string, string> Docs)
        {
            public static IngestManifest Empty =>
                new IngestManifest(new Dictionary<string, string>());
        }


        public static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;

            // ------- Sélection du modèle LLM -------
            // attention les modèles Ollama doivent être préalablement installés en local
            string? modelFromArgs = null;
            for (int i = 0; i + 1 < args.Length; i++)
            {
                if (args[i].Equals("--model", StringComparison.OrdinalIgnoreCase))
                { modelFromArgs = args[i + 1]; break; }
            }
            if (string.IsNullOrWhiteSpace(modelFromArgs))
            {
                Console.WriteLine("Sélectionnez un modèle :");
                Console.WriteLine(" 1. qwen3");
                Console.WriteLine(" 2. llama3 (défaut)");
                Console.WriteLine(" 3. gpt-oss");
                Console.Write("Votre choix (1-3) : ");
                var choice = Console.ReadLine();
                modelFromArgs = choice switch
                {
                    "1" => "qwen3",
                    "3" => "gpt-oss",
                    _ => "llama3",
                };
            }
            var ollamaModel = modelFromArgs;
            Console.WriteLine($"[Config] Modèle Ollama = {ollamaModel}");
            Console.WriteLine($"[Config] Dossier Data = {DataFolder}");

            // ------- Options rerank -------
            var useRerank = args.Any(a => a.Equals("--rerank", StringComparison.OrdinalIgnoreCase));
            var kVec = 30;
            var rTop = 5;
            for (var i = 0; i + 1 < args.Length; i++)
            {
                if (args[i].Equals("--k", StringComparison.OrdinalIgnoreCase) && int.TryParse(args[i + 1], out var k)) kVec = k;
                if (args[i].Equals("--r", StringComparison.OrdinalIgnoreCase) && int.TryParse(args[i + 1], out var r)) rTop = r;
            }
            if (!useRerank)
            {
                Console.Write("Reranking 0=non, 1=oui (défaut 0) : ");
                var rr = Console.ReadLine();
                if (rr == "1") useRerank = true;
                if (useRerank)
                {
                    Console.Write($"k (défaut {kVec}) : ");
                    var kk = Console.ReadLine(); if (int.TryParse(kk, out var k)) kVec = k;
                    Console.Write($"r (défaut {rTop}) : ");
                    var rr2 = Console.ReadLine(); if (int.TryParse(rr2, out var r)) rTop = r;
                }
            }
            Console.WriteLine($"[Config] Rerank={(useRerank ? "ON" : "OFF")} | k={kVec} | r={rTop}");

            // ------- Profil d'embeddings (MiniLM par défaut => pas de préfixe) -------
            var embeddingProfile = new EmbeddingProfile("all-MiniLM-L6-v2", "", "");
            // Exemples :
            // var embeddingProfile = new EmbeddingProfile("intfloat/multilingual-e5-base", "query: ", "passage: ");
            // var embeddingProfile = new EmbeddingProfile("BAAI/bge-m3",               "query: ", "passage: ");

            // ------- Kernel & services -------
            var builder = Kernel.CreateBuilder();
            builder.Services.AddSingleton(new HttpClient());

            builder.Services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(
                sp => new HttpEmbeddingGenerator(sp.GetRequiredService<HttpClient>(), EmbedBase));

            builder.Services.AddSingleton<IChatCompletionService>(
                sp => new OllamaChatService(sp.GetRequiredService<HttpClient>(), OllamaBase, ollamaModel));

            var kernel = builder.Build();
            var embed = kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
            var chat = kernel.GetRequiredService<IChatCompletionService>();

            // ------- Sanity checks -------
            await SanityCheckAsync(ollamaModel);

            // Dimension embeddings
            var probe = await embed.GenerateAsync(new[] { "probe" });
            var dim = probe[0].Vector.Length;
            Console.WriteLine($"[Embed] Dimension détectée = {dim}");

            // Reset optionnel
            Console.Write("Vider la collection Qdrant avant ingestion ? 0=non, 1=oui (défaut 0) : ");
            var resetAns = Console.ReadLine();
            var doReset = resetAns == "1";

            if (doReset)
            {
                await ResetCollectionAsync(Collection, dim);
                if (File.Exists(ManifestPath)) File.Delete(ManifestPath);
            }
            else
            {
                await EnsureCollectionAsync(Collection, dim);
            }

            // PRUNE si pas reset
            if (!doReset)
            {
                Console.Write("Prune Qdrant (supprimer les docs dont la source manque) ? 0=non, 1=oui (défaut 0) : ");
                var pruneAns = Console.ReadLine();
                if (pruneAns == "1")
                    await PruneMissingAsync(Collection, DataFolder);
            }

            // Ingestion Data (dédup par SHA256, payload riche)
            await IngestFolderWithDedupAsync(Collection, DataFolder, embeddingProfile, embed, chunkChars: 1300, overlapChars: 200, batch: 64);

            // ------- REPL -------
            while (true)
            {
                Console.Write("\n> Question (Entrée pour quitter) : ");
                var question = Console.ReadLine()?.Trim();
                if (string.IsNullOrEmpty(question)) break;

                var q = await embed.GenerateAsync(new[] { embeddingProfile.QueryPrefix + question });
                var qv = q[0].Vector.ToArray();

                var prelim = await SearchWithPayloadAsync(Collection, qv, useRerank ? kVec : rTop);

                Console.WriteLine($"\n[Vectoriel] {(useRerank ? kVec : prelim.Count)} candidats (aperçu) :");
                foreach (var h in prelim.Take(Math.Min(5, prelim.Count)))
                    Console.WriteLine($" - (score {h.Score:F3}) {h.DocName} p.{h.Page}  :: {h.Text}");

                List<SearchHit> pickedHits;
                if (useRerank)
                {
                    var reranked = await RerankAsync(question, prelim.Select(h => h.Text), topN: rTop);
                    pickedHits = PickRerankedHits(prelim, reranked);

                    Console.WriteLine("\n[Rerank] Passages retenus :");
                    foreach (var h in pickedHits)
                        Console.WriteLine($" - (score {h.Score:F3}) {h.DocName} p.{h.Page}  :: {h.Text}");
                }
                else
                {
                    pickedHits = prelim.Take(rTop).ToList();
                    Console.WriteLine("\n[TopK] Passages retenus (sans rerank) :");
                    foreach (var h in pickedHits)
                        Console.WriteLine($" - (score {h.Score:F3}) {h.DocName} p.{h.Page}  :: {h.Text}");
                }

                // Construction prompt chat avec contexte et instructions système
                var history = new ChatHistory();
                history.AddSystemMessage("Tu es un assistant technique francophone. Style: clair, professionnel, concis, sans anglicismes ni tutoiement.");
                history.AddUserMessage(
                    "Contexte :\n- " + string.Join("\n- ", pickedHits.Select(x => x.Text)) +
                    $"\n\nQuestion : {question}\n" +
                    "Réponds brièvement en t’appuyant uniquement sur le contexte.");

                // Appel chat et affichage
                var messages = await chat.GetChatMessageContentsAsync(history);
                Console.WriteLine("\n[SK] Réponse :\n" + messages[0].Content);

                // Affichage sources (max 6)
                Console.WriteLine("\n[Sources]");
                foreach (var g in pickedHits.GroupBy(h => (DocName: h.DocName, h.Page)).Take(6))
                    Console.WriteLine($" - [{g.Key.DocName} p.{g.Key.Page}]");
            }
        }

        // =================== Helpers statiques ===================

        private static async Task SanityCheckAsync(string ollamaModel)
        {
            using var http = new HttpClient();
            try
            {
                var r1 = await http.GetAsync($"{EmbedBase}/ready");
                Console.WriteLine(r1.IsSuccessStatusCode ? "[Check] Embeddings OK" : "[Check] Embeddings: /ready KO");

                var r2 = await http.GetAsync($"{QdrantBase}/");
                Console.WriteLine(r2.IsSuccessStatusCode ? "[Check] Qdrant OK" : "[Check] Qdrant KO");

                var mini = new { model = ollamaModel, prompt = "ping", stream = false };
                var r3 = await http.PostAsJsonAsync($"{OllamaBase}/api/generate", mini);
                Console.WriteLine(r3.IsSuccessStatusCode ? "[Check] Ollama OK" : "[Check] Ollama KO");
            }
            catch (Exception ex) { Console.WriteLine("[Check] " + ex.Message); }
        }

        private static async Task EnsureCollectionAsync(string name, int size)
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

        private static async Task ResetCollectionAsync(string name, int size)
        {
            using var http = new HttpClient();

            var del = await http.DeleteAsync($"{QdrantBase}/collections/{name}");
            if (del.IsSuccessStatusCode)
                Console.WriteLine($"[Qdrant] Collection '{name}' supprimée.");
            else if ((int)del.StatusCode != 404)
                del.EnsureSuccessStatusCode();

            var body = new { vectors = new { size, distance = "Cosine" } };
            var create = await http.PutAsJsonAsync($"{QdrantBase}/collections/{name}", body);
            create.EnsureSuccessStatusCode();
            Console.WriteLine($"[Qdrant] Collection '{name}' recréée (size={size}).");
        }

        private static IngestManifest LoadManifest(string path)
        {
            try
            {
                if (!File.Exists(path)) return IngestManifest.Empty;
                var json = File.ReadAllText(path);
                return JsonSerializer.Deserialize<IngestManifest>(json) ?? IngestManifest.Empty;
            }
            catch { return IngestManifest.Empty; }
        }

        private static void SaveManifest(string path, IngestManifest m)
        {
            var json = JsonSerializer.Serialize(m, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, json, Encoding.UTF8);
        }

        private static string ComputeFileSha256(string path)
        {
            using var sha = SHA256.Create();
            using var fs = File.OpenRead(path);
            var hash = sha.ComputeHash(fs);
            return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
        }

        private static async Task IngestFolderWithDedupAsync(
            string collectionName,
            string folder,
            EmbeddingProfile profile,
            IEmbeddingGenerator<string, Embedding<float>> embed,
            int chunkChars = 1200,
            int overlapChars = 200,
            int batch = 64)
        {
            if (!Directory.Exists(folder))
            {
                Console.WriteLine($"[Ingest] Dossier inexistant : {folder}");
                return;
            }

            var manifest = LoadManifest(ManifestPath);

            // 1) INVENTAIRE + HASH avec barre de progression
            var allPaths = Directory.EnumerateFiles(folder, "*.*", SearchOption.AllDirectories)
                .Where(p => p.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".docx", StringComparison.OrdinalIgnoreCase))
                .ToList();

            var files = new List<(string path, string hash)>(allPaths.Count);
            using (var prog = new ConsoleProgress("Scan & hash fichiers", allPaths.Count))
            {
                foreach (var p in allPaths)
                {
                    var hash = ComputeFileSha256(p);
                    files.Add((p, hash));
                    prog.Report();
                }
            }

            var toProcess = files.Where(f => !manifest.Docs.ContainsKey(f.hash)).ToList();
            if (toProcess.Count == 0)
            {
                Console.WriteLine("[Ingest] Rien à faire (tous les documents présents sont déjà indexés).");
                return;
            }

            Console.WriteLine($"[Ingest] {toProcess.Count} nouveau(x) document(s) à indexer sur {files.Count}.");

            // 2) CHUNKING par fichier avec progression (par fichier, le #chunks est variable)
            var loaded = new List<DocumentLoader.LoadedChunk>(capacity: 1024);
            using (var prog = new ConsoleProgress("Chunking documents", toProcess.Count))
            {
                foreach (var f in toProcess)
                {
                    var chunks = await DocumentLoader.LoadAndChunkFileAsync(f.path, f.hash, chunkChars, overlapChars);
                    loaded.AddRange(chunks);
                    prog.Report();
                }
            }

            // -- FILTRAGE : supprime les chunks vides/blank avant tout embedding --
            int before = loaded.Count;
            loaded = loaded.Where(c => !string.IsNullOrWhiteSpace(c.Text)).ToList();
            int removed = before - loaded.Count;
            if (removed > 0)
                Console.WriteLine($"[Ingest] {removed} chunk(s) vides ignorés avant encodage.");


            Console.WriteLine($"[Ingest] {loaded.Count} chunks à encoder.");

            if (loaded.Count == 0)
            {
                Console.WriteLine("[Ingest] Aucune donnée texte exploitable (chunks=0).");
                // Met quand même à jour le manifest pour éviter de re-traiter ces fichiers vides.
                foreach (var f in toProcess) manifest.Docs[f.hash] = f.path;
                SaveManifest(ManifestPath, manifest);
                return;
            }

            // 3) EMBEDDINGS + UPSERT par batch avec progression
            using var http = new HttpClient();
            using (var prog = new ConsoleProgress("Embeddings + Upsert Qdrant", loaded.Count))
            {
                for (int i = 0; i < loaded.Count; i += batch)
                {
                    var slice = loaded.Skip(i).Take(batch).ToArray();

                    // Sécurité batch : retire tout chunk vide résiduel (ne devrait plus arriver avec le filtrage global)
                    var clean = slice.Where(s => !string.IsNullOrWhiteSpace(s.Text)).ToArray();
                    if (clean.Length != slice.Length)
                        Console.WriteLine($"[Ingest] (batch {i}-{i + slice.Length - 1}) {slice.Length - clean.Length} chunk(s) vides ignorés.");

                    slice = clean;
                    if (slice.Length == 0) { prog.Report(0); continue; } // rien à encoder dans ce batch


                    var texts = slice.Select(s => profile.DocumentPrefix + s.Text).ToArray();

                    // Défense en profondeur (ne devrait plus arriver) :
                    if (texts.Any(string.IsNullOrWhiteSpace))
                        throw new InvalidOperationException($"Chunk vide détecté juste avant embedding (batch {i}).");

                    var gens = await embed.GenerateAsync(texts);
                    var vectors = gens.Select(e => e.Vector.ToArray()).ToArray();

                    var points = new List<object>(slice.Length);
                    for (int j = 0; j < slice.Length; j++)
                    {
                        var s = slice[j];
                        var payload = new Dictionary<string, object>
                        {
                            ["text"] = s.Text,
                            ["doc_id"] = s.DocId,   // == SHA256 du fichier
                            ["doc_name"] = s.DocName,
                            ["page"] = s.Page,
                            ["chunk_index"] = s.Index
                        };

                        var rawKey = $"{s.DocId}:{s.Page}:{s.Index}";
                        var pointId = CreateDeterministicGuid(rawKey);
                        points.Add(new { id = pointId, vector = vectors[j], payload });
                    }

                    var upsert = new { points, wait = true };
                    var res = await http.PutAsJsonAsync($"{QdrantBase}/collections/{collectionName}/points", upsert);
                    if (!res.IsSuccessStatusCode)
                    {
                        var err = await res.Content.ReadAsStringAsync();
                        throw new HttpRequestException($"Qdrant upsert failed: {(int)res.StatusCode} {res.ReasonPhrase} :: {err}");
                    }
                    res.EnsureSuccessStatusCode();

                    prog.Report(slice.Length);
                }
            }

            // 4) MAJ MANIFEST
            foreach (var f in toProcess)
                manifest.Docs[f.hash] = f.path;
            SaveManifest(ManifestPath, manifest);

            Console.WriteLine("[Ingest] Terminé (manifest mis à jour).");
        }


        private static async Task PruneMissingAsync(string collectionName, string folder)
        {
            var manifest = LoadManifest(ManifestPath);
            if (manifest.Docs.Count == 0)
            {
                Console.WriteLine("[Prune] Aucun manifest trouvé — rien à faire.");
                return;
            }

            var presentHashes = Directory.Exists(folder)
                ? Directory.EnumerateFiles(folder, "*.*", SearchOption.AllDirectories)
                    .Where(p => p.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".docx", StringComparison.OrdinalIgnoreCase))
                    .Select(ComputeFileSha256)
                    .ToHashSet()
                : new HashSet<string>();

            var staleDocIds = manifest.Docs.Keys.Where(h => !presentHashes.Contains(h)).ToList();
            if (staleDocIds.Count == 0)
            {
                Console.WriteLine("[Prune] Rien à supprimer — toutes les sources sont présentes.");
                return;
            }

            Console.WriteLine($"[Prune] {staleDocIds.Count} doc_id à supprimer de Qdrant (sources absentes).");

            const int batchSize = 256;
            using var http = new HttpClient();
            using (var prog = new ConsoleProgress("Suppression Qdrant", staleDocIds.Count))
            {
                for (int i = 0; i < staleDocIds.Count; i += batchSize)
                {
                    var batch = staleDocIds.Skip(i).Take(batchSize).ToArray();
                    var body = new
                    {
                        filter = new
                        {
                            must = new object[]
                            {
                        new { key = "doc_id", match = new { @any = batch } }
                            }
                        },
                        wait = true
                    };

                    var res = await http.PostAsJsonAsync($"{QdrantBase}/collections/{collectionName}/points/delete", body);
                    res.EnsureSuccessStatusCode();

                    prog.Report(batch.Length);
                }
            }

            foreach (var id in staleDocIds) manifest.Docs.Remove(id);
            SaveManifest(ManifestPath, manifest);

            Console.WriteLine("[Prune] Terminé (Qdrant & manifest synchronisés).");
        }


        private static async Task<List<SearchHit>> SearchWithPayloadAsync(string name, float[] vector, int k)
        {
            using var http = new HttpClient();
            var req = new { vector, limit = k, with_payload = true };
            var res = await http.PostAsJsonAsync($"{QdrantBase}/collections/{name}/points/search", req);
            res.EnsureSuccessStatusCode();

            var json = await res.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);

            var list = new List<SearchHit>();
            foreach (var item in doc.RootElement.GetProperty("result").EnumerateArray())
            {
                var score = item.GetProperty("score").GetDouble();
                var payload = item.GetProperty("payload");

                var text = payload.TryGetProperty("text", out var pText) ? pText.GetString() ?? "" : "";
                var docId = payload.TryGetProperty("doc_id", out var pId) ? pId.GetString() ?? "" : "";
                var docName = payload.TryGetProperty("doc_name", out var pName) ? pName.GetString() ?? "" : "";
                var page = payload.TryGetProperty("page", out var pPage) ? pPage.GetInt32() : 0;
                var chunkIndex = payload.TryGetProperty("chunk_index", out var pIdx) ? pIdx.GetInt32() : 0;

                list.Add(new SearchHit(text, score, docId, docName, page, chunkIndex));
            }
            return list;
        }

        private static async Task<List<(string text, double score)>> RerankAsync(string query, IEnumerable<string> candidates, int topN = 5)
        {
            using var http = new HttpClient();
            http.Timeout = TimeSpan.FromSeconds(120);
            var req = new RerankRequest(query, candidates.ToArray(), topN);
            var res = await http.PostAsJsonAsync("http://localhost:5001/rerank", req);
            res.EnsureSuccessStatusCode();

            var obj = await res.Content.ReadFromJsonAsync<RerankResponse>()
                      ?? throw new InvalidOperationException("Réponse /rerank vide");
            return obj.Items.Select(i => (text: i.Text, score: i.Score)).ToList();
        }

        private static List<SearchHit> PickRerankedHits(List<SearchHit> prelim, List<(string text, double score)> reranked)
        {
            var map = prelim.GroupBy(h => h.Text)
                            .ToDictionary(g => g.Key, g => new Queue<SearchHit>(g));

            var picked = new List<SearchHit>();
            foreach (var (t, s) in reranked)
            {
                if (map.TryGetValue(t, out var q) && q.Count > 0)
                {
                    var h = q.Dequeue();
                    picked.Add(h with { Score = s });
                }
                else
                {
                    picked.Add(new SearchHit(t, s, "", "", 0, 0));
                }
            }
            return picked;
        }

        // Progress bar console avec ETA (usage: using var p = new ConsoleProgress("Titre", total); p.Report(step);)
        private sealed class ConsoleProgress : IDisposable
        {
            private readonly string _label;
            private readonly int _total;
            private readonly int _barWidth;
            private readonly DateTime _start;
            private int _done;
            private int _lastLen;

            public ConsoleProgress(string label, int total, int barWidth = 40)
            {
                _label = label;
                _total = Math.Max(0, total);
                _barWidth = Math.Max(10, barWidth);
                _start = DateTime.UtcNow;
                _done = 0;
                _lastLen = 0;
                Console.CursorVisible = false;
                Draw();
            }

            public void Report(int increment = 1)
            {
                _done = Math.Clamp(_done + increment, 0, _total);
                Draw();
            }

            private void Draw()
            {
                var pct = _total == 0 ? 1.0 : (double)_done / _total;
                var filled = (int)Math.Round(pct * _barWidth);
                var bar = new string('#', filled) + new string('-', _barWidth - filled);

                var elapsed = DateTime.UtcNow - _start;
                TimeSpan eta;
                if (_done <= 0 || _total <= 0)
                    eta = TimeSpan.Zero;
                else
                {
                    var rate = elapsed.TotalSeconds / _done; // sec / item
                    var remain = Math.Max(0, _total - _done) * rate;
                    eta = TimeSpan.FromSeconds(remain);
                }

                string line = $"{_label} [{bar}] {_done}/{_total} ({pct * 100:0.0}%)  ETA {Format(eta)}";
                // effacer l’ancienne ligne si plus courte
                int pad = Math.Max(0, _lastLen - line.Length);
                Console.Write("\r" + line + new string(' ', pad));
                _lastLen = line.Length;
            }

            public void Complete()
            {
                _done = _total;
                Draw();
                Console.WriteLine(); // nouvelle ligne finale
            }

            public void Dispose()
            {
                Complete();
                Console.CursorVisible = true;
            }

            private static string Format(TimeSpan t)
            {
                if (t.TotalHours >= 1) return $"{(int)t.TotalHours:D2}:{t.Minutes:D2}:{t.Seconds:D2}";
                return $"{t.Minutes:D2}:{t.Seconds:D2}";
            }
        }

        private static Guid CreateDeterministicGuid(string input)
        {
            // RFC 4122 version 5 (SHA-1 sur le nom) — simple et suffisant ici
            using var sha1 = SHA1.Create();
            var nameBytes = Encoding.UTF8.GetBytes(input);
            var hash = sha1.ComputeHash(nameBytes);

            // Prend les 16 premiers octets et impose version/variant RFC
            var bytes = new byte[16];
            Array.Copy(hash, 0, bytes, 0, 16);
            bytes[6] = (byte)((bytes[6] & 0x0F) | 0x50); // version 5
            bytes[8] = (byte)((bytes[8] & 0x3F) | 0x80); // variant RFC 4122

            return new Guid(bytes);
        }


    }
}

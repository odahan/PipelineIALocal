// DocumentLoader.cs
// Extraction PDF/DOCX + normalisation + découpage en chunks avec overlap.

using System.Text;
using System.Text.RegularExpressions;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

namespace SkMinimalRagRerankChunk;

public static class DocumentLoader
{
    public sealed record LoadedChunk(string DocId, string DocName, int Page, int Index, string Text);

    // Charge un dossier complet (.pdf/.docx) et découpe en chunks
    public static async Task<List<LoadedChunk>> LoadAndChunkFolderAsync(
        string folder,
        int chunkChars = 1200,
        int overlapChars = 200,
        CancellationToken ct = default)
    {
        var chunks = new List<LoadedChunk>();

        foreach (var path in Directory.EnumerateFiles(folder, "*.*", SearchOption.AllDirectories)
                     .Where(p => p.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase)
                              || p.EndsWith(".docx", StringComparison.OrdinalIgnoreCase)))
        {
            // Pour la version "folder", DocId est un GUID (non stable) — on préfère la version "file" ci-dessous avec un docId passé en param (ex: SHA256).
            var docId = Guid.NewGuid().ToString();
            var docName = Path.GetFileName(path);

            if (path.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
            {
                await foreach (var (page, text) in ReadPdfPagesAsync(path, ct))
                {
                    var idx = 0;
                    foreach (var chunk in Chunk(text, chunkChars, overlapChars))
                        chunks.Add(new LoadedChunk(docId, docName, page, idx++, chunk));
                }
            }
            else
            {
                var full = ReadDocxText(path);
                var clean = Normalize(full);
                var idx = 0;
                foreach (var chunk in Chunk(clean, chunkChars, overlapChars))
                    chunks.Add(new LoadedChunk(docId, docName, 1, idx++, chunk));
            }
        }

        return chunks;
    }

    //  Charge un fichier unique avec un DocId fourni (ex: SHA256 du fichier)
    public static async Task<List<LoadedChunk>> LoadAndChunkFileAsync(
        string path,
        string docId,
        int chunkChars = 1200,
        int overlapChars = 200,
        CancellationToken ct = default)
    {
        var chunks = new List<LoadedChunk>();
        var docName = Path.GetFileName(path);

        if (path.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
        {
            await foreach (var (page, text) in ReadPdfPagesAsync(path, ct))
            {
                var idx = 0;
                foreach (var chunk in Chunk(text, chunkChars, overlapChars))
                    chunks.Add(new LoadedChunk(docId, docName, page, idx++, chunk));
            }
        }
        else // .docx
        {
            var full = ReadDocxText(path);
            var clean = Normalize(full);
            var idx = 0;
            foreach (var chunk in Chunk(clean, chunkChars, overlapChars))
                chunks.Add(new LoadedChunk(docId, docName, 1, idx++, chunk));
        }

        return chunks;
    }

    private static async IAsyncEnumerable<(int page, string text)> ReadPdfPagesAsync(
        string path,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await Task.Yield();
        using var pdf = PdfDocument.Open(path);
        foreach (Page page in pdf.GetPages())
        {
            ct.ThrowIfCancellationRequested();
            string text;
            try
            {
                // Extraction “en ordre de lecture”
                text = ContentOrderTextExtractor.GetText(page);
            }
            catch
            {
                // Fallback si l’extracteur n’est pas dispo selon la version
                var sb = new StringBuilder();
                foreach (var w in page.GetWords()) sb.Append(w.Text).Append(' ');
                text = sb.ToString();
            }
            yield return (page.Number, Normalize(text));
        }
    }

    private static string ReadDocxText(string path)
    {
        using var doc = WordprocessingDocument.Open(path, false);
        var body = doc.MainDocumentPart?.Document?.Body;
        if (body is null) return string.Empty;

        var sb = new StringBuilder();
        foreach (var p in body.Descendants<Paragraph>())
        {
            var text = string.Concat(p.Descendants<Text>().Select(t => t.Text));
            if (!string.IsNullOrWhiteSpace(text)) sb.AppendLine(text);
        }
        return sb.ToString();
    }

    private static string Normalize(string s)
    {
        // Nettoyage léger : dé-césure, espaces, guillemets typographiques
        s = s.Replace("-\n", "");
        s = s.Replace("\r", " ").Replace("\n", " ");
        s = Regex.Replace(s, @"\s+", " ").Trim();
        s = s.Replace('’', '\'').Replace('“', '\"').Replace('”', '\"');
        return s;
    }

    private static IEnumerable<string> Chunk(string text, int size, int overlap)
    {
        if (string.IsNullOrWhiteSpace(text)) yield break;

        var start = 0;
        var len = text.Length;
        var step = Math.Max(1, size - overlap);

        while (start < len)
        {
            var take = Math.Min(size, len - start);
            var slice = text.AsSpan(start, take);

            // Coupe au plus proche d’un séparateur pour éviter de tronquer en plein mot
            int cut = take;
            for (int i = take - 1; i >= Math.Max(0, take - 80); i--)
            {
                var ch = slice[i];
                if (char.IsWhiteSpace(ch) || ch == '.' || ch == ';' || ch == '!' || ch == '?')
                { cut = i + 1; break; }
            }

            yield return slice[..cut].ToString().Trim();
            start += step;
        }
    }
}

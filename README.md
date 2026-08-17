# 💬 Ask Sumit Yadav — an in-browser RAG chatbot

A retrieval-augmented chat interface that answers questions about Sumit Yadav's
research, experience and projects. Both models run **entirely in the visitor's
browser** — no server, no API keys, no telemetry. Live at
**[llm.sumityadav.com.np](https://llm.sumityadav.com.np)**.

![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Runs](https://img.shields.io/badge/Inference-100%25%20client--side-blue)
![Models](https://img.shields.io/badge/Models-ONNX%20%C2%B7%20129%20MB-purple)

Ask it *"what does he do at Astha.ai?"*, *"tell me more"*, *"what awards has he
won?"* — it answers in prose and shows the exact bio sections each answer came
from.

---

## How it works

```
bio.txt
  │  heading-aware chunking (36 sections)
  ▼
┌─────────────────────────────┐        ┌──────────────────────────────┐
│ Dense retrieval             │        │ Lexical retrieval            │
│ gte-small embeddings        │   +    │ BM25 + light stemming        │
│ (cosine over 384-dim)       │        │ + query-vocabulary bridging  │
└──────────────┬──────────────┘        └──────────────┬───────────────┘
               └────────── reciprocal rank fusion ────┘
                                  │  top 3 sections
                                  ▼
                    relevance gate (decline / hedge / answer)
                                  │
                                  ▼
                   LaMini-Flan-T5-77M composes an answer
                                  │
                                  ▼
                   quality gate → extractive fallback
                                  │
                                  ▼
                      answer + "from N sections" citations
```

### Retrieval

Single-vector nearest-neighbour search is not enough on a document this small and
this proper-noun-heavy, so retrieval is **hybrid**:

- **Heading-aware chunking.** `bio.txt` is split on its `##`/`###` structure, and
  each chunk carries its heading path in its searchable text — a chunk retrieved
  in isolation still states what it is about, which matters because the generator
  only ever sees the chunk, never the whole document.
- **Dense + lexical, fused by rank.** Embeddings catch paraphrase; BM25 catches
  exact names like *Qdrant*, *SAFE-MCP* or *maiBERT* that embeddings blur.
  Reciprocal rank fusion combines them, because bounded cosine and unbounded BM25
  scores are not comparable — only their orderings are.
- **Query-vocabulary bridging.** Visitors ask *"where does he work"*; the bio says
  *"Professional Experience"*. A pattern table maps everyday phrasing onto the
  document's vocabulary so the lexical half actually scores.
- **Follow-up rewriting.** *"tell me more"* carries no retrievable content, so
  short elliptical questions inherit the previous turn's topic.
- **A relevance gate** declines off-topic questions instead of confidently
  answering from an unrelated section.

### Answer composition, and why there is a fallback

The generator is a 77M-parameter model, which is small enough to be unreliable.
Its failure modes are specific and recognisable: refusing ("the context does not
provide…") even though retrieval proved the context *does* contain the answer,
echoing the prompt instructions back as content, or looping a phrase.

So generated answers are **validated before display**, and anything degenerate is
replaced by an **extractive answer** — the sentences from the retrieved sections
that best match the question, stitched in original order. That path cannot
hallucinate, because every word is copied from `bio.txt`. Structured sections
(contact details, the skills list) are rendered as themselves, since a contact
block *is* the answer and paraphrasing it only loses the links.

Because answers are validated, token-by-token streaming is deliberately not used
— streaming would put text on screen that then has to be retracted.

Every answer shows its sources, so any claim can be checked against the bio.

---

## Models

| Role | Model | Size (int8 ONNX) |
|---|---|---|
| Retrieval | [`Xenova/gte-small`](https://huggingface.co/Xenova/gte-small) | ~34 MB |
| Answer composition | [`Xenova/LaMini-Flan-T5-77M`](https://huggingface.co/Xenova/LaMini-Flan-T5-77M) | ~95 MB (35 encoder + 59 decoder) |

Both are fetched from the Hugging Face CDN and cached by the browser's Cache API,
so the ~129 MB download is paid once per visitor. Chunk embeddings are cached in
`localStorage`, keyed by a hash of `bio.txt`, so repeat visits skip re-indexing.

**Why this generator?** It is close to the largest instruction-tuned seq2seq model
that fits a ~100 MB budget and actually loads under WebAssembly. `fp16` and
`q4f16` builds — including SmolLM2-135M-Instruct — fail to create an inference
session in onnxruntime-web, so `int8` is the only working quantization here.

> ⚠️ **The same weights are noticeably weaker in the browser than in Node.**
> Identical prompts that answer correctly under native ONNX Runtime regularly come
> back from WebAssembly int8 kernels as refusals or loops. This is measured, not
> theoretical, and it is why the quality gate and extractive fallback exist. If
> you benchmark generation quality, benchmark it in a browser (`npm run smoke`) —
> Node results are optimistic.

---

## Running locally

ES modules and `fetch('bio.txt')` need a web server — opening `index.html`
directly will not work.

```bash
npm run serve         # http://localhost:3000
```

## Tests

Retrieval logic lives in `rag.js`, deliberately dependency-free and DOM-free, so
the exact code the browser runs can be tested in Node.

```bash
npm install
npm test              # retrieval routing, thresholds, answer policy, generation
```

`npm test` checks that each question still resolves to the section you expect,
that off-topic questions stay below the relevance floor, that follow-ups inherit
the right topic while self-contained questions do not, and that degenerate
generations are rejected. **Run it after editing `bio.txt`.**

For the full page in a real browser:

```bash
npm run browsers      # one-time: install Chromium
npm run smoke         # boots the page, asks questions, asserts no bad answers
```

---

## Updating the content

1. Edit **`bio.txt`**. Use `##` for major sections and `###` for entries within
   them; the chunker follows that structure, and each `##`/`###` block becomes an
   independently retrievable section.
2. Write sections so they stand alone — a chunk is retrieved without its
   neighbours, so name the subject rather than relying on "he" from three
   sections earlier.
3. Run `npm test`. If a question now routes to the wrong section, add the missing
   vocabulary to `EXPANSIONS` in `rag.js` rather than contorting the prose.
4. Visitors' cached embeddings invalidate automatically — the cache key is a hash
   of `bio.txt`.

### Tuning

All knobs live in `CONFIG` at the top of `rag.js`. Two notes:

- `confidentAt` / `weakAt` are **corpus-specific**, measured with `npm test`:
  on-topic questions score 0.824–0.921 here and clearly off-topic ones 0.732–0.772.
  Re-measure if you change `embedModel` or substantially rewrite `bio.txt`.
- `maxContextChars` must leave room inside the T5 encoder's 512-token limit.

---

## Project structure

```
index.html      chat UI, model loading, embedding cache
rag.js          retrieval + answer-selection core (pure, testable)
bio.txt         the knowledge base — the only file you normally edit
test/           Node test for the retrieval core
smoke.mjs       headless-browser end-to-end test
models/         a local copy of gte-small (unused: models load from the CDN)
```

## Privacy

Questions never leave the browser. There is no backend, no logging and no
analytics; the only network requests are the one-time model downloads from the
Hugging Face CDN.

## Browser support

Chrome 88+, Firefox 87+, Safari 14+, Edge 88+ — anything with ES modules and
WebAssembly. GitHub Pages does not send cross-origin isolation headers, so WASM
runs single-threaded; the models are small enough that this is fine.

## License

MIT — see [LICENSE](LICENSE).

---

**Built by [Sumit Yadav](https://sumityadav.com.np)** · retrieval and generation by
[Transformers.js](https://github.com/huggingface/transformers.js)

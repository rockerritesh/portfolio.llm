/**
 * rag.js — the retrieval core of the portfolio chatbot.
 *
 * Deliberately dependency-free and DOM-free: every function here is pure, so
 * the same code that runs in the browser can be exercised directly in Node
 * (see `test/retrieval.test.mjs`). Model loading and UI live in index.html.
 */

export const CONFIG = {
    // ~34 MB int8 bi-encoder — dense retrieval.
    embedModel: 'Xenova/gte-small',
    // ~95 MB int8 (35 MB encoder + 59 MB decoder) — instruction-tuned seq2seq
    // that composes the final answer from retrieved context.
    genModel: 'Xenova/LaMini-Flan-T5-77M',
    topK: 3,
    // LaMini-Flan-T5 is a T5: its encoder caps at 512 tokens. The prompt
    // scaffold plus the question costs ~70, so keep context inside the rest.
    maxContextChars: 1250,
    maxNewTokens: 150,
    // Reciprocal-rank-fusion damping constant (the standard default).
    rrfK: 60,
    // gte-small has a compressed cosine range, so these are not generic
    // thresholds — they were measured against this corpus with test/retrieval.test.mjs.
    // Across 18 on-topic questions the best chunk scored 0.824–0.921; across
    // clearly off-topic ones it scored 0.732–0.772. That leaves a real gap, so
    // `weakAt` sits inside it: below it we decline instead of answering from an
    // unrelated section, and the band up to `confidentAt` answers with a visible
    // caveat. Re-run the test and re-measure if embedModel or bio.txt changes
    // substantially — these numbers are corpus-specific, not universal.
    confidentAt: 0.82,
    weakAt: 0.79,
    chunkCharLimit: 900,
    cacheKey: 'portfolio-llm-emb-v1',
};

/* ============================================================
   Markdown → prose
   ============================================================ */

/**
 * Strip markdown to clean prose: link text without URLs, no emphasis markers,
 * no emoji. URLs are pure token waste for both the embedder and the T5 encoder,
 * and they badly pollute lexical matching (every `https` and `github` collides).
 */
export function toPlain(md) {
    return md
        .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
        .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
        .replace(/`([^`]*)`/g, '$1')
        // Unescape markdown escapes before stripping emphasis, so "CORE A\*"
        // becomes "CORE A" rather than leaving a stray backslash behind.
        .replace(/\\([*_#\\`[\]()])/g, '$1')
        .replace(/[*_#>]/g, '')
        .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/gu, '')
        .replace(/^[ \t]*[-•]\s*/gm, '')
        .replace(/[ \t]{2,}/g, ' ')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

const cleanHeading = (s) => toPlain(s).replace(/\s+/g, ' ').trim();

/** Split an over-long section on paragraph/bullet boundaries. */
function splitLong(body, limit) {
    if (body.length <= limit) return [body];
    const blocks = body.split(/\n(?=\s*[-*]\s|\s*\n)/).filter((b) => b.trim());
    const out = [];
    let buf = '';
    for (const b of blocks) {
        if (buf && (buf + '\n' + b).length > limit) { out.push(buf.trim()); buf = b; }
        else buf = buf ? buf + '\n' + b : b;
    }
    if (buf.trim()) out.push(buf.trim());
    return out.length ? out : [body];
}

/**
 * Does this chunk read as sentences, or as a label/link dump?
 *
 * Sections like "Identity & Contact" are lists of `Label: value` lines with no
 * sentence structure. They are legitimate answers to "how do I contact him?",
 * but as *supporting* context they are actively harmful: the generator treats
 * the bag of proper nouns as prose. In testing this produced an answer claiming
 * Sumit "worked with organizations such as Kaggle, Affiliation, and Google
 * Scholar" — all three lifted from contact-section labels. Flagging these lets
 * buildContext admit them only when they are the primary hit.
 */
function isProse(plain) {
    const lines = plain.split('\n').filter((l) => l.trim()).length || 1;
    const sentences = (plain.match(/[.!?](\s|$)/g) || []).length;
    return sentences / lines >= 0.4;
}

/**
 * Heading-aware chunking. Each chunk carries its heading path folded into the
 * searchable text, so a chunk retrieved in isolation still states what it is
 * about — which matters because the generator only ever sees the chunk, never
 * the surrounding document.
 */
export function chunkBio(text, limit = CONFIG.chunkCharLimit) {
    const chunks = [];
    let h2 = '', h3 = '', buf = [];

    const flush = () => {
        const body = buf.join('\n').trim();
        buf = [];
        if (!body || !(h2 || h3)) return;
        const title = h3 && h2 ? `${h2} › ${h3}` : (h3 || h2);
        const heading = h3 ? `${h2}. ${h3}` : h2;
        const parts = splitLong(body, limit);
        parts.forEach((part, i) => {
            const plain = toPlain(part);
            chunks.push({
                // Long sections split into several chunks would otherwise show
                // up as identical entries in the sources list.
                title: parts.length > 1 ? `${title} (${i + 1}/${parts.length})` : title,
                markdown: (h3 ? `### ${h3}\n\n` : '') + part,
                plain: `${heading}. ${plain}`,
                prose: isProse(plain),
            });
        });
    };

    for (const line of text.split('\n')) {
        const h3m = line.match(/^###\s+(.+)$/);
        const h2m = line.match(/^##\s+(.+)$/);
        const h1m = line.match(/^#\s+(.+)$/);
        if (h3m) { flush(); h3 = cleanHeading(h3m[1]); continue; }
        if (h2m) { flush(); h2 = cleanHeading(h2m[1]); h3 = ''; continue; }
        if (h1m) { flush(); h2 = ''; h3 = ''; continue; }
        if (/^-{3,}$/.test(line.trim())) continue;
        buf.push(line);
    }
    flush();
    return chunks;
}

/* ============================================================
   Lexical retrieval — BM25 with light stemming
   ============================================================ */

const STOP = new Set(('a an and are as at be by do does did for from has have he her his how i in is it its me my of on or '
    + 'that the their them there they this to was were what when where which who whom whose why will with you your about tell '
    + 'give show can could would should any some more most').split(' '));

const SUFFIXES = ['ational', 'ization', 'iveness', 'ments', 'ingly', 'ities', 'ical', 'ment', 'ness', 'ing', 'ers', 'ies', 'ed', 'es', 'ly', 's'];

/** Crude suffix stripping — enough to bridge publication/publications,
 *  researcher/research, steering/steer without pulling in a real stemmer. */
export function stem(w) {
    if (w.length <= 4) return w;
    for (const s of SUFFIXES) {
        if (w.endsWith(s) && w.length - s.length >= 4) {
            const base = w.slice(0, -s.length);
            return s === 'ies' ? base + 'y' : base;
        }
    }
    return w;
}

export function tokenize(text) {
    return (text.toLowerCase().match(/[a-z0-9][a-z0-9.+#-]*/g) || [])
        .map((t) => t.replace(/^[.+-]+|[.+-]+$/g, ''))
        .filter((t) => t && t.length > 1 && !STOP.has(t))
        .map(stem);
}

export function buildLexicon(docs) {
    const tfs = [], df = new Map();
    let total = 0;
    for (const d of docs) {
        const toks = tokenize(d);
        total += toks.length;
        const tf = new Map();
        for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
        for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
        tfs.push({ tf, len: toks.length });
    }
    return { tfs, df, avgLen: total / (docs.length || 1), N: docs.length };
}

export function bm25(lex, queryTokens, k1 = 1.5, b = 0.75) {
    const { tfs, df, avgLen, N } = lex;
    return tfs.map(({ tf, len }) => {
        let score = 0;
        for (const q of queryTokens) {
            const f = tf.get(q);
            if (!f) continue;
            const n = df.get(q) || 0;
            const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (len / avgLen)));
        }
        return score;
    });
}

/* ============================================================
   Query understanding — vocabulary bridging + follow-ups
   ============================================================ */

/**
 * Visitors ask in their own words ("where does he work", "any awards?"), which
 * often shares no vocabulary with the bio. These patterns map everyday phrasing
 * onto the terms the document actually uses — this is what makes the lexical
 * half of the hybrid search land instead of scoring zero.
 */
export const EXPANSIONS = [
    [/\b(job|jobs|work|works|working|worked|employ\w*|career|company|companies|workplace|position|role|roles|currently)\b/, 'professional experience astha amnil gradeup engineer researcher current role'],
    [/\b(paper|papers|publication\w*|publish\w*|article|articles|arxiv|acl|conference|preprint|cite|citation\w*)\b/, 'publications acl paper arxiv preprint conference'],
    [/\b(school|college|university|degree|study|studied|studies|education\w*|graduat\w*|campus|thesis|gpa|academic)\b/, 'education pulchowk tribhuvan bachelor computer engineering thesis coursework'],
    [/\b(skill\w*|tech|technolog\w*|stack|language|languages|tool|tools|framework\w*|library|libraries|program\w*)\b/, 'technical skills python pytorch tensorflow languages frameworks databases'],
    [/\b(contact|email|mail|reach|hire|hiring|recruit\w*|phone|mobile|number|linkedin|github|resume|cv|scholar|twitter|website)\b/, 'identity contact email mobile links resume cv linkedin github scholar portfolio'],
    [/\b(award\w*|prize|prizes|win|won|winner|honor\w*|honour\w*|achievement\w*|hackathon\w*|competition\w*|recognition)\b/, 'honors awards winner runner-up best paper hackathon'],
    [/\b(project\w*|built|build|building|made|make|portfolio|demo|demos|open.?source|repo\w*)\b/, 'projects maibert agentguard vibe-coder mcp server demo'],
    [/\b(interpretab\w*|mechanistic|probe\w*|cka|geometry|geometric|representation\w*|steer\w*|activation\w*)\b/, 'mechanistic interpretability representation steering linear probes cka geometry'],
    [/\b(safety|safe|align\w*|refus\w*|guardrail\w*|robust\w*|adversarial|red.?team\w*|jailbreak\w*)\b/, 'ai safety alignment over-refusal steering guardrails robustness'],
    [/\b(maithili|nepali|nepal|low.?resource|multilingual|tirhuta|mithila|devanagari|indic)\b/, 'maithili maibert low-resource nlp tirhuta nepali devanagari'],
    [/\b(agent\w*|mcp|security|secure|zero.?trust|attack\w*|vulnerab\w*|threat\w*|injection|protocol)\b/, 'agentic systems mcp security zero-trust mitre attack vulnerability prompt injection'],
    [/\b(rag|retriev\w*|vector|embedding\w*|qdrant|search|chatbot|llm|vllm|fine.?tun\w*|lora|serving)\b/, 'rag retrieval qdrant vector embedding vllm llm fine-tuning peft lora serving'],
    [/\b(vision|image\w*|yolo|detect\w*|ocr|camera|robot\w*|banknote\w*|counterfeit)\b/, 'computer vision yolov8 detection ocr robotics banknote counterfeit'],
    [/\b(speech|speak\w*|voice|audio|asr|tts|whisper|transcri\w*|text.?to.?speech|speech.?to.?text|sound|pronounc\w*)\b/, 'maithili speech whisper asr tts text-to-speech speech-to-text voice syspin'],
    [/\b(who|about|yourself|himself|intro\w*|introduce|summary|summar\w*|bio|background|profile|overview|tldr)\b/, 'about me professional summary researcher background'],
    [/\b(hobby|hobbies|interest\w*|free.?time|personal|philosoph\w*|fun|outside|life|passion\w*|motto|value\w*)\b/, 'personal philosophy hobbies meditation walking photography interests motto'],
    [/\b(blog|blogs|write|writes|writing|essay\w*|post\w*|youtube|video\w*|talk\w*|channel|content)\b/, 'writing blog talks youtube audio obsession explainers essays'],
    [/\b(community|volunteer\w*|mentor\w*|teach\w*|organiz\w*|club|social|contribut\w*)\b/, 'community social contributions mentor npl coders robotics association'],
    [/\b(timeline|when|year\w*|history|journey|path|start\w*|began|progress\w*)\b/, 'career timeline year began'],
    [/\b(note\w*|document\w*|slide\w*|pdf|material\w*|resource\w*|lecture\w*|certificate\w*|course\w*)\b/, 'documents notes resources study materials certificates pdf'],
];

/**
 * Short elliptical replies ("and that?", "tell me more", "why?") carry no
 * retrievable content on their own.
 *
 * Bare `how`, `when`, `where` and `which` are deliberately NOT here: they open
 * perfectly self-contained questions. Including `how` made "How do I bake
 * sourdough bread?" inherit the previous turn's topic, which pushed an
 * off-topic question above the relevance floor and got it answered instead of
 * declined. Only `how about` — genuinely elliptical — is matched.
 */
export const ELLIPTICAL = /^(and|also|what about|how about|tell me more|more|more on|why|who else|any(thing)? else|else|other|ok|okay|go on|continue|expand|elaborate|really|that|those|it|he|his|him|them|there)\b/i;

/**
 * Build the string actually used for retrieval: the question, plus the previous
 * turn's topic when the question is elliptical, plus bridged vocabulary.
 * Only retrieval sees this — the generator gets the user's original wording.
 */
export function buildSearchQuery(query, history = []) {
    let q = query;
    const words = query.trim().split(/\s+/).length;
    const last = history[history.length - 1];

    // A follow-up needs an elliptical opener (and brevity — a long question that
    // merely starts with "why" or "and" stands on its own), or to consist of
    // nothing but stop words ("more?", "and?").
    //
    // Note there is deliberately no plain word-count rule. Treating any query of
    // three words or fewer as a follow-up broke "What is maiBERT?" — exactly
    // three words, entirely self-contained — which inherited the previous
    // question's topic and was answered with the awards section.
    const isFollowUp = last && (
        (ELLIPTICAL.test(query.trim()) && words <= 6) || tokenize(query).length === 0
    );
    if (isFollowUp) {
        q = `${last.query} ${last.topic || ''} ${query}`;
    }

    let expanded = q;
    for (const [pattern, extra] of EXPANSIONS) {
        if (pattern.test(q)) expanded += ' ' + extra;
    }
    return expanded;
}

/* ============================================================
   Fusion
   ============================================================ */

/** Dot product. Vectors from the embedder are L2-normalised, so this is cosine. */
export const cosine = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
};

/** 0-based rank per original index, highest score first. */
export function ranks(scores) {
    const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
    const out = new Array(scores.length);
    order.forEach(([, i], rank) => { out[i] = rank; });
    return out;
}

/**
 * Reciprocal rank fusion of the dense and lexical rankings. RRF is used rather
 * than a weighted score sum because the two scorers live on entirely different
 * scales (bounded cosine vs unbounded BM25), so their raw values are not
 * comparable — only their orderings are.
 */
export function fuse(chunks, dense, lexical, { rrfK = CONFIG.rrfK, topK = CONFIG.topK } = {}) {
    const dRank = ranks(dense), lRank = ranks(lexical);
    const scored = chunks.map((chunk, i) => ({
        chunk,
        dense: dense[i],
        lexical: lexical[i],
        score: 1 / (rrfK + dRank[i]) + (lexical[i] > 0 ? 1 / (rrfK + lRank[i]) : 0),
    }));
    scored.sort((a, b) => b.score - a.score);
    return { hits: scored.slice(0, topK), bestDense: Math.max(...dense) };
}

/**
 * Truncate to a word boundary so the model never sees a severed word. Cuts at
 * the last space rather than the last sentence end: backing up to a sentence
 * boundary can discard a whole trailing clause, and losing that information
 * measurably degraded answers more than a mid-sentence cut did.
 */
function clip(text, max) {
    if (text.length <= max) return text;
    const cut = text.slice(0, max);
    const space = cut.lastIndexOf(' ');
    return (space > max * 0.8 ? cut.slice(0, space) : cut).trim();
}

/**
 * Concatenate top hits into a context block that fits the encoder budget,
 * strictly in rank order — the best chunk gets everything it needs and later
 * chunks take what is left.
 *
 * Splitting the budget evenly across hits was tried and measurably hurt answer
 * quality: clipping the top chunk left the model with fragments and it filled
 * the gaps by confabulating (in one case inventing employers from a truncated
 * heading). At 77M parameters, one complete relevant passage beats three
 * partial ones, so rank priority wins over coverage here.
 */
export function buildContext(hits, limit = CONFIG.maxContextChars) {
    const parts = [];
    let budget = limit;
    hits.forEach(({ chunk }, i) => {
        // Below this a fragment adds noise rather than information.
        if (budget < 160) return;
        // Label/link dumps only earn a place when they are the primary hit.
        if (i > 0 && !chunk.prose) return;
        const text = clip(chunk.plain, budget);
        if (!text) return;
        parts.push(text);
        budget -= text.length + 1;
    });
    return parts.join('\n');
}

/**
 * Prompt wording matters a lot at 77M parameters, and four variants were A/B'd
 * in a real browser (not just Node) before settling here.
 *
 * The instruction is kept deliberately plain. Variants that asked for "the
 * specific names, numbers, dates and results mentioned in the context" produced
 * richer answers under native ONNX Runtime, but in-browser they frequently made
 * the model echo the instruction back as a complaint — "Please provide the
 * specific names, numbers, dates..." — or narrate the phrase "the third person"
 * as if it were content. Fewer instructions to latch onto means fewer of those
 * failures; `looksDegenerate` catches whatever still slips through.
 */
export function buildPrompt(query, hits) {
    return 'Answer the question using only the context below. '
        + 'Reply with 2 to 3 complete sentences about Sumit Yadav.\n\n'
        + `Context:\n${buildContext(hits)}\n\nQuestion: ${query}\n\nAnswer:`;
}

/* ============================================================
   Output cleanup
   ============================================================ */

/** Trim, de-loop and punctuate. The 77M model occasionally repeats a sentence
 *  verbatim, which reads as a glitch; exact repeats are dropped. */
export function tidy(text) {
    let t = (text || '').trim().replace(/\s+/g, ' ');
    if (!t) return '';
    const seen = new Set(), keep = [];
    for (const s of t.match(/[^.!?]+[.!?]*/g) || [t]) {
        const key = s.trim().toLowerCase();
        if (key.length > 12 && seen.has(key)) continue;
        seen.add(key);
        keep.push(s.trim());
    }
    t = keep.join(' ').trim();
    t = t.charAt(0).toUpperCase() + t.slice(1);
    if (!/[.!?]$/.test(t)) t += '.';
    return t;
}

/** An answer this short is a failed generation, not a terse one. */
export const isTooShort = (answer) => (answer || '').replace(/[^a-z]/gi, '').length < 20;

/**
 * Patterns a degenerate generation falls into. A 77M model executed with
 * WebAssembly int8 kernels is materially less reliable than the same weights
 * under native ONNX Runtime — measured, not assumed: identical prompts that
 * answer correctly in Node regularly come back in-browser as a refusal, an echo
 * of the instruction, or a repetition loop. Since retrieval has already proven
 * the context *does* contain the answer, any of these is a generator failure and
 * must never reach the user; the caller falls back to `extractiveAnswer`.
 */
const DEGENERATE = [
    // "the context does not provide / mention / contain ..."
    /\b(context|text|passage|information|document)\b[^.]{0,48}?\b(does not|doesn'?t|do not|don'?t|did not|didn'?t|is not|isn'?t|not)\b[^.]{0,20}?\b(provide|provided|contain|mention|include|specify|state|say)/i,
    /\b(not provided|no answer|cannot (be )?answer|can'?t answer|unable to answer|no specific answer|not mentioned|not specified)\b/i,
    // echoing the instruction back as a request
    /\bplease provide\b|\bprovide the specific\b/i,
    // echoing the "third person" instruction as content
    /\bthe (first|second|third) person\b/i,
    /\bin the third person\b/i,
    // answering with a bare section heading
    /^(the answer is:?\s*)?[A-Z][\w&' ]{0,40}(&|and) [A-Z][\w' ]{0,40}\.?$/,
];

/** Does any 5-word shingle repeat three or more times? */
function loops(text) {
    const words = text.toLowerCase().split(/\s+/);
    if (words.length < 15) return false;
    const seen = new Map();
    for (let i = 0; i + 5 <= words.length; i++) {
        const key = words.slice(i, i + 5).join(' ');
        const n = (seen.get(key) || 0) + 1;
        if (n >= 3) return true;
        seen.set(key, n);
    }
    return false;
}

export function looksDegenerate(answer) {
    const t = (answer || '').trim();
    if (!t) return true;
    if (DEGENERATE.some((re) => re.test(t))) return true;
    return loops(t);
}

/* ============================================================
   Extractive fallback
   ============================================================ */

const splitSentences = (text) =>
    text.split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 28
            // Section lead-ins ("Awards and recognition Sumit has received:")
            // introduce content rather than being content, and make a poor
            // opening line for an answer.
            && !s.endsWith(':'));

/**
 * Compose an answer by selecting the sentences from the retrieved context that
 * best match the question, in their original order.
 *
 * This is the safety net under the generator: it cannot hallucinate, because
 * every sentence is copied verbatim from bio.txt. It still reads as an answer
 * rather than a dumped section, which is the point — the visitor asked a
 * question and should get prose back.
 *
 * Returns null when the best chunk is a label/link dump (a contact block has no
 * sentences to select); the caller should render that chunk's markdown instead.
 */
export function extractiveAnswer(query, hits, maxSentences = 3) {
    if (!hits.length || !hits[0].chunk.prose) return null;

    const qTokens = new Set(tokenize(query));
    if (!qTokens.size) return null;

    const candidates = [];
    hits.forEach(({ chunk }, hitIdx) => {
        if (hitIdx > 0 && !chunk.prose) return;
        // Drop the synthetic heading sentence prepended in chunkBio.
        const body = chunk.plain.replace(/^[^.]{0,120}\.\s*/, '');
        splitSentences(body).forEach((text, i) => {
            const tokens = tokenize(text);
            if (!tokens.length) return;
            const overlap = tokens.reduce((n, t) => n + (qTokens.has(t) ? 1 : 0), 0);
            candidates.push({
                text, hitIdx, i,
                // Normalise by length so a long sentence doesn't win on volume
                // alone, and favour earlier sentences of higher-ranked chunks,
                // which is where sections state their main point.
                score: (overlap / Math.sqrt(tokens.length)) - hitIdx * 0.12 - i * 0.02,
            });
        });
    });

    if (!candidates.length) return null;

    const picked = candidates
        .sort((a, b) => b.score - a.score)
        .slice(0, maxSentences)
        .sort((a, b) => (a.hitIdx - b.hitIdx) || (a.i - b.i));

    const text = picked.map((p) => p.text).join(' ').trim();
    return text.length < 40 ? null : text;
}

/**
 * Decide what the visitor actually sees, given the generated answer and the
 * retrieved hits. Kept here (rather than in the page) so the policy is testable.
 *
 * Three sources of truth, in order of preference:
 *   - `markdown`  the retrieved section rendered as-is. Correct when the best
 *                 section is a structured list — a contact block or a skills
 *                 table *is* the answer, and prose paraphrase only loses the
 *                 links and grouping. Observed: the generator answered "how can
 *                 I contact him?" with "you can visit the provided information
 *                 on their website", which is strictly worse than the list.
 *   - `text`      the generated answer, when it is usable and substantial.
 *   - `text`      the extractive answer otherwise — including when generation is
 *                 merely thin. A stub like "Maithili is a type of language
 *                 technology" passes the degeneracy filter but says less than
 *                 the sentences it was built from, so prefer those.
 */
export function chooseAnswer(query, hits, generated) {
    if (!hits.length) return null;

    // A structured section answers better as itself than as paraphrase.
    if (!hits[0].chunk.prose) return { kind: 'markdown', chunk: hits[0].chunk };

    const extracted = extractiveAnswer(query, hits);
    const usable = generated && !isTooShort(generated) && !looksDegenerate(generated);
    const thin = extracted && generated
        && generated.length < 120
        && extracted.length > generated.length * 1.3;

    if (usable && !thin) return { kind: 'text', text: generated, source: 'model' };
    if (extracted) return { kind: 'text', text: extracted, source: 'extractive' };
    return { kind: 'markdown', chunk: hits[0].chunk };
}

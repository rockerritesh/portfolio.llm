/**
 * End-to-end check of the retrieval core against the real bio.txt.
 *
 *   npm install && npm test
 *
 * Loads the same models and the same rag.js the browser uses, so a regression
 * here is a regression in production. Run this after editing bio.txt — it will
 * tell you if a question stopped resolving to the section you expect.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { pipeline } from '@huggingface/transformers';
import {
    CONFIG, chunkBio, buildLexicon, bm25, tokenize,
    buildSearchQuery, cosine, fuse, buildPrompt, tidy, isTooShort, chooseAnswer,
} from '../rag.js';

const here = dirname(fileURLToPath(import.meta.url));
const bio = readFileSync(join(here, '..', 'bio.txt'), 'utf8');

/** Question → substring expected in the winning section's title. */
const EXPECTED = [
    ['Who is Sumit, in short?',                     'About'],
    ['What does he do at Astha.ai?',                'Astha'],
    ['Where does he work right now?',               'Astha'],
    ['Tell me about his RAG experience',            'Previous Roles'],
    ['What is SafeConstellations about?',           'SafeConstellations'],
    ['What is maiBERT?',                            'maiBERT'],
    ['Does he work on Maithili speech?',            'Maithili'],
    ['What awards has he won?',                     'Honors'],
    ['Where did he go to university?',              'Education'],
    ['What programming languages does he know?',    'Technical Skills'],
    ['How can I contact him?',                      'Identity'],
    ['What are his hobbies?',                        'Personal'],
    ['Does he have a YouTube channel?',             'Writing'],
    ['What did he do in 2021?',                     'Timeline'],
    ['Tell me about MCP security work',             'Security'],
    ['Any computer vision projects?',               'Vision'],
    ['Is he involved in the community?',            'Community'],
    ['Where can I find his CV?',                    'Identity'],
];

const OFF_TOPIC = [
    'What is the capital of France?',
    'How do I bake sourdough bread?',
    "What's his favourite football team?",
    'Give me a recipe for chicken curry',
];

const GENERATE = [
    'What does he do at Astha.ai?',
    'Tell me about his RAG experience',
    'What is SafeConstellations about?',
    'What awards has he won?',
];

/* ---------- index ---------- */

const chunks = chunkBio(bio);
const lexicon = buildLexicon(chunks.map((c) => c.plain));
console.log(`Indexed ${chunks.length} sections from bio.txt`);

const oversized = chunks.filter((c) => c.plain.length > CONFIG.chunkCharLimit + 400);
if (oversized.length) {
    console.log(`  note: ${oversized.length} chunk(s) exceed the soft size limit:`);
    oversized.forEach((c) => console.log(`    ${c.plain.length} chars — ${c.title}`));
}

const embedder = await pipeline('feature-extraction', CONFIG.embedModel, { dtype: 'q8' });
const embed = async (texts) =>
    (await embedder(texts, { pooling: 'mean', normalize: true })).tolist();

const vectors = [];
for (let i = 0; i < chunks.length; i += 8) {
    vectors.push(...await embed(chunks.slice(i, i + 8).map((c) => c.plain)));
}

async function search(query, history = []) {
    const sq = buildSearchQuery(query, history);
    const [qv] = await embed([sq]);
    const dense = vectors.map((v) => cosine(qv, v));
    return fuse(chunks, dense, bm25(lexicon, tokenize(sq)));
}

/* ---------- routing ---------- */

let pass = 0, fail = 0;
console.log('\n── Routing ───────────────────────────────────────────');
for (const [query, expected] of EXPECTED) {
    const { hits, bestDense } = await search(query);
    const top = hits[0].chunk.title;
    // Credit any of the top-K, since all of them reach the generator as context.
    const rank = hits.findIndex((h) => h.chunk.title.toLowerCase().includes(expected.toLowerCase()));
    const ok = rank === 0;
    if (ok) pass++; else if (rank > 0) pass++; else fail++;
    const mark = rank === 0 ? 'PASS' : rank > 0 ? `PASS@${rank + 1}` : 'FAIL';
    console.log(`${mark.padEnd(7)} ${bestDense.toFixed(3)}  "${query}"`);
    if (rank !== 0) console.log(`${' '.repeat(8)}→ got "${top}", wanted ~"${expected}"`);
}

/* ---------- out-of-scope ---------- */

console.log('\n── Out-of-scope (want dense < ' + CONFIG.weakAt + ') ──────────');
let declined = 0;
for (const query of OFF_TOPIC) {
    const { bestDense } = await search(query);
    const ok = bestDense < CONFIG.weakAt;
    if (ok) declined++;
    console.log(`${(ok ? 'DECLINE' : 'ANSWER ').padEnd(7)} ${bestDense.toFixed(3)}  "${query}"`);
}

/* ---------- follow-ups ---------- */

console.log('\n── Follow-up rewriting ───────────────────────────────');
const first = await search('What is maiBERT?');
const history = [{ query: 'What is maiBERT?', topic: first.hits[0].chunk.title }];
for (const q of ['tell me more', 'why does it matter?']) {
    const { hits } = await search(q, history);
    const stayed = /maibert|maithili/i.test(hits[0].chunk.title + hits[0].chunk.plain);
    console.log(`${(stayed ? 'PASS' : 'FAIL').padEnd(7)} "${q}" → ${hits[0].chunk.title}`);
    if (!stayed) fail++;
}

// Regression: a short but self-contained question must NOT inherit the previous
// topic. "What is maiBERT?" is three words and once got answered with the awards
// section because a word-count rule classified it as a follow-up.
console.log('\n── Self-contained questions ignore history ───────────');
const priorAwards = [{ query: 'What awards has he won?', topic: 'Honors & Awards' }];
for (const [q, want] of [['What is maiBERT?', /maibert/i], ['Where did he study?', /education/i]]) {
    const { hits } = await search(q, priorAwards);
    const ok = want.test(hits[0].chunk.title);
    console.log(`${(ok ? 'PASS' : 'FAIL').padEnd(7)} "${q}" → ${hits[0].chunk.title}`);
    if (!ok) fail++;
}

// The answer-selection policy: structured sections render as themselves,
// degenerate generations never survive.
console.log('\n── Answer selection policy ───────────────────────────');
const contact = await search('How can I contact him?');
const contactPick = chooseAnswer('How can I contact him?', contact.hits, 'To contact Sumit Yadav, you can visit the provided information on their website.');
console.log(`${(contactPick.kind === 'markdown' ? 'PASS' : 'FAIL').padEnd(7)} contact question → ${contactPick.kind}`);
if (contactPick.kind !== 'markdown') fail++;

const mai = await search('What is maiBERT?');
for (const [label, generated] of [
    ['refusal',      'The provided context does not provide information about maiBERT.'],
    ['instruction',  'Please provide the specific names, numbers, dates and results mentioned in the context.'],
    ['third person', 'The third person is Sumit Yadav, an AI researcher.'],
    ['loop',         'The first BERT for Maithili is the first BERT for Maithili is the first BERT for Maithili is the first BERT for Maithili.'],
]) {
    const pick = chooseAnswer('What is maiBERT?', mai.hits, generated);
    const rejected = pick.source !== 'model';
    console.log(`${(rejected ? 'PASS' : 'FAIL').padEnd(7)} rejects ${label} → ${pick.source || pick.kind}`);
    if (!rejected) fail++;
}

/* ---------- generation ---------- */

console.log('\n── Generated answers ─────────────────────────────────');
const gen = await pipeline('text2text-generation', CONFIG.genModel, { dtype: 'q8' });
for (const query of GENERATE) {
    const { hits } = await search(query);
    const prompt = buildPrompt(query, hits);
    const t0 = Date.now();
    const out = await gen(prompt, { max_new_tokens: CONFIG.maxNewTokens, do_sample: false, repetition_penalty: 1.15 });
    const answer = tidy(out[0].generated_text);
    const ms = Date.now() - t0;
    if (isTooShort(answer)) fail++;
    console.log(`\nQ: ${query}`);
    console.log(`A: ${answer}`);
    console.log(`   [${ms} ms · prompt ${prompt.length} chars · ${hits.map((h) => h.chunk.title).join(' | ')}]`);
}

console.log(`\n${'─'.repeat(54)}`);
console.log(`Routing: ${pass}/${EXPECTED.length} · Declined off-topic: ${declined}/${OFF_TOPIC.length}`);
if (fail) { console.log(`FAILURES: ${fail}`); process.exit(1); }
console.log('All checks passed.');

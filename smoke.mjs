// Temporary browser smoke test: serves the site, drives it in headless Chromium,
// asks a spread of questions and asserts that no degenerate answer is ever shown.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.txt': 'text/plain', '.png': 'image/png' };
const server = createServer(async (req, res) => {
    try {
        const u = req.url.split('?')[0];
        const p = join(process.cwd(), u === '/' ? 'index.html' : u);
        const b = await readFile(p);
        res.writeHead(200, { 'Content-Type': TYPES[extname(p)] || 'application/octet-stream' });
        res.end(b);
    } catch { res.writeHead(404).end('nope'); }
});
await new Promise((r) => server.listen(4173, r));

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });

await page.goto('http://localhost:4173/', { waitUntil: 'load' });
console.log('title:', await page.title());
console.log('favicon link:', await page.$eval('link[rel="icon"]', (n) => n.getAttribute('href')));
console.log('avatar loaded:', await page.$eval('.avatar', (n) => n.complete && n.naturalWidth > 0));

// Wait for boot: both models loaded (status returns to Ready with input enabled).
const deadline = Date.now() + 600_000;
let seen = '';
while (Date.now() < deadline) {
    const s = await page.$eval('#status-text', (n) => n.textContent);
    const state = await page.$eval('#status', (n) => n.dataset.state);
    if (s !== seen) { console.log('  status:', s); seen = s; }
    if (state === 'error') break;
    if (s === 'Ready' && !(await page.$eval('#input', (n) => n.disabled))) {
        // generator warm-up also finishes by setting Ready a second time
        await new Promise((r) => setTimeout(r, 2500));
        if (await page.$eval('#status-text', (n) => n.textContent) === 'Ready') break;
    }
    await new Promise((r) => setTimeout(r, 1500));
}
console.log('state:', await page.$eval('#status', (n) => n.dataset.state));
console.log('chips:', await page.$$eval('.chip', (n) => n.length));

const DEGENERATE = /does not (provide|contain|mention)|not provided|cannot answer|please provide|the (first|second|third) person|in the third person/i;

async function ask(q) {
    const before = await page.$$eval('.msg.bot', (n) => n.length);
    await page.fill('#input', q);
    await page.click('#send');
    await page.waitForFunction(
        (n) => document.querySelectorAll('.msg.bot').length > n
            && !document.querySelector('#input').disabled,
        before, { timeout: 180_000 },
    );
    const text = await page.$$eval('.msg.bot .bubble', (n) => n[n.length - 1].innerText.trim());
    const sources = await page.$$eval('.msg.bot:last-of-type .cited-head', (n) => n.map((x) => x.textContent));
    return { text, sources };
}

const QUESTIONS = [
    'What does he do at Astha.ai?',
    'What awards has he won?',
    'What is maiBERT?',
    'Tell me about his RAG experience',
    'Where did he study?',
    'What are his hobbies?',
    'How can I contact him?',
];

let bad = 0;
for (const q of QUESTIONS) {
    const { text, sources } = await ask(q);
    const degenerate = DEGENERATE.test(text);
    if (degenerate) bad++;
    console.log(`\n${degenerate ? 'BAD ' : 'OK  '} Q: ${q}`);
    console.log(`     A: ${text.split('\n')[0].slice(0, 260)}`);
    console.log(`     sources: ${sources.length}`);
}

// Out-of-scope must be declined, not answered.
const off = await ask('How do I bake sourdough bread?');
const declined = /bio covers|rather not guess/i.test(off.text);
console.log(`\n${declined ? 'OK  ' : 'BAD '} off-topic declined: ${off.text.slice(0, 120)}`);

// Follow-up should stay on topic.
await ask('What is maiBERT?');
const follow = await ask('tell me more');
console.log(`follow-up sources: ${follow.sources[0] || '(none)'}`);

console.log('\npage errors:', errors.length ? errors.slice(0, 5) : 'none');
await browser.close();
server.close();

const ok = bad === 0 && declined && errors.length === 0;
console.log(ok ? '\nSMOKE TEST PASSED' : `\nSMOKE TEST FAILED (degenerate answers: ${bad})`);
process.exit(ok ? 0 : 1);

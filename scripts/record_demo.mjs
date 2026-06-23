// Render proof artifacts for the REGIME site: clean section stills + a scroll-through MP4.
// Usage: node scripts/record_demo.mjs   (outputs to ./media/)
import { spawn, execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync, rmSync, readdirSync } from 'node:fs';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const SITE = 'file://' + process.cwd() + '/design/index.html';
const MEDIA = process.cwd() + '/media';
const FRAMES = '/tmp/regime-frames';
const PORT = 9390;
const W = 1280, H = 800;

mkdirSync(MEDIA, { recursive: true });
rmSync(FRAMES, { recursive: true, force: true }); mkdirSync(FRAMES, { recursive: true });

const chrome = spawn(CHROME, [
  '--headless=new', '--hide-scrollbars', '--mute-audio', '--enable-webgl',
  '--remote-debugging-port=' + PORT, `--window-size=${W},${H}`,
  '--force-device-scale-factor=2', '--user-data-dir=/tmp/regime-rec-' + Date.now(), 'about:blank',
], { stdio: 'ignore' });

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function wsUrl() {
  for (let i = 0; i < 60; i++) {
    try { const r = await fetch(`http://127.0.0.1:${PORT}/json`); const l = await r.json();
      const p = l.find(t => t.type === 'page'); if (p?.webSocketDebuggerUrl) return p.webSocketDebuggerUrl; } catch {}
    await sleep(120);
  }
  throw new Error('no devtools');
}
let _id = 0;
function rpc(ws, m, p = {}) {
  const id = ++_id;
  return new Promise((res, rej) => {
    const f = ev => { const o = JSON.parse(ev.data); if (o.id === id) { ws.removeEventListener('message', f); o.error ? rej(new Error(o.error.message)) : res(o.result); } };
    ws.addEventListener('message', f); ws.send(JSON.stringify({ id, method: m, params: p }));
  });
}

const ws = new WebSocket(await wsUrl());
await new Promise(r => ws.addEventListener('open', r, { once: true }));
await rpc(ws, 'Page.enable'); await rpc(ws, 'Runtime.enable');
await rpc(ws, 'Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 2, mobile: false });
await rpc(ws, 'Page.navigate', { url: SITE });
await sleep(2600); // load + fonts + hero intro

const scrollTo = y => rpc(ws, 'Runtime.evaluate', { expression: `window.lenis?lenis.scrollTo(${y},{immediate:true}):window.scrollTo(0,${y});` });
const Y = (await rpc(ws, 'Runtime.evaluate', { returnByValue: true,
  expression: `(()=>{const q=s=>document.querySelector(s);return{verdict:q('#verdict').offsetTop,reason:q('#method').offsetTop+innerHeight*1.4,results:q('#results').offsetTop};})()`
})).result.value;

// ---- clean section stills ----
async function still(name, y, wait = 1700) {
  await scrollTo(y); await sleep(wait);
  const { data } = await rpc(ws, 'Page.captureScreenshot', { format: 'png' });
  writeFileSync(`${MEDIA}/${name}.png`, Buffer.from(data, 'base64'));
  console.log('still:', name);
}
await still('01-hero', 0);
await still('02-verdict', Y.verdict - 30, 3600); // race -> tie -> stamp payoff completes ~3s in
await still('03-markov-3d', Y.reason);
await still('04-results', Y.results - 30);

// ---- scroll-through video via screencast ----
await scrollTo(0); await sleep(600);
const frames = [];
const onFrame = ev => {
  const o = JSON.parse(ev.data);
  if (o.method === 'Page.screencastFrame') {
    frames.push(o.params.data);
    rpc(ws, 'Page.screencastFrameAck', { sessionId: o.params.sessionId }).catch(() => {});
  }
};
ws.addEventListener('message', onFrame);
await rpc(ws, 'Page.startScreencast', { format: 'jpeg', quality: 85, maxWidth: 1600, maxHeight: 1000, everyNthFrame: 1 });

const t0 = Date.now();
// scripted cinematic scroll with dwell points so section animations play
await rpc(ws, 'Runtime.evaluate', { awaitPromise: true, expression: `(async()=>{
  const L=window.lenis; const q=s=>document.querySelector(s);
  const go=(y,d)=>new Promise(r=>{ if(L){L.scrollTo(y,{duration:d}); setTimeout(r,d*1000+150);} else {window.scrollTo(0,y); setTimeout(r,d*1000);} });
  await new Promise(r=>setTimeout(r,1600));                                   // hero
  await go(q('#verdict').offsetTop-30, 2.2); await new Promise(r=>setTimeout(r,3200)); // race -> tie -> stamp
  await go(q('#method').offsetTop+innerHeight*1.4, 2.6); await new Promise(r=>setTimeout(r,3200)); // 3D markov
  await go(q('#results').offsetTop-30, 2.2); await new Promise(r=>setTimeout(r,2600)); // chart draws
  await go(document.body.scrollHeight, 2.4); await new Promise(r=>setTimeout(r,900));  // footer
})()` });
const durSec = (Date.now() - t0) / 1000;
await rpc(ws, 'Page.stopScreencast');
ws.removeEventListener('message', onFrame);
await sleep(200);

console.log(`captured ${frames.length} frames over ${durSec.toFixed(1)}s`);
frames.forEach((d, i) => writeFileSync(`${FRAMES}/f-${String(i).padStart(4, '0')}.jpg`, Buffer.from(d, 'base64')));
// encode INPUT at the true capture rate, OUTPUT at 30fps -> real-time playback
const capFps = Math.max(1, frames.length / durSec).toFixed(2);

ws.close(); chrome.kill(); await sleep(200);

// ---- encode ----
const n = readdirSync(FRAMES).length;
if (n > 5) {
  execFileSync('ffmpeg', ['-y', '-framerate', String(capFps), '-i', `${FRAMES}/f-%04d.jpg`,
    '-r', '30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=1280:-2', '-movflags', '+faststart',
    `${MEDIA}/regime-demo.mp4`], { stdio: 'ignore' });
  console.log(`video: media/regime-demo.mp4  (capture ${capFps}fps -> 30fps real-time, ${n} frames)`);
} else {
  console.log('not enough frames to encode video');
}
console.log('done -> media/');
process.exit(0);

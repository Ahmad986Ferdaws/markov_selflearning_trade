# design/ — the static site

One page, served as-is. Nothing in this folder is built; what is here is what is deployed.

| File | What it is |
|---|---|
| [index.html](index.html) | The whole site: markup, CSS and JS in one file. The equity chart is drawn from real held-out engine output, not a generated series. |
| [favicon.svg](favicon.svg) | The tab icon. |
| [robots.txt](robots.txt) | Allows everything; it is one public page. |
| [vercel.json](vercel.json) | Response headers: the Content-Security-Policy, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and a one-week cache for the favicon. |
| [philosophy.md](philosophy.md) | The original design notes and the record of the refinement pass. |

`.vercel/`, the project link Vercel writes locally, is git-ignored.

## Serve it locally

```bash
make site            # python -m http.server on design/, http://localhost:8000
PORT=9000 make site  # another port
```

Opening `index.html` straight from the file system also works. Either way the four CDN
scripts need network; without them the page adds the `nogsap` class and renders static.

## How it is deployed

A Vercel project named `regime`: static, no framework, no build step, `design/` as the root.
The response headers come from `vercel.json`; nothing else is configured. The social-card
image has to be an absolute URL into this repository's `media/` folder, because `media/`
lies outside the deployed root.

## Design language

[philosophy.md](philosophy.md) holds the notes the site started from and what the
refinement pass changed. The current build follows the Robinhood-2024 language recorded in
[docs/STATUS.md](../docs/STATUS.md): black, white and neutrals with one neon accent;
Newsreader, Schibsted Grotesk and IBM Plex Mono. Three rules are fixed, and they win where
the older notes disagree: no glow, no scroll cues, and the chart shows engine data, never an
invented series.

Any visual change is checked at desktop width and at 375 px before it is committed.

## The four CDN scripts and their hashes

`index.html` loads GSAP 3.12.5, ScrollTrigger 3.12.5, Lenis 1.1.14 and Three.js r128 from
cdnjs and jsDelivr. Each `<script>` tag carries `integrity="sha384-..."` and
`crossorigin="anonymous"`, and each host is listed in the CSP `script-src` in `vercel.json`.
To bump a version, recompute the hash from the exact URL and change the URL and the hash in
the same commit:

```bash
curl -s URL | openssl dgst -sha384 -binary | openssl base64 -A
```

A hash that does not match makes the browser refuse the script and the page falls back to
its static layout. That is the intended failure; do not fix it by removing the attribute.
`tests/test_site_assets.py` pins the attributes and the CSP hosts.

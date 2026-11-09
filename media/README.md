# media/ — the proof media

Four screenshots and one video of the site, rendered from `design/index.html` by the demo
recorder. They are evidence of what the page looked like when the numbers were published,
not design sources.

| File | What it shows | Where it is used |
|---|---|---|
| `01-hero.png` | The top of the page. | not embedded |
| `02-verdict.png` | The `#verdict` band after its payoff animation has finished: 90.9% against 90.9%. | embedded in [README](../README.md); the site's social card (`og:image`, `twitter:image`) |
| `03-markov-3d.png` | The three-dimensional transition-matrix scene in the method section. | not embedded |
| `04-results.png` | The results band: held-out equity net of costs. | embedded in [README](../README.md) |
| `regime-demo.mp4` | A scripted scroll through the whole page, about 21 seconds in real time. | linked from [README](../README.md) |

## Regenerating

```bash
make demo          # node scripts/record_demo.mjs, writes into media/
make clean-media   # deletes the rendered png/mp4/gif files; this README stays
```

[`scripts/record_demo.mjs`](../scripts/record_demo.mjs) needs Node (it uses the built-in
`fetch` and `WebSocket`), Google Chrome at the path written at the top of the file (started
headless and driven over the DevTools protocol), and `ffmpeg` on the `PATH` for the video.
The page loads its four CDN scripts, so the recorder needs network. Stills are taken at
1280 × 800 with a device scale of 2; the video is captured by screencast and encoded at
30 frames per second so it plays in real time.

## Binary evidence

`.gitattributes` marks `*.png` and `*.mp4` as binary: never diffed, never line-ending
converted. A change to one of these files shows in a pull request only as a binary change.

# Design Philosophy — "Quiet Instrument"

A visual movement for instruments that measure markets without lying about them.

## The philosophy

**Quiet Instrument** is the aesthetic of a precision tool seen in a darkened room. It rejects the casino — no confetti greens, no urgent reds screaming for a trade. Instead it treats financial truth the way an observatory treats starlight: as a faint, valuable signal that deserves a black, calm field to be seen against. The interface is mostly darkness, and the data is the only light. This is a screen that respects the user's intelligence and refuses to manufacture excitement the numbers don't earn.

**Space and form.** The composition breathes in deep ink — near-black navy that recedes, so luminous data floats forward. Surfaces are barely-there cards, lifted from the void by a single hairline and the faintest inner glow, never by heavy borders or drop shadows. Generous negative space is not emptiness; it is the silence that makes a single number audible. Every panel is a quiet rectangle in a strict grid, the way instrument gauges sit on a control surface — aligned to the millimetre, the product of painstaking attention.

**Color and material.** A disciplined, limited palette: a void of deep navy, a single cool accent of electric teal-cyan that reads as "live signal," and three restrained semantics — a muted emerald for gain, a desaturated rose for loss, an amber that glows only when the instrument must warn. Color is never decoration; it is information. The teal is used sparingly enough that when it appears, it means something. Material is glass and light — soft radial glows, subtle gradients that suggest depth without skeuomorphic gimmickry, master-level restraint over spectacle.

**Scale and rhythm.** Numbers are the heroes, set in a monospaced face with tabular figures so digits align like a ledger and shift without jitter — the typographic honesty of a real instrument. The largest type on the screen is reserved for the one number that matters; everything else recedes into small, clinical labels in a clean grotesk, tracked wide and dimmed to a whisper. The rhythm is hero-number, then quiet-label, then chart, repeated with the patient consistency of someone who has refined this layout a hundred times.

**Composition and motion.** Charts draw themselves like a plotter pen laying down ink — lines that animate into being rather than appearing all at once, numbers that count up to their truth, a glow that pulses once and settles. Motion is slow, eased, and purposeful: nothing bounces, nothing flashes. The eye is led from the headline finding, down through the equity curves, across the regime timeline, to the honest warning panel — a deliberate path composed with the care of a master craftsman.

**The discipline of honesty.** Above all, this instrument is built to tell the truth even when the truth is unimpressive. When the model has no edge, the design says so plainly, in calm amber, without shame. That intellectual honesty is itself the aesthetic — a refusal to dress a null result as a triumph. The result must look meticulously crafted, labored over for countless hours by someone at the absolute top of their field: every alignment intentional, every glow calibrated, every digit set with reverence for what it measures.

## Refinement pass — July 2026 (component best-practices audit)

A design-refinement pass audited `index.html` against a component best-practices reference (ui-design-brain) without changing the brand language, fonts, palette, motion stack, or the real held-out data. What changed:

- **Semantics & structure.** Added a `<main>` landmark; the skip link now targets it ("Skip to main content"). Section labels became real `h2` headings so the outline runs strictly h1 → h2 → h3 (the big verdict statement is now a styled paragraph, keeping one heading per level). The pull-quote is a proper `<blockquote>` with `<cite>`. The nav wordmark is a link back to top.
- **Navigation.** A scroll-spy sets `aria-current` on the section link in view, rendered as an underline + weight shift (clear active state). Footer links moved into a labeled `<nav>`; the fixed-position rule was scoped to `#nav` so it can't leak onto other nav elements. The dead `#` "Source" link now points at the real results JSON.
- **Table & chart.** The results table gained a visually-hidden caption, `scope="col"` headers, and a keyboard-focusable scroll region; the pending LLM-agent row announces "not yet run" to screen readers. The equity chart is now keyboard-operable (arrow keys scrub, Shift for 10-step, Home/End, Escape clears) with a visible focus ring; the hint text says so.
- **Interaction & touch.** Buttons meet 44 px height; nav and footer links got padding to reach comfortable targets. Verb-first CTA labels throughout ("Read the method").
- **Type & contrast.** Muted ink darkened one step for AA at small sizes; sub-12 px micro-labels (axis, scrub, tags, chips, badge) raised to 12 px; the disclosure paragraph raised to 13 px.
- **Rhythm & responsive.** Odd spacing values snapped to the 4/8 px grid; the 90.9 ≡ 90.9 tie-band re-clamped so it no longer overflows at 375 px; a dynamic copyright line completes the footer. Reduced-motion now also halts the ticker marquee.

Deliberately untouched: the fonts, neon `#CCFF00` accent, GSAP/Lenis/Three.js choreography, `.nogsap` fallback, SRI-pinned CDN tags, and every real data point.

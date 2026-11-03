# Copyright (c) 2026 AMLT course project (FlowDec review).
# Renders report/flowdec_slides.html: one self-contained presentation page.
from __future__ import annotations

import re
from pathlib import Path

from beartype import beartype

import slides_content as sc

ROOT = Path(__file__).resolve().parents[1]
OUT = sc.DECK_HTML
DOCS_DIR = OUT.parent

REQUIRED_LABEL = "Required criterion"


# --------------------------------------------------------------------------- #
# Paths: content stores project-root-relative paths; the HTML lives in report/.
# --------------------------------------------------------------------------- #


@beartype
def rel_url(root_relative: str) -> str:
    """Turn a project-root-relative path into a URL relative to report/."""
    return pure_relpath(root_relative)


@beartype
def pure_relpath(root_relative: str) -> str:
    """A POSIX path from report/ to the given project-root-relative path."""
    import os

    return os.path.relpath(ROOT / root_relative, DOCS_DIR).replace("\\", "/")


# --------------------------------------------------------------------------- #
# Block rendering.
# --------------------------------------------------------------------------- #


@beartype
def render_bullets(block: sc.Bullets) -> str:
    tag = "ol" if block.ordered else "ul"
    items = "\n".join(f"    <li>{sc.md_to_html(item)}</li>" for item in block.items)
    return f'<{tag} class="bullets">\n{items}\n</{tag}>'


@beartype
def render_para(block: sc.Para) -> str:
    return f'<p class="para">{sc.md_to_html(block.text)}</p>'


@beartype
def render_callout(block: sc.Callout) -> str:
    return f'<div class="callout">{sc.md_to_html(block.text)}</div>'


@beartype
def render_table(block: sc.Table) -> str:
    head = "".join(f"<th>{sc.md_to_html(h)}</th>" for h in block.headers)
    body_rows = []
    for row in block.rows:
        cells = "".join(f"<td>{sc.md_to_html(c)}</td>" for c in row)
        body_rows.append(f"      <tr>{cells}</tr>")
    body = "\n".join(body_rows)
    caption = f'<figcaption>{sc.md_to_html(block.caption)}</figcaption>' if block.caption else ""
    return (
        '<figure class="table-figure">\n'
        '  <div class="table-wrap"><table>\n'
        f"    <thead><tr>{head}</tr></thead>\n"
        f"    <tbody>\n{body}\n    </tbody>\n"
        "  </table></div>\n"
        f"  {caption}\n"
        "</figure>"
    )


@beartype
def render_picture(block: sc.Picture) -> str:
    caption = f"<figcaption>{sc.md_to_html(block.caption)}</figcaption>" if block.caption else ""
    return (
        '<figure class="picture-figure"'
        f' style="--pic-width:{round(block.width * 100)}%">\n'
        f'  <img src="{rel_url(block.path)}" alt="{sc.strip_md(block.caption) or Path(block.path).stem}">\n'
        f"  {caption}\n"
        "</figure>"
    )


@beartype
def render_audio(block: sc.Audio) -> str:
    players: list[str] = []
    for slug in block.clips:
        clip = sc.AUDIO_BY_SLUG[slug]
        players.append(
            '    <figure class="player">\n'
            f'      <img src="{rel_url(clip.wave)}" alt="waveform of {clip.label}">\n'
            '      <figcaption class="player-head">\n'
            f'        <span class="player-title">{clip.label}</span>\n'
            f'        <span class="player-note">{sc.md_to_html(clip.caption)}</span>\n'
            "      </figcaption>\n"
            f'      <audio controls preload="none" src="{rel_url(clip.mp3)}"></audio>\n'
            "    </figure>"
        )
    joined = "\n".join(players)
    caption = f'<p class="audio-caption">{sc.md_to_html(block.caption)}</p>' if block.caption else ""
    return f'<div class="audio-group">\n{caption}\n  <div class="players">\n{joined}\n  </div>\n</div>'


@beartype
def dispatch(block: sc.Block) -> str:
    """Render one content block to HTML."""
    if isinstance(block, sc.Bullets):
        return render_bullets(block)
    if isinstance(block, sc.Para):
        return render_para(block)
    if isinstance(block, sc.Callout):
        return render_callout(block)
    if isinstance(block, sc.Table):
        return render_table(block)
    if isinstance(block, sc.Picture):
        return render_picture(block)
    # `sc.Block` has no members left, so `block` is an `Audio` here.
    return render_audio(block)


# --------------------------------------------------------------------------- #
# The inline SVG diagram (read from the asset built by build_slide_assets.py).
# --------------------------------------------------------------------------- #


@beartype
def inline_diagram(svg_rel: str) -> str:
    svg_path = ROOT / svg_rel
    if not svg_path.is_file():
        return f'<img src="{rel_url(sc.DIAGRAM_PNG)}" alt="FlowDec pipeline">'
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r"<\?xml.*?\?>", "", text, flags=re.DOTALL)
    text = text.replace("<svg ", '<svg class="diagram-inline" aria-label="FlowDec pipeline diagram" ', 1)
    return f'<figure class="picture-figure" style="--pic-width:100%">\n  {text}\n</figure>'


# --------------------------------------------------------------------------- #
# Sorting the blocks so an image never crowds out the text (PPTX uses the same
# order, so the two decks stay parallel).
# --------------------------------------------------------------------------- #


@beartype
def sorted_blocks(slide: sc.Slide) -> list[sc.Block]:
    """Text first, then visuals, preserving the authored order inside a group."""
    text_kinds = (sc.Bullets, sc.Para, sc.Callout, sc.Table)

    def rank(block: sc.Block) -> int:
        return 0 if isinstance(block, text_kinds) else 1

    return sorted(slide.blocks, key=rank)


# --------------------------------------------------------------------------- #
# Page template.
# --------------------------------------------------------------------------- #

CSS = """
:root{
  --navy:#102345;--text:#243d61;--muted:#647992;--blue:#1676f3;--purple:#8954ef;
  --line:#dce7f5;--paper:#f7faff;--surface:#fff;--bg:#fbfdff;--tint:#eff6ff;
  --radius:20px;--font:Inter,"Segoe UI",Arial,sans-serif;--mono:"SFMono-Regular",Consolas,Menlo,monospace;
  --shadow:0 12px 45px #1f4d7c10;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:96px}
body{margin:0;background:var(--bg);color:var(--navy);font:16px/1.62 var(--font);-webkit-font-smoothing:antialiased}
h1,h2,h3,p,figure{margin:0}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}
button{font:inherit;cursor:pointer;color:inherit;border:1px solid var(--line);border-radius:10px;
  background:var(--surface);padding:9px 13px;font-size:12px;transition:background .2s,border-color .2s,transform .2s}
button:hover{border-color:#9fc7ff}
button[aria-pressed=true]{background:var(--tint);border-color:#9fc7ff;color:var(--blue)}
:focus-visible{outline:3px solid #5c9cff;outline-offset:3px}
.wrap{max-width:1320px;margin:auto;padding:0 32px}
.mono{font-family:var(--mono)}
.header{position:sticky;top:0;z-index:30;background:var(--surface);border-bottom:1px solid var(--line)}
.header .wrap{display:flex;align-items:center;gap:18px;min-height:68px;flex-wrap:wrap;padding-top:8px;padding-bottom:8px}
.brand{display:flex;align-items:center;gap:11px;font-size:19px;font-weight:750;letter-spacing:-.6px}
.brand-mark{width:34px;height:34px;display:grid;place-items:center;border-radius:10px;
  background:linear-gradient(140deg,#278aff,#7961e8);color:#fff;font:18px var(--mono)}
.brand small{color:var(--muted);font:10px var(--mono);letter-spacing:1px;margin-left:4px}
.nav{display:flex;gap:2px;margin:auto;flex-wrap:wrap}
.nav a{padding:7px 10px;border-radius:8px;color:var(--muted);font-size:12px}
.nav a.active,.nav a:hover{background:var(--tint);color:var(--blue);text-decoration:none}
.actions{display:flex;gap:7px;align-items:center}
.hero{background:linear-gradient(180deg,var(--surface),var(--paper));border-bottom:1px solid var(--line);
  padding:58px 0 44px}
.eyebrow{display:flex;gap:10px;align-items:center;font:10px var(--mono);letter-spacing:1.4px;
  text-transform:uppercase;color:#4c729d}
.eyebrow:before{content:'';height:2px;width:26px;background:var(--blue)}
.hero h1{font-size:clamp(34px,3.6vw,52px);line-height:1.08;letter-spacing:-1.8px;margin:20px 0 16px}
.hero p{font-size:17px;color:var(--muted);max-width:900px}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:26px}
.badge{font:11px var(--mono);padding:7px 11px;border-radius:8px;border:1px solid var(--line);background:var(--paper);color:var(--muted)}
section.content{padding:52px 0 8px}
.section-top{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:24px}
.section-top h2{font-size:clamp(26px,2.6vw,38px);line-height:1.14;letter-spacing:-1.1px;margin:12px 0 10px}
.section-top .sub{color:var(--muted);font-size:15px;max-width:940px}
.section-index{font:52px/1 var(--mono);color:var(--line);letter-spacing:-3px;flex:none}
.required{display:inline-flex;align-items:center;gap:7px;font:11px var(--mono);
  background:#eefaf3;border:1px solid #bde1ce;color:#127a4d;border-radius:999px;padding:5px 11px;margin-bottom:6px}
.required:before{content:'✓'}
.bullets{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:13px}
.bullets li{max-width:1120px}
.para{margin:0 0 14px;max-width:1120px;font-size:16px}
code{font-family:var(--mono);font-size:.88em;background:var(--tint);border:1px solid var(--line);
  border-radius:6px;padding:1px 6px;color:#124b8f;white-space:nowrap}
.callout{margin:18px 0;padding:18px 22px;border:1px solid #cce0fc;border-radius:12px;background:var(--tint);
  font-size:15px;color:var(--text)}
.bullets+.callout{margin-top:22px}
.picture-figure{margin:20px 0;max-width:var(--pic-width,90%)}
.picture-figure img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:14px;background:#fff}
.picture-figure figcaption,.table-figure figcaption{margin-top:10px;font-size:12.5px;color:var(--muted)}
.diagram-inline{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:14px;background:#fff}
.table-figure{margin:20px 0}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:14px;background:var(--surface)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:11px 16px;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--paper);font:11px var(--mono);letter-spacing:.6px;text-transform:uppercase;color:var(--muted)}
td{color:var(--text);font-variant-numeric:tabular-nums}
td:first-child{color:var(--navy);font-weight:650}
tbody tr:last-child td{border-bottom:0}
.audio-group{margin:18px 0}
.audio-caption{font-size:14px;font-weight:600;color:var(--navy);margin-bottom:12px}
.players{display:grid;gap:14px}
.player{margin:0;border:1px solid var(--line);border-radius:14px;background:var(--surface);overflow:hidden}
.player img{width:100%;height:96px;object-fit:cover;display:block;background:#fff}
.player-head{display:flex;gap:12px;align-items:baseline;padding:11px 16px 0;flex-wrap:wrap}
.player-title{font-size:14px;font-weight:700;color:var(--navy)}
.player-note{font-size:12px;color:var(--muted)}
.player audio{width:100%;display:block;padding:8px 14px 14px}
.notes{margin:22px 0 0;border:1px dashed var(--line);border-radius:12px;background:var(--surface);padding:14px 18px}
.notes summary{font-size:12px;color:var(--muted);cursor:pointer}
.notes p{margin-top:10px;font-size:13.5px;color:var(--text)}
.footer{margin-top:56px;padding:26px 0 40px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
.footer .wrap{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
.lecture-bar{display:none;position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--surface);
  border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 35px #17355526;padding:9px 13px;z-index:40;
  align-items:center;gap:12px;white-space:nowrap;font-size:12px}
.lecture .lecture-bar{display:flex}
.lecture .hero,.lecture .footer,.lecture .nav{display:none}
.lecture section.content{display:none}
.lecture section.content.lecture-current{display:block;padding-top:26px;padding-bottom:120px}
.lecture section.content.lecture-current h2{font-size:clamp(28px,3vw,42px)}
.lecture .section-index{color:var(--blue)}
html[data-theme=dark]{color-scheme:dark;--navy:#f3f5f8;--text:#d8dde6;--muted:#a8b1c0;--blue:#71adff;
  --line:#2a3444;--paper:#111621;--surface:#0d1119;--bg:#06080d;--tint:#132035;--shadow:none}
html[data-theme=dark] .hero{background:linear-gradient(180deg,#0b0f17,#06080d)}
html[data-theme=dark] .picture-figure img,
html[data-theme=dark] .diagram-inline{background:#fff}
html[data-theme=dark] .player img{background:#fff}
html[data-theme=dark] .callout{border-color:#274568}
html[data-theme=dark] .badge,html[data-theme=dark] th{background:#0f1622}
html[data-theme=dark] code{color:#9fd0ff;background:#12203a}
@media(max-width:1080px){.nav{order:3;flex-basis:100%}.section-index{display:none}}
@media(max-width:620px){.wrap{padding:0 16px}.nav a{padding:6px 8px}.lecture-bar{gap:6px}}
@media print{
  .header,.lecture-bar,.actions,.nav{display:none!important}
  body{background:#fff}
  section.content{display:block!important;break-inside:avoid;padding:18px 0}
  .picture-figure{max-width:100%}
  .notes{display:none}
}
"""

JS = """
const $=(s,r=document)=>r.querySelector(s);
const $$=(s,r=document)=>Array.from(r.querySelectorAll(s));
const sections=$$('section.content');
let lectureMode=false,sectionIndex=0;

function setTheme(dark){
  document.documentElement.dataset.theme=dark?'dark':'light';
  $('#theme').setAttribute('aria-pressed',String(dark));
  $('#theme').textContent=dark?'Theme: dark':'Theme: light';
  try{localStorage.setItem('flowdec-deck-theme',dark?'dark':'light')}catch(e){}
}
$('#theme').onclick=()=>setTheme(document.documentElement.dataset.theme!=='dark');

function showSection(i){
  sectionIndex=Math.max(0,Math.min(sections.length-1,i));
  sections.forEach((s,j)=>s.classList.toggle('lecture-current',j===sectionIndex));
  $('#lectureCount').textContent=(sectionIndex+1)+' / '+sections.length;
  $('#prevSection').disabled=sectionIndex===0;
  $('#nextSection').disabled=sectionIndex===sections.length-1;
  setNavActive(sections[sectionIndex].id);
  window.scrollTo({top:0,behavior:'instant'});
}
function setLecture(enabled){
  lectureMode=enabled;
  document.body.classList.toggle('lecture',enabled);
  $('#lecture').setAttribute('aria-pressed',String(enabled));
  $('#lecture').textContent=enabled?'Exit lecture mode':'Lecture mode';
  if(enabled){showSection(sectionIndex);}
  else{sections.forEach(s=>s.classList.remove('lecture-current'));
       sections[sectionIndex].scrollIntoView({behavior:'instant',block:'start'});}
}
$('#lecture').onclick=()=>setLecture(!lectureMode);
$('#prevSection').onclick=()=>showSection(sectionIndex-1);
$('#nextSection').onclick=()=>showSection(sectionIndex+1);
$('#exitLecture').onclick=()=>setLecture(false);

document.addEventListener('keydown',e=>{
  if(!lectureMode)return;
  if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;
  if(e.key==='Escape'){setLecture(false);return}
  if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();showSection(sectionIndex+1)}
  if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();showSection(sectionIndex-1)}
  if(e.key==='Home'){e.preventDefault();showSection(0)}
  if(e.key==='End'){e.preventDefault();showSection(sections.length-1)}
});

function setNavActive(id){$$('.nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+id))}
$$('.nav a').forEach(a=>a.addEventListener('click',e=>{
  const target=document.getElementById(a.hash.slice(1));
  if(!target)return;
  if(lectureMode){e.preventDefault();showSection(sections.indexOf(target))}
  else{setNavActive(a.hash.slice(1))}
}));
if('IntersectionObserver' in window){
  const observer=new IntersectionObserver(entries=>{
    entries.forEach(en=>{if(en.isIntersecting&&!lectureMode)setNavActive(en.target.id)});
  },{rootMargin:'-12% 0px -70% 0px'});
  sections.forEach(s=>observer.observe(s));
}
try{setTheme(localStorage.getItem('flowdec-deck-theme')==='dark')}catch(e){setTheme(false)}
setUpVideoAudio();

// PowerPoint-style fallback: some browsers refuse to play mp3 in a plain <audio>
// element; a short retry on error is enough to surface the problem to the speaker.
function setUpVideoAudio(){
  $$('audio').forEach(el=>el.addEventListener('error',()=>{
    el.insertAdjacentHTML('afterend','<p class="player-note">Audio file could not be loaded.</p>');
  }));
}

// Marks the page as fully initialised, which makes the deck verifiable from a
// headless browser (and costs nothing in a normal one).
document.documentElement.dataset.deckReady='1';
"""


@beartype
def media_manifest() -> list[str]:
    """Every asset the HTML deck references, as project-root-relative paths."""
    paths: list[str] = []
    for slide in sc.build_slides():
        for block in slide.blocks:
            if isinstance(block, sc.Picture):
                paths.append(block.path)
            elif isinstance(block, sc.Audio):
                for slug in block.clips:
                    clip = sc.AUDIO_BY_SLUG[slug]
                    paths += [clip.mp3, clip.wave]
    return sorted(set(paths))


@beartype
def render_nav(slides: list[sc.Slide]) -> str:
    links = "".join(
        f'<a href="#{s.sid}">{s.nav}</a>' for s in slides if s.sid != "title"
    )
    return f'<nav class="nav" aria-label="Sections">{links}</nav>'


@beartype
def render_section(index: int, slide: sc.Slide) -> str:
    parts: list[str] = [f'<section class="content" id="{slide.sid}">']
    parts.append('  <div class="section-top">')
    parts.append("    <div>")
    parts.append(f'      <div class="eyebrow">{sc.md_to_html(slide.eyebrow)}</div>')
    if slide.required:
        parts.append(f'      <div class="required">{REQUIRED_LABEL}: {sc.md_to_html(slide.required)}</div>')
    parts.append(f"      <h2>{sc.md_to_html(slide.title)}</h2>")
    parts.append(f'      <p class="sub">{sc.md_to_html(slide.subtitle)}</p>')
    parts.append("    </div>")
    parts.append(f'    <div class="section-index">{index:02d}</div>')
    parts.append("  </div>")

    for block in sorted_blocks(slide):
        if isinstance(block, sc.Picture) and block.path == sc.DIAGRAM_PNG:
            parts.append("  " + inline_diagram(sc.DIAGRAM_SVG))
        else:
            parts.append("  " + dispatch(block).replace("\n", "\n  "))

    if slide.notes:
        parts.append(
            '  <details class="notes"><summary>Speaker notes</summary>'
            f"<p>{sc.md_to_html(slide.notes)}</p></details>"
        )
    parts.append("</section>")
    return "\n".join(parts)


@beartype
def build_html() -> str:
    slides = sc.build_slides()
    body = "\n".join(render_section(i, s) for i, s in enumerate(slides, start=1))
    required = [s.required for s in slides if s.required]

    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="FlowDec (ICLR 2025) course project presentation: paper review and independent testing on our own 48 kHz audio.">
<title>FlowDec — course project presentation (AMLT ITMO 2026)</title>
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#verdict" style="position:fixed;top:-60px">Skip to content</a>

<header class="header"><div class="wrap">
  <a class="brand" href="#title"><span class="brand-mark">&#9698;</span>FlowDec <small>/ AMLT 2026</small></a>
  {render_nav(slides)}
  <div class="actions">
    <button id="theme" aria-pressed="false">Theme: light</button>
    <button id="lecture" aria-pressed="false">Lecture mode</button>
  </div>
</div></header>

<main>
  <div class="hero"><div class="wrap">
    <div class="eyebrow">Course project &middot; Advanced Machine Learning &middot; ITMO 2026</div>
    <h1>FlowDec: a flow-based full-band<br>general audio codec</h1>
    <p>Paper review and independent testing on our own 48 kHz audio &mdash; a non-adversarial codec with a
       conditional-flow-matching postfilter that samples in 6 network evaluations.</p>
    <div class="badges">
      <span class="badge">{len(slides)} sections</span>
      <span class="badge">10 min + 5 min Q&amp;A</span>
      <span class="badge">{len(required)} required criteria covered</span>
      <span class="badge">{len(sc.AUDIO_CLIPS)} audio clips (A/B vs Opus)</span>
      <span class="badge">Press &larr; / &rarr; in lecture mode</span>
    </div>
  </div></div>

{body}

  <footer class="footer"><div class="wrap">
    <span>FlowDec &mdash; Welker et al., ICLR 2025. Course project, AMLT ITMO 2026. Slides in English, matching report/report.md.</span>
    <span>Generated by scripts/build_slides_html.py from scripts/slides_content.py.</span>
  </div></footer>
</main>

<div class="lecture-bar" role="group" aria-label="Lecture navigation">
  <button id="prevSection">&larr; Prev</button>
  <span id="lectureCount" class="mono">1 / {len(slides)}</span>
  <button id="nextSection">Next &rarr;</button>
  <button id="exitLecture">Exit &middot; Esc</button>
</div>

<script>{JS}</script>
</body>
</html>
"""


@beartype
def verify() -> None:
    """Fail loudly if the deck references an asset that was never built."""
    missing = [p for p in media_manifest() if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(
            "missing assets (run scripts/build_slide_assets.py first):\n  " + "\n  ".join(missing)
        )


@beartype
def main() -> None:
    verify()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.1f} KB, {len(sc.build_slides())} sections)")


if __name__ == "__main__":
    main()

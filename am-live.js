/* active_monkey — live loader.
   The lab notebook (EXPERIMENTS.md) is append-only and never stops growing.
   This module fetches the raw markdown from GitHub at load time, parses every
   "## Exp N — title" block, auto-classifies win/wall/partial, and MERGES with the
   curated snapshot in experiments-data.js (curated entries keep their polished
   one-liners / metrics / chapter; newly-appended experiments flow in automatically).
   If the fetch is blocked (offline / sandbox), it silently falls back to the snapshot.

   API (plain JS, on window):
     AM_syncLive() -> Promise<{ experiments, tally, live:boolean, count:number, error?:string }>
     AM_tally(list) -> { total, win, wall, partial, from, to }
*/
(function(){
  const RAW_URL = "https://raw.githubusercontent.com/Mirrorad1/active_monkey/main/EXPERIMENTS.md";
  const KNOWN_MAX = 35; // anything beyond this is treated as a freshly-appended experiment

  function clean(s){
    return (s||"")
      .replace(/\s*\n\s*/g," ")
      .replace(/\s{2,}/g," ")
      .replace(/->/g,"→")
      .replace(/\s+([.,;:])/g,"$1")
      .trim();
  }
  function firstSentence(s, cap=190){
    if(!s) return "";
    let cut = s.search(/\.\s/);
    let out = cut>0 ? s.slice(0,cut+1) : s;
    if(out.length>cap){ out = out.slice(0,cap).replace(/[ ,;:]+\S*$/,"")+"…"; }
    return out;
  }
  function parseBullets(body){
    const lines = body.split("\n");
    const bullets=[]; let cur=null;
    for(const ln of lines){
      if(/^\s*[-*]\s+/.test(ln)){
        if(cur) bullets.push(cur);
        cur = ln.replace(/^\s*[-*]\s+/,"");
      } else if(cur!==null && ln.trim() && !/^\s*##/.test(ln)){
        cur += " "+ln.trim();
      }
    }
    if(cur) bullets.push(cur);
    return bullets.map(b=>{
      const m = b.match(/^([A-Za-z][A-Za-z0-9 /&'’()-]*?):\s*(.*)$/);
      return m ? {key:m[1].trim(), val:m[2]} : {key:"", val:b};
    });
  }
  function classify(status, title){
    const s = (status+" "+title).toLowerCase();
    if(/\bnegative\b|\bfails?\b|collapse|culprit|\bwall\b|stalls?/.test(s)) return "wall";
    if(/\bpartial\b|progress|scaffolding|sharpening|confound|limits?\b/.test(s)) return "partial";
    if(/\bpositive\b|milestone|decisive|\bclean\b|capstone|\bworks?\b/.test(s)) return "win";
    return "win";
  }
  function parse(md){
    const blocks = md.split(/\n(?=##\s)/);
    const exps=[];
    for(const blk of blocks){
      const m = blk.match(/^##\s+Exp\s+(\d+)\s*[—\-–]\s*([^\n]+)/);
      if(!m) continue;
      const n = parseInt(m[1],10);
      let rawTitle = m[2].trim();
      const status = (rawTitle.match(/\(([^)]*)\)\s*$/)||[])[1] || "";
      let title = rawTitle.replace(/\s*\([^)]*\)\s*$/,"").trim();
      title = title.charAt(0).toUpperCase()+title.slice(1);
      if(!/[.?!…]$/.test(title)) title += ".";
      const bullets = parseBullets(blk.slice(m[0].length));
      const get = key => { const b = bullets.find(x=>new RegExp("^"+key,"i").test(x.key)); return b?clean(b.val):""; };
      let setup=get("Setup"), result=get("Result"), implication=get("Implication");
      // Fallback for short / untagged entries (e.g. one-line "consolidation" notes
      // that don't use Setup:/Result:/Implication: bullets): keep them anyway so the
      // count stays honest, and surface their body text under "What happened".
      const bodyText = clean(bullets.map(b=>(b.key?b.key+": ":"")+b.val).join(" "));
      if(!setup && !result && !implication){
        if(!bodyText) continue;     // truly empty block -> not a real experiment
        result = bodyText;
      }
      exps.push({
        n, title, setup, result, implication,
        kind: classify(status,title+" "+bodyText),
        one: firstSentence(implication) || firstSentence(result) || firstSentence(bodyText) || title,
        source:"live"
      });
    }
    exps.sort((a,b)=>a.n-b.n);
    return exps;
  }
  function merge(parsed, curated){
    const cByN={}; curated.forEach(e=>cByN[e.n]=e);
    const out = parsed.map(p=> cByN[p.n] ? cByN[p.n] : {...p, chapter:"frontier"});
    curated.forEach(c=>{ if(!out.find(o=>o.n===c.n)) out.push(c); });
    out.sort((a,b)=>a.n-b.n);
    return out;
  }
  function tally(list){
    const base = window.AM_TALLY || {from:4.81,to:1.61};
    return {
      total:list.length,
      win:list.filter(e=>e.kind==="win").length,
      wall:list.filter(e=>e.kind==="wall").length,
      partial:list.filter(e=>e.kind==="partial").length,
      from:base.from, to:base.to
    };
  }

  async function AM_syncLive(){
    const curated = (window.AM_EXPERIMENTS||[]).slice();
    try{
      const ctrl = new AbortController();
      const to = setTimeout(()=>ctrl.abort(), 7000);
      const res = await fetch(RAW_URL, {signal:ctrl.signal, cache:"no-store"});
      clearTimeout(to);
      if(!res.ok) throw new Error("HTTP "+res.status);
      const md = await res.text();
      const parsed = parse(md);
      if(!parsed.length) throw new Error("no experiments parsed");
      const experiments = merge(parsed, curated);
      return { experiments, tally:tally(experiments), live:true, count:experiments.length };
    }catch(err){
      return { experiments:curated, tally:tally(curated), live:false, count:curated.length, error:String(err&&err.message||err) };
    }
  }

  window.AM_syncLive = AM_syncLive;
  window.AM_tally = tally;
})();

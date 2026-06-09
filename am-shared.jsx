/* active_monkey — shared React components & hooks (exported to window).
   Loaded after React + Babel on every page. */

const { useState, useEffect, useRef, useCallback } = React;

/* ---------- palette / tweak application ---------- */
const AM_PALETTES = {
  growth:  { accent:"#2f6b4a", accentDeep:"#23543a", win:"#2f6b4a", winDeep:"#23543a", wall:"#5b6a7f", partial:"#ad7a2b" },
  compass: { accent:"#2c5d86", accentDeep:"#224b6e", win:"#2c5d86", winDeep:"#224b6e", wall:"#8a6a52", partial:"#b0832a" },
  ember:   { accent:"#bf6a35", accentDeep:"#a1542a", win:"#356b4d", winDeep:"#28543b", wall:"#6c6456", partial:"#bf6a35" },
};
const AM_FONTS = {
  editorial: { serif:"'Newsreader',Georgia,serif", mono:"'IBM Plex Mono',ui-monospace,monospace" },
  grotesk:   { serif:"'Fraunces',Georgia,serif",    mono:"'Spline Sans Mono',ui-monospace,monospace" },
};
function applyTweaks(t){
  const r = document.documentElement;
  const p = AM_PALETTES[t.palette] || AM_PALETTES.growth;
  r.style.setProperty('--accent',p.accent);
  r.style.setProperty('--accent-deep',p.accentDeep);
  r.style.setProperty('--win',p.win);
  r.style.setProperty('--win-deep',p.winDeep);
  r.style.setProperty('--wall',p.wall);
  r.style.setProperty('--partial',p.partial);
  const f = AM_FONTS[t.fontPair] || AM_FONTS.editorial;
  r.style.setProperty('--serif',f.serif);
  r.style.setProperty('--mono',f.mono);
  document.body.classList.toggle('no-grid', t.gridPaper === false);
  document.body.dataset.motion = t.motion === false ? 'off' : 'on';
}

/* ---------- hooks ---------- */
function useInView(opts){
  const ref = useRef(null);
  const [seen,setSeen] = useState(false);
  useEffect(()=>{
    const el = ref.current; if(!el) return;
    if(document.body.dataset.motion === 'off'){ setSeen(true); return; }
    const io = new IntersectionObserver(([e])=>{
      if(e.isIntersecting){ setSeen(true); io.disconnect(); }
    },{ threshold: opts?.threshold ?? .18, rootMargin: opts?.rootMargin ?? '0px 0px -8% 0px' });
    io.observe(el);
    return ()=>io.disconnect();
  },[]);
  return [ref,seen];
}

function useCountUp(target, run, dur=1100, decimals=0){
  const [v,setV] = useState(0);
  useEffect(()=>{
    if(!run){ return; }
    if(document.body.dataset.motion === 'off'){ setV(target); return; }
    let raf, t0;
    const step = (ts)=>{
      if(!t0) t0 = ts;
      const p = Math.min(1,(ts-t0)/dur);
      const e = 1-Math.pow(1-p,3);
      setV(target*e);
      if(p<1) raf = requestAnimationFrame(step);
      else setV(target);
    };
    raf = requestAnimationFrame(step);
    return ()=>cancelAnimationFrame(raf);
  },[run,target]);
  return decimals ? v.toFixed(decimals) : Math.round(v);
}

/* ---------- nav ---------- */
function Nav({active}){
  const link = (href,label,key,cls="")=>(
    <a href={href} className={(active===key?"active ":"")+cls}>{label}</a>
  );
  return (
    <nav className="nav">
      <div className="nav-in">
        <a className="brand" href="index.html">
          <img className="logo" src="assets/monkey-180.png" alt="active_monkey" width="26" height="26"/><span><b>active</b>_monkey</span>
        </a>
        <div className="nav-links">
          {link("index.html","Home","home")}
          {link("journey.html","The Journey","journey")}
          {link("open_problem.html","The Open Problem","open")}
          <a className="gh" href="https://github.com/Mirrorad1/active_monkey" target="_blank" rel="noopener">GitHub ↗</a>
        </div>
      </div>
    </nav>
  );
}

/* ---------- footer ---------- */
function Footer(){
  return (
    <footer className="foot">
      <div className="foot-in">
        <span>active_monkey · <span className="muted">free energy = surprise = the reward</span></span>
        <span>the trail lives in <a href="https://github.com/Mirrorad1/active_monkey/blob/main/EXPERIMENTS.md" target="_blank" rel="noopener">EXPERIMENTS.md</a> — a little mind, coming online.</span>
      </div>
    </footer>
  );
}

/* ---------- animated surprise plot ----------
   series: array of bits/char values (descending). Draws on reveal. */
function SurprisePlot({series, height=130, showAxis=true, eager=false}){
  const [ref,seenRaw] = useInView({threshold:.12});
  const seen = seenRaw || eager;
  const w = 600, h = height, padL = 4, padR = 4, padT = 12, padB = 14;
  const max = Math.max(...series), min = Math.min(...series);
  const xs = (i)=> padL + (i/(series.length-1))*(w-padL-padR);
  const ys = (v)=> padT + (1-(v-min)/(max-min))*(h-padT-padB);
  const pts = series.map((v,i)=>[xs(i),ys(v)]);
  const line = pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
  const area = line + ` L ${xs(series.length-1).toFixed(1)} ${h-padB} L ${padL} ${h-padB} Z`;
  const [pathRef,setLen] = [useRef(null), null];
  const dashRef = useRef(null);
  useEffect(()=>{
    const el = pathRef.current; if(!el) return;
    const L = el.getTotalLength();
    el.style.strokeDasharray = L;
    if(!seen){ el.style.strokeDashoffset = L; return; }
    if(document.body.dataset.motion==='off'){ el.style.transition='none'; el.style.strokeDashoffset=0; return; }
    el.style.transition='none'; el.style.strokeDashoffset=L;
    requestAnimationFrame(()=>{ requestAnimationFrame(()=>{ el.style.transition='stroke-dashoffset 1.6s cubic-bezier(.4,0,.2,1)'; el.style.strokeDashoffset=0; }); });
  },[seen]);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{display:'block',width:'100%',height:h}}>
        <defs>
          <linearGradient id="sgrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--accent)" stopOpacity=".18"/>
            <stop offset="1" stopColor="var(--accent)" stopOpacity="0"/>
          </linearGradient>
        </defs>
        {showAxis && [0.5,0.5].map((_,i)=>{
          const y = padT + (i+1)/3*(h-padT-padB);
          return <line key={i} x1="0" y1={y} x2={w} y2={y} stroke="var(--line-soft)" strokeWidth="1" strokeDasharray="2 4"/>;
        })}
        <path d={area} fill="url(#sgrad)" opacity={seen?1:0} style={{transition:'opacity 1.2s ease .4s'}}/>
        <path ref={pathRef} d={line} fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round"/>
        <circle cx={xs(0)} cy={ys(series[0])} r="3.4" fill="var(--accent)" opacity={seen?1:0} style={{transition:'opacity .4s'}}/>
        <circle cx={xs(series.length-1)} cy={ys(series[series.length-1])} r="3.4" fill="var(--accent)" opacity={seen?1:0} style={{transition:'opacity .4s 1.4s'}}/>
      </svg>
    </div>
  );
}

/* ---------- live experiments ----------
   Starts from the bundled snapshot (instant), then pulls EXPERIMENTS.md live and
   re-renders with any newly-appended experiments. status: loading|live|snapshot. */
function useLiveExperiments(){
  const [state,setState] = useState(()=>({
    exps: window.AM_EXPERIMENTS || [],
    tally: (window.AM_tally ? window.AM_tally(window.AM_EXPERIMENTS||[]) : (window.AM_TALLY||{total:0,win:0,wall:0,partial:0})),
    status: "loading"
  }));
  useEffect(()=>{
    let alive = true;
    if(!window.AM_syncLive){ setState(s=>({...s,status:"snapshot"})); return; }
    window.AM_syncLive().then(r=>{
      if(!alive) return;
      setState({ exps:r.experiments, tally:r.tally, status:r.live?"live":"snapshot" });
    });
    return ()=>{ alive=false; };
  },[]);
  return state;
}

function SyncChip({status, count}){
  const label = status==="loading" ? "syncing…" : status==="live" ? "live" : "snapshot";
  return (
    <span className="live" title={status==="live"
        ? "Pulled live from EXPERIMENTS.md — newly-appended experiments appear automatically."
        : status==="loading" ? "Fetching the latest from EXPERIMENTS.md…"
        : "Showing the bundled snapshot (live fetch unavailable here)."}>
      <span className="ld" style={status==="loading"?{opacity:.5}:undefined}></span>
      A mind, running
      <span style={{color:"var(--ink-3)",fontWeight:400,letterSpacing:".02em"}}>· {label} · {count} exp</span>
    </span>
  );
}

Object.assign(window,{ applyTweaks, useInView, useCountUp, Nav, Footer, SurprisePlot, AM_PALETTES, useLiveExperiments, SyncChip });

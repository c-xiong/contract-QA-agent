// Dependency-free UI regression checks with synthetic data and a mocked DOM.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('app/api/static/index.html', 'utf8');
const script = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].at(-1)[1];
const elements = new Map();
class Element {
  constructor(id='') { this.id=id; this.dataset={}; this.style={}; this.children=[]; this.listeners={}; this.value=''; this.disabled=false; this.textContent=''; this.clientWidth=1392; this.classes=new Set(); this.classList={add:(c)=>this.classes.add(c),remove:(c)=>this.classes.delete(c),contains:(c)=>this.classes.has(c)}; }
  set innerHTML(s) { this.html=s; this.children=[]; for (const m of s.matchAll(/id="([^"]+)"/g)) elements.set(m[1], new Element(m[1])); for (const m of s.matchAll(/id="([^"]+)"[^>]*data-state="([^"]+)"/g)) elements.get(m[1]).dataset.state=m[2]; }
  get innerHTML() {return this.html || '';}
  addEventListener(type, fn) {this.listeners[type]=fn;}
  setAttribute(k,v) {this[k]=v;}
  removeAttribute(k) {delete this[k];}
  appendChild(e) {this.children.push(e);}
  querySelector() {return new Element();}
  querySelectorAll() {return [];}
  insertAdjacentHTML(_,text) {this.html=(this.html||'')+text;}
  get options() {return this.children;}
  focus() {}
}
for (const m of html.matchAll(/id="([^"]+)"/g)) elements.set(m[1], new Element(m[1]));
const streams=[];
class EventSource {
  constructor(url) {this.url=url; this.listeners={}; streams.push(this);}
  addEventListener(type,fn) {this.listeners[type]=fn;}
  close() {this.closed=true;}
  emit(type,data) {this.listeners[type]?.({data:JSON.stringify(data)});}
}
const context = vm.createContext({document:{getElementById:id=>elements.get(id), createElement:()=>new Element(), querySelector:()=>new Element(), documentElement:new Element()}, window:{matchMedia:()=>({matches:true,addEventListener(){}}),addEventListener(){}}, localStorage:{setItem(){}}, fetch:()=>new Promise(()=>{}), EventSource, URLSearchParams, console});
vm.runInContext(script, context);
const el=id=>elements.get(id);
el('arm').appendChild(Object.assign(new Element(),{value:'rrf_hybrid_rerank'}));
el('example').listeners.click();
assert.equal(el('docs').value,''); assert(!el('mode')); assert.equal(el('arm').value,'rrf_hybrid_rerank');
el('form').listeners.submit({preventDefault(){}});
assert(!new URL('http://local'+streams[0].url).searchParams.has('document_ids'));
assert.equal(new URL('http://local'+streams[0].url).searchParams.get('mode'),'v2');
assert.equal(el('ask').textContent,'Stop');
// Synthetic fixtures exercise rendering only; these are not evaluation questions.
const evidence = Array.from({length:5}, (_,i)=>({source_chunk_id:`synthetic-${i}`, document_id:'doc-900', page_number:i+1, section_id:String(i+1), excerpt:'Synthetic insurance excerpt [***]', pulled_by:'search'}));
const usage = {searches_used:1,max_searches:3,tool_calls:2,max_tool_calls:12,model_calls:3,max_model_calls:12,evidence_items:5,max_evidence_items:20,retries:0,repair_attempts:0,max_repair_attempts:1,input_tokens:100,output_tokens:20,deadline_seconds:120};
const citations = [{...evidence[0],index:1,internal:'[doc-900, p. 1, §1]',document_title:'Synthetic contract'}];
const fixture = [['start',{agent_mode:'v2',live_model:false,documents:1,chunks:5,retriever:'bm25'}]];
['initialize','decide','authorize','execute','decide','authorize','write','verify','terminal'].forEach((node,i)=>fixture.push(['progress',{
  ...usage, limits:usage, run_id:'synthetic-run', sequence:i+1, total_ms:10*i, node,
  event_type:node==='initialize'?'run_started':node==='terminal'?'run_completed':node==='execute'?'tool_started':'node_completed',
  status:node==='terminal'?'completed':'researching', evidence_ids:[],evidence_refs:[],
  tool:node==='execute'?'document_search':null,
}]));
fixture.push(['done',{status:'completed',total_ms:100,steps:9,run_id:'synthetic-run',stop_reason:'verified',usage,evidence,citations,citation_errors:[],answer:'Synthetic coverage $[***] before sale and $[***] after sale [doc-900, p. 1, §1].'}]);
const frames = process.env.UI_REPLAY ? JSON.parse(fs.readFileSync(process.env.UI_REPLAY)) : fixture;
for (const [type,payload] of frames) streams[0].emit(type,payload);
assert.match(el('status').textContent,/completed/);
assert(!/NaN|undefined/.test(el('budget').innerHTML+el('steps').textContent));
assert.match(el('budget').innerHTML,/2 \/ 12/);
assert.equal(el('n-terminal').dataset.state,'terminal');
for (const node of ['initialize','decide','authorize','execute','write','verify']) assert.equal(el(`n-${node}`).dataset.state,'done');
assert.equal(el('n-repair').dataset.state,'pending');
for (const edge of ['v-init-decide','v-decide-authorize','v-authorize-execute','v-execute-decide','v-authorize-write','v-write-verify','v-verify-terminal']) assert(el(edge).classList.contains('on'),edge);
assert(!el('v-verify-repair').classList.contains('on'));
assert.match(el('route-history').textContent,/execute → decide → authorize → write/);
assert.equal(el('ask').textContent,'Ask'); assert(streams[0].closed);
assert(el('log').children.some(e=>e.innerHTML.includes('document_search')));
assert.match(el('ev-sum').textContent,/5 admitted · 1 cited/);
assert(!el('answer').innerHTML.includes('<em>'));
assert(!el('answer').innerHTML.includes('<strong>'));
assert.match(el('answer').innerHTML,/&#42;&#42;&#42;/);
assert.match(el('answer').innerHTML,/Source redaction/);
const finishedStrip=el('strip').innerHTML, finishedMeta=el('run-meta').textContent;
el('example').listeners.click();
assert.equal(el('strip').innerHTML,finishedStrip);
assert.equal(el('n-terminal').dataset.state,'terminal');
assert.equal(el('run-meta').textContent,finishedMeta);
assert.equal(streams.length,1);
assert.equal(vm.runInContext('renderInlineMarkdown("<script>alert(1)</script> $[***] and $[***] **bold**")',context), '&lt;script&gt;alert(1)&lt;/script&gt; $[&#42;&#42;&#42;] and $[&#42;&#42;&#42;] <strong>bold</strong>');
vm.runInContext('paintBudget({input_tokens:null,output_tokens:null})',context);
assert.match(el('budget').innerHTML,/unknown/);
el('form').listeners.submit({preventDefault(){}});
el('form').listeners.submit({preventDefault(){}});
assert(streams[1].closed); assert.equal(el('status').textContent,'cancelled');
streams[1].emit('done',{}); assert.equal(el('status').textContent,'cancelled');
el('form').listeners.submit({preventDefault(){}});
assert.equal(new URL('http://local'+streams[2].url).searchParams.get('mode'),'v2');
streams[2].emit('error', {detail:'test disconnect'});
assert(streams[2].closed); assert.equal(el('status').textContent,'error');
assert.equal(streams.length,3);
// Even when no progress update reached the UI, completion must restore only the recorded route.
const path=[];
for (const [kind,p] of frames) if (kind==='progress' && path.at(-1)!==p.node) path.push(p.node);
el('form').listeners.submit({preventDefault(){}});
streams[3].emit('done',{...frames.at(-1)[1],execution_path:path});
assert.equal(el('n-decide').dataset.state,'done');
assert.equal(el('n-repair').dataset.state,'pending');
assert(el('v-execute-decide').classList.contains('on'));
assert.equal(el('b-decide').textContent,2);
assert.equal(el('route-history').textContent,`Actual route · completed · ${path.join(' → ')}`);
// An early abstention must not invent a search, write, verify, or repair branch.
el('form').listeners.submit({preventDefault(){}});
streams[4].emit('done',{...frames.at(-1)[1],status:'abstained',execution_path:['initialize','decide','terminal'],verification_ok:false});
assert.equal(el('n-terminal').dataset.state,'abstained');
for (const node of ['execute','write','verify','repair']) assert.equal(el(`n-${node}`).dataset.state,'pending');
assert(!el('v-authorize-execute').classList.contains('on'));
assert(el('v-stop').classList.contains('on'));
console.log('UI event replay passed: all-document example, V2 completion, budget, unknown tokens, cancellation, stale events, V2-only requests, redactions, preserved completed graph, cited/context distinction, no reconnect. DOM is mocked; this is not visual browser QA.');

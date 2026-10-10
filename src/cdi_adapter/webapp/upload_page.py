"""The admin upload screen: the front desk enters the token and mobile number, adds the pages of one paper, and reads the result.

No reviewer or FHIR links: those screens are closed unless the owner switches them on (webapp/surface.py). The page's script
is the upload screen's own (limits from ``GET api/upload/limits``, the same Send retried gives the same job, nothing is
resized in the browser).

* The upload section appears only when both the token and a valid mobile number are filled. If prescriptions are already
  uploaded for that mobile number the screen says so and waits for "Proceed".
* Results are shown by PATIENT (name + mobile number), never by batch. Every group and every section is collapsible.
* With thousands of prescriptions there is no list to pick from: type part of a mobile number (or a name) and choose.
* The lab-name mapping table (many written names -> one standard test) is shown, and can be added to, at the bottom.
"""
from .page import _UPLOAD_CSS
from .theme import BRAND_CSS, HEADER_HTML, UPLOAD_ICON

# the same header without the links to the screens that are closed
_HEADER = HEADER_HTML[:HEADER_HTML.index("<nav")] + "</header>" + chr(10)

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prescription reader — admin upload</title>
<style>%%CSS%%</style>
<style>
#camdlg{border:none;border-radius:16px;padding:16px;width:min(96vw,760px);box-shadow:0 20px 60px rgba(0,0,0,.35)}
#camdlg::backdrop{background:rgba(15,23,42,.6)}
#camdlg video{width:100%;max-height:70vh;border-radius:12px;background:#000;display:block}
.camrow{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap;align-items:center}
.fld{display:flex;flex-direction:column;gap:4px;min-width:200px;flex:1}
.fld label{font-weight:600;font-size:13px;color:var(--muted,#5b6b8c)}
.fld input[type=text]{width:100%;min-height:44px;font-size:16px}
.row2{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.suggest{position:relative}
.suggest ul{position:absolute;left:0;right:0;top:100%;z-index:30;background:#fff;border:1px solid var(--line-strong,#cfd9ec);border-radius:10px;
  box-shadow:0 8px 24px rgba(16,42,90,.14);list-style:none;margin:4px 0 0;padding:4px;max-height:300px;overflow:auto}
.suggest li{padding:9px 10px;border-radius:8px;cursor:pointer;overflow-wrap:anywhere}
.suggest li[aria-selected=true],.suggest li:hover{background:var(--blue-50,#eef4ff)}
.notice-warn{background:var(--warn-bg,#fdf5e6);border:1px solid var(--warn-line,#f0dcae);border-radius:12px;padding:12px 14px;margin-top:12px}
.notice-warn ul{margin:6px 0 10px;padding-left:20px}
.fielderr{color:var(--err,#d64545);font-size:13px;min-height:18px;margin-top:6px}
.ptsum{font-weight:600;margin:0 0 10px}
details.cf-det{border:1px solid var(--line,#e4eaf4);border-radius:12px;margin-top:10px;background:var(--surface,#fff)}
details.cf-det>summary{cursor:pointer;padding:10px 14px;list-style:none;display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-height:44px}
details.cf-det>summary::-webkit-details-marker{display:none}
details.cf-det>summary::before{content:"▸";color:var(--muted,#5b6b8c);transition:transform .15s}
details.cf-det[open]>summary::before{transform:rotate(90deg)}
details.cf-det>summary:focus-visible{outline:none;box-shadow:var(--ring,0 0 0 3px rgba(29,78,216,.3));border-radius:12px}
details.cf-det>.det-body{padding:2px 14px 14px}
details.sec{margin-top:8px}
details.sec>summary{font-weight:650;font-size:15px;min-height:40px}
.jobbox .jerr{color:var(--danger,#b42318);font-size:14px;margin:4px 0}
.sumtbl table{width:100%;border-collapse:collapse;font-size:14px}
.sumtbl th,.sumtbl td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line,#e4eaf4);vertical-align:top}
.sumtbl tbody th{width:30%;color:var(--muted,#5b6b8c);font-weight:600}
.sumtbl thead th{color:var(--muted,#5b6b8c);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.3px}
.sumtbl .none,.none{color:var(--faint,#93a1bd)}
.sumtbl .note{margin:6px 0 0;font-size:13px;color:var(--muted,#5b6b8c)}
.sumtbl .tbox{overflow-x:auto}
.sumjson{margin:10px 0 6px;font-size:15px}
.nmbox .nmedit{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;align-items:center}
.nmbox .nm-in{min-height:40px;min-width:220px;flex:1;max-width:360px}
.nmbox .nmchips{margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.docthumb{display:inline-flex;gap:12px;align-items:flex-start;margin:6px 0 10px;flex-wrap:wrap}
.docthumb button{border:1px solid var(--line-strong,#cfd9ec);border-radius:10px;background:#fff;padding:4px;cursor:zoom-in;line-height:0}
.docthumb img{max-width:150px;max-height:200px;border-radius:6px;display:block}
.nmcrop{margin:6px 0}
.nmcrop button{border:1px solid var(--line-strong,#cfd9ec);border-radius:10px;background:#fff;padding:4px;cursor:zoom-in;line-height:0;max-width:100%}
.nmcrop img{max-width:100%;max-height:120px;display:block;border-radius:6px}
#imgdlg{border:none;border-radius:0;padding:0;width:100vw;height:100vh;max-width:100vw;max-height:100vh;box-shadow:none}
.vbtn{border:1px solid var(--line-strong,#cfd9ec);border-radius:8px;background:#fff;padding:3px 10px;font-size:13px;font-weight:600;cursor:zoom-in;color:var(--blue-700,#1e40af)}
#imgdlg::backdrop{background:rgba(15,23,42,.65)}
#imgdlg .bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 12px;border-bottom:1px solid var(--line,#e4eaf4);background:#fff}
#imgdlg .bar .grow{flex:1}
#imgdlg .imgwrap{overflow:auto;height:calc(100% - 58px);background:#111;text-align:center}
#imgdlg img{display:block;margin:0 auto;max-width:none;background:#fff}
.tabs{display:flex;gap:6px;border-bottom:2px solid var(--line,#e4eaf4);margin:0 0 16px;flex-wrap:wrap}
.tabs button{appearance:none;border:none;background:transparent;font:inherit;font-weight:650;font-size:15px;color:var(--muted,#5b6b8c);
  padding:12px 18px;min-height:48px;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px;border-radius:10px 10px 0 0;display:inline-flex;gap:8px;align-items:center}
.tabs button[aria-selected=true]{color:var(--blue-700,#1e40af);border-bottom-color:var(--blue-600,#1d4ed8)}
.tabs button:hover{background:var(--blue-50,#eef4ff)}
.tabs button:focus-visible{outline:none;box-shadow:var(--ring,0 0 0 3px rgba(29,78,216,.3))}
.maptbl{width:100%;border-collapse:collapse;font-size:14px}
.maptbl th,.maptbl td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line,#e4eaf4);vertical-align:top}
.maptbl .off td{opacity:.5}
.mapadd{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
.mapadd input[type=text]{min-height:40px;flex:1;min-width:150px}
@media (max-width:600px){.cf-main{padding:16px 16px 48px}}
</style>
</head>
<body>
%%HEADER%%
<div class="cf-pagetitle">
  <h1>Prescription reader</h1>
  <p class="cf-sub">Enter the token and mobile number, add the pages of the prescription, and read the result. Values marked <b>needs a check</b> are doubtful: nothing is guessed.</p>
</div>
<main class="cf-main">
  <div class="tabs" role="tablist" aria-label="Prescription reader">
    <button type="button" role="tab" id="tab-up" aria-selected="true" aria-controls="panel-up">Capture &amp; upload</button>
    <button type="button" role="tab" id="tab-ex" aria-selected="false" aria-controls="panel-ex" tabindex="-1">Extracted <span class="pill ok" id="ex-badge" hidden></span></button>
  </div>
<section id="panel-up" role="tabpanel" aria-labelledby="tab-up">
  <div class="card" id="patient-card">
    <h2>1 · Token and mobile number</h2>
    <p class="hint">Both are needed before you can upload. The result is kept under this patient (name and mobile number).</p>
    <div class="row2">
      <div class="fld"><label for="token">Token number</label>
        <input type="text" id="token" maxlength="20" autocomplete="off" inputmode="text" placeholder="for example T-17" aria-describedby="pt-err"/></div>
      <div class="fld suggest"><label for="phone">Mobile number</label>
        <input type="text" id="phone" maxlength="18" autocomplete="off" inputmode="numeric" placeholder="10 digits"
          role="combobox" aria-expanded="false" aria-controls="phone-sugg" aria-autocomplete="list" aria-describedby="pt-err"/>
        <ul id="phone-sugg" role="listbox" aria-label="Patients with this mobile number" hidden></ul></div>
    </div>
    <div class="fld" style="max-width:360px"><label for="dept">Doctor's department <span class="hint">(optional: helps read a hard investigation name)</span></label>
      <select id="dept"><option value="">not known</option><option value="ent">ENT</option><option value="cardiology">Cardiology</option><option value="neurology">Neurology</option><option value="orthopaedics">Orthopaedics</option><option value="gynaecology">Gynaecology</option><option value="ophthalmology">Ophthalmology</option><option value="paediatrics">Paediatrics</option><option value="dental">Dental</option><option value="general medicine">General medicine</option><option value="surgery">Surgery</option><option value="endocrinology">Endocrinology</option><option value="rheumatology">Rheumatology</option><option value="gastroenterology">Gastroenterology</option><option value="pulmonology">Pulmonology / chest</option><option value="urology">Urology</option><option value="dermatology">Dermatology</option></select></div>
    <div class="fielderr" id="pt-err" role="alert"></div>
    <div class="notice-warn" id="existing" hidden aria-live="polite">
      <div id="existing-msg"></div>
      <ul id="existing-list"></ul>
      <span class="muted">The upload step below is open: send this one if it is a new prescription, or more pages of the same token.</span>
    </div>
  </div>

  <div class="card" id="form-card" hidden>
    <h2>2 · Upload</h2>
    <p class="ptsum" id="pt-sum"></p>
    <p class="hint">Add <b>every page</b> of this paper: the front, the back, continuation sheets, and notes below a ruled line. If a newer visit is written on the back or lower down, its lab tests and doctor booking are the ones used; the earlier visits are listed too.</p>
    <div class="dropzone" id="dz" tabindex="0" role="button" aria-label="Choose or drop photos, scans or PDFs">
      %%ICON%%
      <div class="dz-title">Drop photos, scans or PDFs here, or click to choose</div>
      <div class="dz-sub" id="dzsub">JPG, PNG, TIFF or PDF</div>
    </div>
    <div class="up-actions">
      <button class="btn btn-ghost" type="button" id="cam">Take a picture</button>
      <button class="btn btn-ghost" type="button" id="pick">Choose files</button>
      <span class="muted" id="pgcount" aria-live="polite"></span>
    </div>
    <input type="file" id="files" multiple hidden
      accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,image/jpeg,image/png,image/tiff,application/pdf"/>
    <input type="file" id="camera" hidden accept="image/*" capture="environment"/>
    <ol class="pagelist" id="pages" aria-label="Pages to send"></ol>
    <fieldset class="grouping" id="grouping" hidden>
      <legend>These files are</legend>
      <label><input type="radio" name="grp" value="one_document" checked/> pages of <b>one prescription</b> (one result)</label>
      <label><input type="radio" name="grp" value="separate"/> <b>separate files</b> (each is read on its own)</label>
      <div class="muted" id="grphint"></div>
    </fieldset>
    <div class="up-err" id="uperr" role="alert" hidden></div>
    <div style="margin-top:16px;display:flex;gap:10px;align-items:center">
      <button class="btn btn-primary" id="go">Read prescriptions</button>
      <span id="hint" class="muted"></span>
    </div>
  </div>

  <div class="card" id="emr-card" hidden>
    <h2>Progress</h2>
    <p class="muted" style="margin:0">You can keep adding prescriptions while earlier ones are being read.</p>
    <div id="jobs"></div>
  </div>

  <div class="card" id="current-card" hidden>
    <h2>Current prescription</h2>
    <p class="hint">The prescription just read is shown here. When the next prescription is uploaded (a new token or mobile number) it moves to the <b>Extracted</b> tab.</p>
    <div id="current-body" aria-live="polite"></div>
  </div>
</section>
<section id="panel-ex" role="tabpanel" aria-labelledby="tab-ex" hidden>
  <div class="card" id="flat-card">
    <h2>Extracted details (table)</h2>
    <p class="hint">One row per prescription and lab test. Type the patient's mobile number or the token number: the rows appear as you type.</p>
    <div class="row2">
      <div class="fld"><label for="fl-phone">Mobile number</label><input type="text" id="fl-phone" inputmode="numeric" maxlength="18" autocomplete="off" placeholder="10 digits or part of it"/></div>
      <div class="fld"><label for="fl-token">Token number</label><input type="text" id="fl-token" maxlength="20" autocomplete="off" placeholder="for example T-17"/></div>
    </div>
    <div class="muted" id="fl-note" aria-live="polite"></div>
    <div style="overflow-x:auto"><table class="emr-grid" id="fl-table" hidden></table></div>
    <p><a class="btn btn-ghost btn-sm" id="fl-csv" href="#" hidden>Download all columns (CSV)</a></p>
  </div>
  <div class="card" id="patients-card">
    <h2>Extracted prescriptions</h2>
    <p class="hint">Type part of a mobile number (or a name) to find a patient. Results are grouped by patient, not by upload.</p>
    <div class="fld suggest" style="max-width:520px">
      <label for="psearch">Find a patient</label>
      <input type="text" id="psearch" autocomplete="off" placeholder="mobile number or name" role="combobox" aria-expanded="false"
        aria-controls="psearch-sugg" aria-autocomplete="list"/>
      <ul id="psearch-sugg" role="listbox" aria-label="Matching patients" hidden></ul>
    </div>
    <div id="groups" aria-live="polite"></div>
  </div>

  <div class="card" id="map-card">
    <details class="cf-det" id="map-det" style="margin-top:0">
      <summary><b>Lab test mapping table</b> <span class="muted" id="map-count"></span></summary>
      <div class="det-body">
        <p class="hint" style="margin-top:0">Many written names go to ONE standard test (for example creatinine, creatine, sr creatinine, RFT → Creatinine). Every name the reader maps is looked up here first. Add a row, or switch one off.</p>
        <div class="mapadd">
          <input type="text" id="m-alias" placeholder="written name (for example s creat)" aria-label="Written name"/>
          <input type="text" id="m-canon" placeholder="standard test (for example Creatinine)" aria-label="Standard test"/>
          <input type="text" id="m-loinc" placeholder="code (optional)" aria-label="Code" style="max-width:150px"/>
          <input type="text" id="m-note" placeholder="note (optional)" aria-label="Note"/>
          <button class="btn btn-primary btn-sm" type="button" id="m-add">Add</button>
        </div>
        <div class="fielderr" id="m-err" role="alert"></div>
        <div class="fld" style="max-width:360px"><label for="mapfilter">Filter</label><input type="text" id="mapfilter" placeholder="type to filter" autocomplete="off"/></div>
        <div class="tbox" style="overflow-x:auto;margin-top:8px"><table class="maptbl" id="maptbl"></table></div>
      </div>
    </details>
  </div>
</section>
  <p class="muted" style="text-align:center"><a href="/status">System status</a></p>
</main>

<dialog id="imgdlg" aria-label="Prescription image">
  <div class="bar">
    <b id="img-title">Prescription image</b>
    <span class="grow"></span>
    <button class="btn btn-ghost btn-sm" type="button" id="img-prev" aria-label="Previous page">‹ Page</button>
    <span class="muted" id="img-pg"></span>
    <button class="btn btn-ghost btn-sm" type="button" id="img-next" aria-label="Next page">Page ›</button>
    <button class="btn btn-ghost btn-sm" type="button" id="img-out" aria-label="Zoom out">−</button>
    <button class="btn btn-ghost btn-sm" type="button" id="img-in" aria-label="Zoom in">+</button>
    <button class="btn btn-ghost btn-sm" type="button" id="img-fit">Fit</button>
    <button class="btn btn-ghost btn-sm" type="button" id="img-view">Original photo</button>
    <button class="btn btn-primary btn-sm" type="button" id="img-close">Close</button>
  </div>
  <div class="imgwrap"><img id="img-el" alt="The prescription page"/></div>
</dialog>

<dialog id="camdlg" aria-label="Camera">
  <video id="camvideo" autoplay playsinline muted></video>
  <div class="camrow">
    <button class="btn btn-primary" type="button" id="camshot">Capture</button>
    <button class="btn btn-ghost" type="button" id="camswitch" hidden>Switch camera</button>
    <button class="btn btn-ghost" type="button" id="camdone">Done</button>
    <span class="muted" id="camcount" aria-live="polite"></span>
  </div>
</dialog>

<script>

const $=s=>document.querySelector(s);
const STAGES=["ingest","classify","ocr","extract","terminology","validate"];
const STAGE_LABEL={ingest:"Ingest",classify:"Classify",ocr:"OCR",extract:"Extract",terminology:"Terminology",validate:"Validate"};
const TICK='<svg class="tick" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';
let TIMER=null,JOBS=[];
// ---- two tabs: the whole capture / upload flow, and the Extracted list. A read prescription is sent to Extracted (collapsed).
let NEWCOUNT=0;
function showTab(name){
  const ex=name==="ex";
  $("#panel-up").hidden=ex; $("#panel-ex").hidden=!ex;
  for(const [id,on] of [["#tab-up",!ex],["#tab-ex",ex]]){ const t=$(id); t.setAttribute("aria-selected",String(on)); t.tabIndex=on?0:-1; }
  if(ex){ NEWCOUNT=0; badge();
    // the prescription just read is in the table straight away: its mobile and token number are filled in, so the tab is never empty after a read
    const done=JOBS.find(x=>x.finished&&!x.err);
    if(done&&!$("#fl-phone").value&&!$("#fl-token").value){ $("#fl-phone").value=done.phone; $("#fl-token").value=done.token; flatSearch(); } }
  try{ history.replaceState(null,"",ex?"#extracted":"#upload"); }catch(e){}
}
function badge(){ const b=$("#ex-badge"); b.hidden=!NEWCOUNT; b.textContent=NEWCOUNT?(NEWCOUNT+" new"):""; }
// ---- the flat table: one row per prescription x lab test, found by mobile number or token number ------------------
const FLAT_HEAD={intake_token_no:"Token",intake_phone:"Mobile",patient_name:"Patient",patient_name_status:"Name check",patient_age_text:"Age",patient_sex:"Sex",
  doctor_name:"Doctor",doctor_department:"Department",doctor_clinic_name:"Clinic",doc_type:"Type",result_status:"Result",latest_visit_date:"Visit date",
  lab_test_seq:"#",lab_test_as_written:"Test (as written)",lab_test_standard_name:"Standard name",lab_test_status:"Test check",lab_test_reason:"Why",
  booking_needed:"Booking needed",booking_when_text:"Booking when",booking_as_written:"Booking (as written)",document_id:"Document"};
let FLAT_TIMER=null,FLAT_SEQ=0;
function flatQuery(){ return "phone="+encodeURIComponent($("#fl-phone").value.replace(/[^0-9]/g,""))+"&token="+encodeURIComponent($("#fl-token").value.trim()); }
async function flatSearch(){
  const phone=$("#fl-phone").value.replace(/[^0-9]/g,""), token=$("#fl-token").value.trim(), tb=$("#fl-table"), note=$("#fl-note"), csv=$("#fl-csv");
  if(!phone&&!token){ tb.hidden=true; csv.hidden=true; note.textContent=""; return; }
  const my=++FLAT_SEQ; note.textContent="searching…";
  let j=null; try{ const r=await fetch("api/flat/search?"+flatQuery()); if(r.ok) j=await r.json(); }catch(e){}
  if(my!==FLAT_SEQ) return;
  if(!j){ note.textContent="The table could not be read just now. Please try again."; tb.hidden=true; csv.hidden=true; return; }
  const cols=j.columns, rows=j.rows;
  if(!rows.length){ note.textContent="Nothing found for this mobile number / token number."; tb.hidden=true; csv.hidden=true; return; }
  const docs=new Set(rows.map(r=>r.document_id)).size;
  note.textContent=docs+" prescription"+(docs===1?"":"s")+", "+rows.length+" row"+(rows.length===1?"":"s")+".";
  tb.innerHTML="<thead><tr>"+cols.map(c=>"<th>"+esc(FLAT_HEAD[c]||c)+"</th>").join("")+"</tr></thead><tbody>"
    +rows.map(r=>"<tr>"+cols.map(c=>"<td>"+(r[c]==null||r[c]===""?'<span class="none">—</span>':esc(c==="document_id"?String(r[c]).slice(0,8):r[c]))+"</td>").join("")+"</tr>").join("")+"</tbody>";
  tb.hidden=false; csv.href="api/flat/search?fmt=csv&"+flatQuery(); csv.hidden=false;
}
for(const id of ["#fl-phone","#fl-token"]) $(id).addEventListener("input",()=>{ clearTimeout(FLAT_TIMER); FLAT_TIMER=setTimeout(flatSearch,250); });
$("#tab-up").onclick=()=>showTab("up");
$("#tab-ex").onclick=()=>showTab("ex");
$(".tabs").addEventListener("keydown",e=>{
  if(e.key!=="ArrowLeft"&&e.key!=="ArrowRight") return;
  const ex=$("#tab-ex").getAttribute("aria-selected")==="true"; showTab(ex?"up":"ex"); $(ex?"#tab-up":"#tab-ex").focus();
});
function esc(s){ return String(s==null?"":s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

// ---- token and mobile number: the upload appears only when both are filled --------------------------------
function phoneDigits(v){ let d=String(v||"").replace(/[\s\-().]/g,""); if(d.startsWith("+91")) d=d.slice(3); else if(d.length===12&&d.startsWith("91")) d=d.slice(2); else if(d.length===11&&d.startsWith("0")) d=d.slice(1); return d; }
function phoneOk(d){ return /^[6-9][0-9]{9}$/.test(d); }
function tokenOk(t){ return /^[A-Za-z0-9][A-Za-z0-9\-_\/]{0,19}$/.test(t); }
function fmtPhone(p){ return p&&p.length===10?p.slice(0,5)+" "+p.slice(5):(p||""); }
function fmtDate(iso){ if(!iso) return ""; const d=new Date(iso); return isNaN(d)?"":d.toLocaleDateString([], {day:"2-digit",month:"short",year:"numeric"}); }
function plural(n,u){ return n+" "+u+(n===1?"":"s"); }
let ACK="",EXIST=null,EXIST_FOR="";                  // EXIST: what is already uploaded for EXIST_FOR (the typed number)
let SEND_KEY=null,SENDING=false;

function gate(){
  const t=$("#token").value.trim(), raw=$("#phone").value.trim(), d=phoneDigits(raw);
  const tOk=tokenOk(t), pOk=phoneOk(d);
  let err="";
  if(t&&!tOk) err="The token can use letters, numbers and - _ / only, up to 20 characters.";
  else if(raw&&(d.length>=10)&&!pOk) err="The mobile number must be 10 digits and start with 6, 7, 8 or 9.";
  $("#pt-err").textContent=err;
  const clash=tOk&&pOk&&TOKEN_CLASH.get(t+"|"+d);          // the same token already used today for another mobile number
  if(clash&&!err) err=clash;
  $("#pt-err").textContent=err;                           // (again: the token check came back after the first draw)
  const have=pOk&&EXIST_FOR===d&&EXIST&&!clash;          // the look-up for this number has come back, and the token is free
  const dup=have&&EXIST.count>0;
  $("#existing").hidden=!dup;
  if(dup){
    $("#existing-msg").innerHTML='<b>'+plural(EXIST.count,"prescription")+' already uploaded</b> for '+esc(fmtPhone(d))+'. Uploading now will make this #'+(EXIST.count+1)+'.';
    $("#existing-list").innerHTML=EXIST.prescriptions.slice(0,5).map(p=>'<li>Token '+esc(p.token_no||"—")+' · '+esc(p.patient_name||"name not read")+' · '+esc(fmtDate(p.uploaded))+' · '+esc(p.filename||"")+'</li>').join("")
      +(EXIST.count>5?'<li class="none">and '+(EXIST.count-5)+' more</li>':"");
  }
  const open=tOk&&pOk&&have;                              // the upload step opens by itself once the token and a valid mobile number are in: no button to press (an earlier upload for this number is only shown as a note)
  const wasOpen=!$("#form-card").hidden;
  $("#form-card").hidden=!open;
  $("#pt-sum").textContent=open?("Token "+t+" · mobile "+fmtPhone(d)):"";
  if(open&&!wasOpen){ try{ $("#form-card").scrollIntoView({behavior:"smooth",block:"nearest"}); }catch(e){} }
  if(!open) say("");
  render();
}
const TOKEN_CLASH=new Map(), TOKEN_ASKED=new Set();      // "token|mobile" -> the refusal text, or ""
async function checkToken(){
  const t=$("#token").value.trim(), d=phoneDigits($("#phone").value), k=t+"|"+d;
  if(!tokenOk(t)||!phoneOk(d)||TOKEN_ASKED.has(k)) return;
  TOKEN_ASKED.add(k);
  try{ const r=await fetch("api/intake/token-check?token="+encodeURIComponent(t)+"&phone="+encodeURIComponent(d)); if(r.ok){ const j=await r.json(); TOKEN_CLASH.set(k,j.conflict?j.message:""); } }catch(e){}
  gate();
}
let PHONE_SEQ=0;
async function checkExisting(){
  const d=phoneDigits($("#phone").value);
  if(!phoneOk(d)){ EXIST=null; EXIST_FOR=""; gate(); return; }
  if(EXIST_FOR===d) { gate(); return; }
  const my=++PHONE_SEQ; EXIST=null; EXIST_FOR="";
  let j={count:0,prescriptions:[]};
  try{ const r=await fetch("api/intake/existing?phone="+encodeURIComponent(d)); if(r.ok) j=await r.json(); }catch(e){}   // a failed look-up never blocks the desk
  if(my!==PHONE_SEQ) return;
  EXIST=j; EXIST_FOR=d; gate();
}
$("#token").addEventListener("input",()=>{ SEND_KEY=null; gate(); checkToken(); });
$("#phone").addEventListener("input",()=>{ SEND_KEY=null; ACK=""; gate(); checkExisting(); checkToken(); });
$("#proceed")&&($("#proceed").onclick=()=>{ ACK=phoneDigits($("#phone").value); gate(); });

// ---- autocomplete (a list of thousands is never shown: type, then choose) ---------------------------------
function suggest(input,list,fetcher,label,onPick){
  let items=[],idx=-1,timer=null,seq=0;
  function close(){ list.hidden=true; idx=-1; input.setAttribute("aria-expanded","false"); input.removeAttribute("aria-activedescendant"); }
  function draw(){
    list.innerHTML=items.map((it,i)=>'<li role="option" id="'+list.id+'-'+i+'" data-i="'+i+'" aria-selected="'+(i===idx)+'">'+label(it)+'</li>').join("");
    list.hidden=!items.length; input.setAttribute("aria-expanded",String(!!items.length));
    if(idx>=0) input.setAttribute("aria-activedescendant",list.id+"-"+idx);
  }
  function pick(i){ const it=items[i]; close(); if(it) onPick(it); }
  input.addEventListener("input",()=>{
    clearTimeout(timer);
    timer=setTimeout(async()=>{ const my=++seq; let r=[]; try{ r=await fetcher(input.value); }catch(e){} if(my!==seq) return; items=r||[]; idx=-1; draw(); },200);
  });
  input.addEventListener("keydown",e=>{
    if(list.hidden) return;
    if(e.key==="ArrowDown"){ e.preventDefault(); idx=Math.min(items.length-1,idx+1); draw(); }
    else if(e.key==="ArrowUp"){ e.preventDefault(); idx=Math.max(0,idx-1); draw(); }
    else if(e.key==="Enter"&&idx>=0){ e.preventDefault(); pick(idx); }
    else if(e.key==="Escape"){ close(); }
  });
  list.addEventListener("mousedown",e=>{ const li=e.target.closest("li"); if(li){ e.preventDefault(); pick(+li.dataset.i); } });
  input.addEventListener("blur",()=>setTimeout(close,150));
}
async function searchPatients(q){
  q=String(q||"").trim(); if(q.length<2) return [];
  const r=await fetch("api/intake/search?q="+encodeURIComponent(/[A-Za-z]/.test(q)?q:phoneDigits(q)||q)); return r.ok?(await r.json()).patients:[];
}
function patientLabel(p){ return '<b>'+esc(fmtPhone(p.phone))+'</b> · '+esc(p.name||"name not read")+' <span class="none">'+plural(p.prescriptions,"prescription")+'</span>'; }
suggest($("#phone"),$("#phone-sugg"),async q=>{ const d=phoneDigits(q).replace(/\D/g,""); return d.length<2?[]:searchPatients(d); },patientLabel,
  p=>{ $("#phone").value=p.phone; ACK=""; checkExisting(); });
suggest($("#psearch"),$("#psearch-sugg"),searchPatients,patientLabel,async p=>{ $("#psearch").value=""; await showPatient(p); });

// ---- the pages to send ---------------------------------------------------------------------------------
const dz=$("#dz"),fileInput=$("#files"),camInput=$("#camera");
// What the server will accept (settings, GET api/upload/limits). The server checks everything
// again: this only lets the screen say it sooner. Defaults are used only if that call fails.
let LIMITS={max_files:10,max_file_mb:50,max_total_mb:150,min_short_side_px:600};
const OK_TYPES=["application/pdf","image/png","image/jpeg","image/tiff"];
const OK_EXT=/\.(pdf|png|jpe?g|tiff?)$/i;
let PAGES=[],NEXT_ID=1;

function limitText(){ return "JPG, PNG, TIFF or PDF · up to "+LIMITS.max_files+" files · each up to "+LIMITS.max_file_mb+" MB"; }
$("#dzsub").textContent=limitText();
fetch("api/upload/limits").then(r=>r.ok?r.json():null).then(j=>{ if(j){ LIMITS=j; $("#dzsub").textContent=limitText(); render(); } }).catch(()=>{});

function say(msg){ const e=$("#uperr"); e.hidden=!msg; e.textContent=msg||""; }
function mb(n){ return n<1e6 ? Math.max(1,Math.round(n/1e3))+" KB" : (n/1e6).toFixed(1).replace(/\.0$/,"")+" MB"; }
function isPdf(f){ return f.type==="application/pdf"||/\.pdf$/i.test(f.name); }
function isTiff(f){ return f.type==="image/tiff"||/\.tiff?$/i.test(f.name); }

function addFiles(list){
  if(SENDING) return;
  let problem="";
  for(const f of list){
    if(!(OK_TYPES.includes(f.type)||OK_EXT.test(f.name))){ problem=problem||('Please add a photo, PDF or scan. "'+f.name+'" is not one.'); continue; }
    if(f.size>LIMITS.max_file_mb*1e6){ problem=problem||('"'+f.name+'" is larger than '+LIMITS.max_file_mb+' MB.'); continue; }
    if(PAGES.length>=LIMITS.max_files){ problem=problem||('You can send up to '+LIMITS.max_files+' files at once.'); break; }
    if(PAGES.reduce((n,p)=>n+p.file.size,0)+f.size>LIMITS.max_total_mb*1e6){ problem=problem||('Together the files are larger than '+LIMITS.max_total_mb+' MB. Please send fewer pages at once.'); continue; }
    // the file is kept EXACTLY as chosen: nothing is resized or re-compressed here
    const p={id:NEXT_ID++,file:f,url:null,w:null,h:null,warn:"",noPreview:false};
    if(!isPdf(f)&&!isTiff(f)){
      p.url=URL.createObjectURL(f);
      const im=new Image();
      im.onload=()=>{ p.w=im.naturalWidth; p.h=im.naturalHeight;
        if(Math.min(p.w,p.h)<LIMITS.min_short_side_px)
          p.warn="Small picture ("+p.w+" × "+p.h+" px): the writing may not be readable. Move closer and retake it."
        ;render(); };
      im.onerror=()=>{ p.noPreview=true; render(); };
      im.src=p.url;
    } else { p.noPreview=true; }
    PAGES.push(p);
  }
  say(problem); SEND_KEY=null; render();
}
function dropPage(i){ const p=PAGES[i]; if(p&&p.url) URL.revokeObjectURL(p.url); PAGES.splice(i,1); SEND_KEY=null; say(""); render(); }
function movePage(i,d){ const j=i+d; if(j<0||j>=PAGES.length) return; [PAGES[i],PAGES[j]]=[PAGES[j],PAGES[i]]; SEND_KEY=null; render(); }

function render(){
  const n=PAGES.length, anyPdf=PAGES.some(p=>isPdf(p.file));
  $("#pgcount").textContent=n?(n+(n===1?" page":" pages")+" added"):"";
  $("#pages").innerHTML=PAGES.map((p,i)=>{
    const thumb=(p.url&&!p.noPreview)?'<img class="thumb" src="'+p.url+'" alt="Preview of page '+(i+1)+'"/>'
      :'<div class="thumb ph" aria-hidden="true">'+(isPdf(p.file)?"PDF":isTiff(p.file)?"TIFF":"…")+'</div>';
    const dis=SENDING?" disabled":"";
    return '<li class="pg" data-i="'+i+'">'+thumb
      +'<div class="pg-body"><div class="pg-name"><b>Page '+(i+1)+'</b> · '+esc(p.file.name)+'</div>'
      +'<div class="muted">'+mb(p.file.size)+(p.w?(' · '+p.w+' × '+p.h+' px'):'')+'</div>'
      +(p.warn?'<div class="pill warn pg-warn">'+esc(p.warn)+'</div>':'')+'</div>'
      +'<div class="pg-ctl">'
      +'<button class="btn btn-ghost btn-sm" data-act="up" aria-label="Move page '+(i+1)+' up"'+(i===0||dis?" disabled":"")+'>↑</button>'
      +'<button class="btn btn-ghost btn-sm" data-act="down" aria-label="Move page '+(i+1)+' down"'+(i===n-1||dis?" disabled":"")+'>↓</button>'
      +'<button class="btn btn-ghost btn-sm" data-act="del" aria-label="Remove page '+(i+1)+'"'+(dis?" disabled":"")+'>✕</button>'
      +'</div></li>';
  }).join("");
  const g=$("#grouping"); g.hidden=n<2;
  const one=g.querySelector('input[value="one_document"]'), sep=g.querySelector('input[value="separate"]');
  one.disabled=anyPdf||SENDING; sep.disabled=SENDING;
  if(anyPdf){ sep.checked=true; }
  $("#grphint").textContent=anyPdf?"A PDF already holds all of its pages, so these are sent as separate files.":"";
  $("#go").disabled=!n||SENDING;
  for(const b of document.querySelectorAll(".up-actions .btn")) b.disabled=SENDING;
}
$("#pages").addEventListener("click",e=>{
  const b=e.target.closest("button[data-act]"); if(!b) return;
  const i=+b.closest("li").dataset.i, a=b.dataset.act;
  if(a==="up") movePage(i,-1); else if(a==="down") movePage(i,1); else dropPage(i);
});
document.querySelectorAll('#grouping input').forEach(r=>r.addEventListener("change",()=>{ SEND_KEY=null; }));

dz.onclick=()=>{ if(!SENDING) fileInput.click(); };
dz.onkeydown=e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); dz.onclick(); } };
dz.ondragover=e=>{ e.preventDefault(); dz.classList.add("drag"); };
dz.ondragleave=()=>dz.classList.remove("drag");
dz.ondrop=e=>{ e.preventDefault(); dz.classList.remove("drag"); addFiles([...e.dataTransfer.files]); };
$("#pick").onclick=()=>fileInput.click();
// A phone or tablet (touch screen): the device's own camera app via <input capture> - full-resolution photo,
// autofocus, flash. A computer: a live camera view in this page (getUserMedia needs https or localhost).
const TOUCH=window.matchMedia&&matchMedia("(pointer: coarse)").matches;
const CAN_LIVE=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia&&window.isSecureContext&&window.HTMLDialogElement);
$("#cam").onclick=()=>{ if(TOUCH||!CAN_LIVE) camInput.click(); else openCamera(); };
let STREAM=null,CAMS=[],CAMI=0,SHOTS=0;
const camVideo=$("#camvideo"),camDlg=$("#camdlg");
function stopStream(){ if(STREAM){ STREAM.getTracks().forEach(t=>t.stop()); STREAM=null; } camVideo.srcObject=null; }
async function startStream(deviceId){
  stopStream();
  const size={width:{ideal:3840},height:{ideal:2160}};
  STREAM=await navigator.mediaDevices.getUserMedia({audio:false,video:deviceId?{deviceId:{exact:deviceId},...size}:{facingMode:{ideal:"environment"},...size}});
  camVideo.srcObject=STREAM; await camVideo.play().catch(()=>{});
}
async function openCamera(){
  say(""); SHOTS=0; $("#camcount").textContent="";
  try{
    await startStream();
    CAMS=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==="videoinput");
    $("#camswitch").hidden=CAMS.length<2;
    camDlg.showModal();
  }catch(e){
    stopStream();
    say(e&&e.name==="NotAllowedError"?"The camera is blocked for this page. Allow it in the browser's address bar, or use Choose files."
       :"No camera could be opened on this device. Use Choose files instead.");
  }
}
$("#camshot").onclick=()=>{
  if(!camVideo.videoWidth) return;
  const cv=document.createElement("canvas"); cv.width=camVideo.videoWidth; cv.height=camVideo.videoHeight;
  cv.getContext("2d").drawImage(camVideo,0,0);
  cv.toBlob(b=>{ if(!b) return; const d=new Date(), z=n=>String(n).padStart(2,"0");
    const f=new File([b],"camera-"+d.getFullYear()+z(d.getMonth()+1)+z(d.getDate())+"-"+z(d.getHours())+z(d.getMinutes())+z(d.getSeconds())+".jpg",{type:"image/jpeg"});
    addFiles([f]); SHOTS++; $("#camcount").textContent=SHOTS+(SHOTS===1?" page":" pages")+" added"; },"image/jpeg",0.95);
};
$("#camswitch").onclick=async()=>{ if(CAMS.length<2) return; CAMI=(CAMI+1)%CAMS.length; try{ await startStream(CAMS[CAMI].deviceId); }catch(e){ say("That camera could not be opened."); } };
$("#camdone").onclick=()=>camDlg.close();
camDlg.addEventListener("close",stopStream);
fileInput.onchange=()=>{ addFiles([...fileInput.files]); fileInput.value=""; };
camInput.onchange=()=>{ addFiles([...camInput.files]); camInput.value=""; };

function newKey(){ return (crypto.randomUUID?crypto.randomUUID():String(Date.now())+Math.random().toString(16).slice(2)).replace(/[^A-Za-z0-9_-]/g,""); }
async function plainError(r){
  if(r.status>=500) return "Something went wrong on our side. Please try again in a moment.";
  try{ const j=await r.json(); if(typeof j.detail==="string"&&j.detail) return j.detail; }catch(e){}
  return "Something is wrong with what was sent. Please check the pages and try again.";
}
$("#go").onclick=async()=>{
  if(SENDING) return;                               // pressed twice while the files are still going up: nothing
  const token=$("#token").value.trim(), phone=phoneDigits($("#phone").value);
  if(!tokenOk(token)||!phoneOk(phone)){ say("Please enter the token number and a 10-digit mobile number first."); return; }
  if(!PAGES.length){ say("Please add at least one photo, PDF or scan."); return; }
  say(""); SENDING=true; render();
  const sent=PAGES.slice();
  const fd=new FormData();
  fd.append("token_no",token); fd.append("phone",phone);
  const dept=$("#dept").value; if(dept){ fd.append("department",dept); try{ localStorage.setItem("cdi_dept",dept); }catch(e){} }
  fd.append("grouping",sent.length>1&&document.querySelector('input[name=grp]:checked').value==="one_document"?"one_document":"separate");
  for(const p of sent) fd.append("files",p.file,p.file.name);
  SEND_KEY=SEND_KEY||newKey();                       // the same Send retried = the same job
  $("#hint").textContent="sending…";
  let r;
  try{ r=await fetch("api/jobs",{method:"POST",body:fd,headers:{"Idempotency-Key":SEND_KEY}}); }
  catch(e){ say("We could not reach the server. Please check the connection and try again."); $("#hint").textContent=""; SENDING=false; render(); return; }
  if(!r.ok){ say(await plainError(r)); $("#hint").textContent=""; if(r.status<500) SEND_KEY=null; SENDING=false; render(); return; }
  const id=(await r.json()).job_id;
  // accepted: it is read in the background. The desk is free straight away for the next patient: the token and
  // mobile number are cleared so the next paper is never filed under this one by mistake.
  TOKEN_ASKED.clear(); TOKEN_CLASH.clear();                       // a token just used is looked up afresh next time
  retireCurrent();                                                 // the next prescription is on its way: the one shown moves to Extracted
  JOBS=JOBS.filter(x=>!x.finished||x.err);                         // and its progress card goes with it (a stopped one stays: it needs a look)
  JOBS.unshift({id,token,phone,names:sent.map(p=>p.file.name),j:null,finished:false,err:""});
  for(const p of sent) if(p.url) URL.revokeObjectURL(p.url);
  PAGES=[]; SEND_KEY=null; SENDING=false; $("#hint").textContent="";
  $("#token").value=""; $("#phone").value=""; ACK=""; EXIST=null; EXIST_FOR=""; gate();
  $("#emr-card").hidden=false; renderJobs(); poll();
};

// ---- the result, shown under the patient (name + mobile number), every part collapsible --------------------
const GROUPS=new Map(), OPEN=new Set();               // GROUPS: "phone|name" -> {phone,name,docs:Map(document_id -> doc)}
let SEQ=0;
function gkey(phone,name){ return phone+"|"+(name||""); }
// a name read a little differently each time ("Onkar" / "Oukar") is the same patient on the same mobile number
function nameKey(n){ return String(n||"").replace(/^\s*(mr|mrs|ms|miss|master|baby|dr|smt|shri|sri|sh|late)\b\.?\s*/i,"").toLowerCase().replace(/[^a-z ]/g,"").replace(/\s+/g," ").trim(); }
function lev(a,b){ const m=a.length,n=b.length; let p=Array.from({length:n+1},(_,j)=>j); for(let i=1;i<=m;i++){ const c=[i]; for(let j=1;j<=n;j++) c[j]=Math.min(p[j]+1,c[j-1]+1,p[j-1]+(a[i-1]===b[j-1]?0:1)); p=c; } return p[n]; }
function sameName(a,b){ const x=nameKey(a),y=nameKey(b); if(!x||!y) return x===y; if(x===y) return true; return 1-lev(x,y)/Math.max(x.length,y.length)>=0.8; }
function addDoc(phone,name,doc){
  let g=[...GROUPS.values()].find(h=>h.phone===phone&&sameName(h.name,name));
  if(!g){ const k=gkey(phone,name); g={key:k,phone,name:name||null,docs:new Map()}; GROUPS.set(k,g); }
  const old=g.docs.get(doc.document_id)||{};
  g.docs.set(doc.document_id,{...old,...doc,seq:old.seq||++SEQ});
  return g;
}
async function showPatient(p){                         // chosen from the search: load that patient's prescriptions
  let list=[];
  try{ const r=await fetch("api/intake/prescriptions?phone="+encodeURIComponent(p.phone)+"&name="+encodeURIComponent(p.name||"")); if(r.ok) list=(await r.json()).prescriptions; }catch(e){}
  for(const d of list) addDoc(p.phone,p.name,d);
  const gk=([...GROUPS.values()].find(h=>h.phone===p.phone&&sameName(h.name,p.name))||{key:gkey(p.phone,p.name)}).key;
  OPEN.add("g:"+gk); renderGroups();
  const el=document.querySelector('details.grp[data-k="'+CSS.escape(gk)+'"]'); if(el) el.scrollIntoView({block:"nearest"});
}
async function loadResultsOf(job){                     // the finished job's JSON: put each document under its patient
  let j;
  try{ const r=await fetch("api/jobs/"+job.id+"/result.json"); if(!r.ok) throw 0; j=await r.json(); }
  catch(e){ job.err="The result could not be loaded. Please try again in a moment."; return; }
  for(const r of (j.results||[])){
    const it=r.intake||{}, name=it.patient_name||((r.patient||{}).name||{}).value||null, phone=it.phone||job.phone;
    retireCurrent();                                               // the one before moves to Extracted
    JOBS=JOBS.filter(x=>x===job||!x.finished||x.err);              // its progress card leaves this tab with it
    const g=addDoc(phone,name,{document_id:r.document_id,token_no:it.token_no||job.token,filename:r.filename,status:r.status,result:r,uploaded:new Date().toISOString(),current:true});
    job.groupKey=g.key;
  }
  renderGroups(); renderCurrent();
}
async function loadDoc(id){
  for(const g of GROUPS.values()){ const d=g.docs.get(id); if(d&&!d.result&&!d.loading){
    d.loading=true;
    try{ const r=await fetch("api/documents/"+id+"/result.json"); if(r.ok) d.result=await r.json(); else d.err="This result could not be loaded."; }catch(e){ d.err="This result could not be loaded."; }
    d.loading=false; renderGroups(); renderCurrent(); } }
}
function renderGroups(){
  const newest=g=>[...g.docs.values()].reduce((m,d)=>(d.uploaded||"")>m?(d.uploaded||""):m,"");
  const list=[...GROUPS.values()].sort((a,b)=>newest(b).localeCompare(newest(a)));
  const history=list.filter(g=>[...g.docs.values()].some(d=>!d.current));
  $("#groups").innerHTML=history.length?history.map(g=>{
    const docs=[...g.docs.values()].filter(d=>!d.current).sort((a,b)=>(b.uploaded||"").localeCompare(a.uploaded||"")||b.seq-a.seq);      // newest first
    return '<details class="cf-det grp" data-k="'+esc(g.key)+'"'+(OPEN.has("g:"+g.key)?" open":"")+'><summary><b>'+esc(g.name||"Name not read")+'</b> · '
      +esc(fmtPhone(g.phone))+' <span class="pill">'+plural(docs.length,"prescription")+'</span></summary><div class="det-body">'
      +docs.map(docHtml).join("")+'</div></details>';
  }).join(""):'<p class="muted" style="margin-top:12px">Nothing here yet. Earlier prescriptions appear here once the next one is uploaded, or find a patient above.</p>';
}
function retireCurrent(){                                       // the next prescription was uploaded: the earlier ones become history (Extracted)
  let n=0;
  for(const g of GROUPS.values()) for(const d of g.docs.values()) if(d.current){ d.current=false; n++; }
  if(n){ if($("#tab-ex").getAttribute("aria-selected")!=="true"){ NEWCOUNT+=n; badge(); } renderGroups(); renderCurrent(); }
}
function docBody(d){
  const r=d.result;
  const thumb=r&&r.document_id?'<div class="docthumb"><button type="button" class="img-open" data-doc="'+esc(r.document_id)+'" data-pages="'+(r.page_count||1)+'" aria-label="Open the prescription image"><img src="api/intake/page-image?document_id='+encodeURIComponent(r.document_id)+'&w=300" alt="The prescription (click to enlarge)" loading="lazy"/></button><span class="muted">Click the picture to see the prescription full size.</span></div>':"";
  return r?thumb+summaryHtml(r)+jsonBlock(r):(d.err?'<p class="jerr">'+esc(d.err)+'</p>':'<p class="muted">Loading…</p>');
}
function renderCurrent(){
  const cur=[];
  for(const g of GROUPS.values()) for(const d of g.docs.values()) if(d.current) cur.push({g,d});
  $("#current-card").hidden=!cur.length;
  $("#current-body").innerHTML=cur.map(({g,d})=>'<h3 style="margin:8px 0 2px">'+esc(g.name||"Name not read")+' · '+esc(fmtPhone(g.phone))+' · token '+esc(d.token_no||"—")
    +' <button type="button" class="vbtn img-open" data-doc="'+esc(d.document_id)+'" data-pages="'+(d.result?(d.result.page_count||1):1)+'" aria-label="View the prescription image full screen">View image</button></h3>'+docBody(d)).join("");
}
function docHtml(d){
  const r=d.result, st=r?(r.status==="complete"?'<span class="pill ok">all values accepted</span>':r.status==="needs_check"?'<span class="pill warn">needs a check</span>':'<span class="pill">'+esc(r.status||"")+'</span>'):"";
  const body=docBody(d);
  return '<details class="cf-det doc" data-d="'+esc(d.document_id)+'"'+(OPEN.has("d:"+d.document_id)?" open":"")+'><summary>Token <b>'+esc(d.token_no||"—")+'</b> · '
    +esc(fmtDate(d.uploaded))+' · '+esc(d.filename||"")+' '+st+' <button type="button" class="vbtn img-open" data-doc="'+esc(d.document_id)+'" data-pages="'+(r?(r.page_count||1):(d.pages||1))+'" aria-label="View the prescription image full screen">View image</button></summary><div class="det-body">'+body+'</div></details>';
}
function jsonBlock(r){
  const dl=r.document_id?'<a class="btn btn-ghost btn-sm" href="api/documents/'+esc(r.document_id)+'/result.json?download=true" download>Download JSON</a>':"";
  return '<details class="cf-det sec"><summary>Result (JSON)</summary><div class="det-body">'+dl+'<pre class="resjson" tabindex="0" aria-label="Result JSON">'+esc(JSON.stringify(r,null,2))+'</pre></div></details>';
}
// ---- the prescription image viewer: any page, zoom, the page as read or the original photo
const IMG={doc:null,pages:1,page:1,view:"page",zoom:100};
let IMG_URL=null;
async function imgShow(){
  const el=$("#img-el");
  el.style.width=IMG.zoom+"%";
  const url="api/intake/page-image?document_id="+encodeURIComponent(IMG.doc)+"&page="+IMG.page+"&view="+IMG.view;
  try{
    const r=await fetch(url); if(!r.ok) throw 0;
    IMG.pages=Math.max(1,+r.headers.get("X-Page-Count")||IMG.pages);          // the server knows how many pages there are
    const blob=await r.blob(); if(IMG_URL) URL.revokeObjectURL(IMG_URL); IMG_URL=URL.createObjectURL(blob); el.src=IMG_URL;
  }catch(e){ el.removeAttribute("src"); el.alt="The picture could not be loaded."; }
  $("#img-pg").textContent=IMG.pages>1?("Page "+IMG.page+" of "+IMG.pages):"";
  $("#img-prev").hidden=$("#img-next").hidden=IMG.pages<2;
  $("#img-view").textContent=IMG.view==="page"?"Original photo":"Page as read";
  $("#img-title").textContent=IMG.view==="page"?"Prescription (the page as read)":"Prescription (the photo as uploaded)";
}
function openImage(doc,pages){ IMG.doc=doc; IMG.pages=Math.max(1,+pages||1); IMG.page=1; IMG.zoom=100; IMG.view="page"; imgShow(); if(!$("#imgdlg").open) $("#imgdlg").showModal(); }
$("#img-close").onclick=()=>$("#imgdlg").close();
$("#img-in").onclick=()=>{ IMG.zoom=Math.min(400,IMG.zoom+25); $("#img-el").style.width=IMG.zoom+"%"; };
$("#img-out").onclick=()=>{ IMG.zoom=Math.max(50,IMG.zoom-25); $("#img-el").style.width=IMG.zoom+"%"; };
$("#img-fit").onclick=()=>{ IMG.zoom=100; $("#img-el").style.width="100%"; };
$("#img-prev").onclick=()=>{ if(IMG.page>1){ IMG.page--; imgShow(); } };
$("#img-next").onclick=()=>{ if(IMG.page<IMG.pages){ IMG.page++; imgShow(); } };
$("#img-view").onclick=()=>{ IMG.view=IMG.view==="page"?"original":"page"; imgShow(); };
$("#imgdlg").addEventListener("click",e=>{ if(e.target===$("#imgdlg")) $("#imgdlg").close(); });
document.addEventListener("click",e=>{ const b=e.target.closest(".img-open"); if(b){ e.preventDefault(); openImage(b.dataset.doc,b.dataset.pages); } });
document.addEventListener("click",async e=>{                  // the patient's name: pick another reading, or confirm / correct it (either tab)
  const cand=e.target.closest(".nm-cand"); if(cand){ cand.closest(".nmbox").querySelector(".nm-in").value=cand.dataset.name; return; }
  const ok=e.target.closest(".nm-ok"); if(!ok) return;
  const box=ok.closest(".nmbox"), id=box.dataset.doc, err=box.querySelector(".nm-err");
  err.textContent=""; ok.disabled=true;
  let r; try{ r=await fetch("api/intake/name",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({document_id:id,name:box.querySelector(".nm-in").value})}); }
  catch(x){ err.textContent="We could not reach the server."; ok.disabled=false; return; }
  if(!r.ok){ err.textContent=await plainError(r); ok.disabled=false; return; }
  applyName(id,await r.json());
});
function applyName(id,j){                                    // the confirmed name decides the patient group; the result is loaded again
  for(const g of [...GROUPS.values()]){
    const d=g.docs.get(id); if(!d) continue;
    g.docs.delete(id); if(!g.docs.size) GROUPS.delete(g.key);
    d.result=null; d.loading=false;
    const ng=addDoc(g.phone,j.patient_name,d); OPEN.add("g:"+ng.key); OPEN.add("d:"+id);
    break;
  }
  renderGroups(); renderCurrent(); loadDoc(id);
}
$("#groups").addEventListener("toggle",e=>{                  // keep what is open open when the list is drawn again
  const el=e.target; if(!(el instanceof HTMLDetailsElement)) return;
  const key=el.classList.contains("grp")?"g:"+el.dataset.k:el.classList.contains("doc")?"d:"+el.dataset.d:null;
  if(!key) return;
  if(el.open){ OPEN.add(key); if(el.classList.contains("doc")) loadDoc(el.dataset.d); } else OPEN.delete(key);
},true);

// ---- the tables shown for one prescription: patient, organisation, doctor, doctor booking, lab tests, visits ----
// Only what the result JSON says: a value that is not on the page is not shown, never filled in.
function vget(v){ return (v&&typeof v==="object"&&"value" in v)?v:{value:null,status:"absent",reason:null}; }
function vcell(v){
  v=vget(v); const none=v.value===null||v.value===undefined||v.value==="";
  if(none) return "";
  let t=none?'<span class="none">not on the page</span>':esc(typeof v.value==="boolean"?(v.value?"yes":"no"):v.value);
  if(!none&&v.status==="needs_check") t+=' <span class="pill warn" title="'+esc(v.reason||"")+'">needs a check</span>';
  return t;
}
function section(id,title,inner){ return '<details class="cf-det sec sumtbl" data-tbl="'+id+'" open><summary>'+title+'</summary><div class="det-body">'+inner+'</div></details>'; }
function rowsTable(id,title,rows,note){
  // a value that is not on the page is not shown at all (the cell is "")
  const shown=rows.filter(r=>r[1]!=="");
  return section(id,title,'<div class="tbox"><table><tbody>'
    +(shown.length?shown.map(r=>'<tr><th scope="row">'+esc(r[0])+'</th><td>'+r[1]+'</td></tr>').join("")
      :'<tr><td class="none">Nothing readable on the page.</td></tr>')
    +'</tbody></table></div>'+(shown.length&&note?'<p class="note">'+note+'</p>':''));
}
function bookingOf(r){
  const f=r.follow_up||{}, text=f.text||f.value||"";
  const doc=vget((r.doctor||{}).name).value;
  const tests=(r.lab_tests||[]).filter(t=>t.status!=="rejected").map(t=>t.as_written||t.text).filter(Boolean);
  let needed, when="", cls="ok";
  if(!text){ needed="No follow-up is written on the page"; cls=""; }
  else if(f.kind==="as_needed"){ needed="Only if needed"; when="as needed"; cls=""; }
  else{
    needed="Yes"; cls="warn";
    if(f.kind==="interval"&&f.interval_value!=null){
      const a=f.interval_value, b=f.interval_value_max, u=f.interval_unit||"";
      when="after "+(b!=null&&b!==a?a+"–"+b+" "+u+"s":plural(a,u));
    } else if(f.kind==="date"&&f.date){ when="on "+f.date; }
    else when="time not clear: please read the note";
  }
  const bring=/report|result|test|investigation/i.test(text);
  return {needed,cls,when,text,doc,tests,bring,status:f.status};
}
// A handwritten name is never final on its own: it is "to confirm" until a person confirms or corrects it here.
function nameCell(r){
  const P=r.patient||{}, I=r.intake||{}, v=vget(P.name), id=r.document_id||"", shown=v.value||"", confirmed=!!I.name_confirmed;
  const pill=confirmed?'<span class="pill ok">confirmed'+(I.name_confirmed_by?' by '+esc(I.name_confirmed_by):'')+'</span>'
    :shown?'<span class="pill warn" title="'+esc(v.reason||"")+'">to confirm</span>':'<span class="pill warn">name not read: please type it</span>';
  const chips=(I.name_candidates||[]).map(c=>'<button type="button" class="btn btn-ghost btn-sm nm-cand" data-name="'+esc(c)+'">'+esc(c)+'</button>').join("");
  const crop='<div class="nmcrop"><div class="muted">As written on the paper (click to enlarge the page):</div><button type="button" class="img-open" data-doc="'+esc(id)+'" data-pages="'+(r.page_count||1)+'" aria-label="Open the prescription image">'
    +'<img src="api/intake/name-crop?document_id='+encodeURIComponent(id)+'" alt="The patient name as written on the paper" loading="lazy" onerror="this.closest(\'.nmcrop\').hidden=true"/></button></div>';
  return '<div class="nmbox" data-doc="'+esc(id)+'">'+crop+(shown?'<b>'+esc(shown)+'</b> ':'')+pill
    +'<div class="nmedit"><input type="text" class="nm-in" maxlength="80" value="'+esc(shown)+'" aria-label="Patient name as on the paper" placeholder="the name as on the paper"/> '
    +'<button type="button" class="btn btn-primary btn-sm nm-ok">'+(confirmed?'Change':'Confirm')+'</button></div>'
    +(chips&&!confirmed?'<div class="nmchips muted">Other readings of the name: '+chips+'</div>':'')
    +(I.name_read&&I.name_read!==shown?'<div class="muted">As read from the page: '+esc(I.name_read)+'</div>':'')
    +'<div class="fielderr nm-err" role="alert"></div></div>';
}
function summaryHtml(r){
  r=r||{}; const P=r.patient||{}, D=r.doctor||{}, O=r.organization||D.clinic||{}, V=r.visits||[];
  const patient=rowsTable("tbl-patient","Patient details",[
    ["Name",nameCell(r)],["Age",vcell(P.age_text)],["Date of birth",vcell(P.dob)],["Sex",vcell(P.sex)],
    ["Patient ID (MRN)",vcell(P.mrn)],["Phone",vcell(P.phone)],["Address",vcell(P.address)],
    ["Token number",r.intake&&r.intake.token_no?esc(r.intake.token_no):""],["Mobile number (typed at upload)",r.intake&&r.intake.phone?esc(fmtPhone(r.intake.phone)):""]]);
  const org=rowsTable("tbl-organization","Organisation (hospital / clinic)",[
    ["Name",vcell(O.name)],["Address",vcell(O.address)],["Phone",vcell(O.phone)]],
    "Read from the letterhead or stamp. A government hospital may print only its name and no doctor.");
  const doctor=rowsTable("tbl-doctor","Doctor details",[
    ["Name",vcell(D.name)],["Registration no.",vcell(D.reg_no)],["Department",vcell(D.department)],
    ["Designation",vcell(D.designation)],["Qualification",vcell(D.qualification)],
    ["Stamp on the page",vcell(D.stamp_present)],["Signature on the page",vcell(D.signature_present)]],
    "Stamp and signature are a visual guess by the model and are not verified.");   // shown only when a stamp / signature row is
  const b=bookingOf(r), lat=V.find(v=>v.is_latest);
  const booking=rowsTable("tbl-booking","Doctor booking",[
    ["Booking needed",'<span class="pill '+b.cls+'">'+esc(b.needed)+'</span>'+(b.status==="needs_check"&&b.text?' <span class="pill warn">needs a check</span>':"")],
    ["With",b.doc?esc(b.doc):'<span class="none">doctor not read</span>'],
    ["When",b.when?esc(b.when):'<span class="none">—</span>'],
    ["As written",b.text?esc(b.text):'<span class="none">—</span>'],
    ["Next date written",(r.follow_up||{}).next_date?esc((r.follow_up||{}).next_date)+((r.follow_up||{}).next_date_iso?' <span class="none">('+esc((r.follow_up||{}).next_date_iso)+')</span>':""):""],
    ["From the visit",lat&&V.length>1?esc(lat.where+(lat.date_text?" · dated "+lat.date_text:"")+" (the latest of "+V.length+" dated visits)"):""],
    ["Bring to the visit",b.bring&&b.tests.length?esc("Results of: "+b.tests.join(", ")):b.bring?"Reports (as written)":'<span class="none">nothing written</span>']],
    b.needed==="Yes"?"The date is not guessed: it is counted from the day of the visit written on the prescription.":"");
  const allLabs=r.lab_tests||[];
  const isUnrec=t=>/^not a recognised test name/.test(t.reason||"");
  const isUnconf=t=>/^not confirmed on the page/.test(t.reason||"");
  const labs=allLabs.filter(t=>t.status!=="rejected"&&!isUnrec(t));            // a test the lists know is ALWAYS listed, never put in a footnote
  const unrec=allLabs.filter(t=>t.status!=="rejected"&&isUnrec(t));
  const unconf=[];
  const dropped=allLabs.filter(t=>t.status==="rejected");
  const prep=r.lab_preparation||[];
  const labRows=labs.map((t,i)=>{
    const ctx=(t.context||[]).map(c=>esc(c.text)+' <span class="none">('+esc(c.kind)+')</span>').join("<br>");
    const std=t.standard_name?esc(t.standard_name)+' <span class="none">('+esc(t.standard_source||"")+')</span>'
      +(t.code?'<br><span class="none">'+esc(t.code_system||"")+" "+esc(t.code)+'</span>':"")
      :t.code?esc(t.code_display||t.code)+' <span class="none">'+esc(t.code_system||"")+" "+esc(t.code)+'</span>':'<span class="none">not matched to a standard test</span>';
    const pr=(t.preparation||[]).length?t.preparation.map(esc).join("<br>"):'<span class="none">none written</span>';
    const st=t.status==="accepted"?'<span class="pill ok">read</span>':t.status==="rejected"?'<span class="pill err">rejected</span>'
      :isUnconf(t)?'<span class="pill warn" title="'+esc(t.reason||"")+'">needs a check: not confirmed by the text reader, look at the image</span>'
      :'<span class="pill warn" title="'+esc(t.reason||"")+'">needs a check</span>';
    const match=(t.gate_recognised?'<div class="none">lab list: recognised</div>':'<div class="none">lab list: not placed</div>')
      +(t.page_support!=null?'<div class="none">page match '+Math.round(t.page_support*100)+'%</div>':'');
    return '<tr><td>'+(i+1)+'</td><td>'+esc(t.as_written||t.text||"")+'</td><td>'+std+match+'</td><td>'+pr+'</td><td>'+(ctx||'<span class="none">—</span>')+'</td><td>'+st+'</td></tr>';
  }).join("");
  const droppedNote=dropped.length?'<p class="note">Left out, not shown as tests: '+dropped.map(t=>esc(t.as_written||t.text||"")+(t.reason?" ("+esc(t.reason)+")":"")).join("; ")+'.</p>':"";
  const unrecNote=unrec.length?'<p class="note">Read from the page but not recognised as a test name, so not listed above (please check the page): '+unrec.map(t=>esc(t.as_written||t.text||"")).join("; ")+'.</p>':"";
  const unconfNote=unconf.length?'<p class="note">The reader suggested these, but nothing on the page supports them, so they are not listed as tests (check the page): '+unconf.map(t=>esc(t.as_written||t.text||"")).join("; ")+'.</p>':"";
  const prepOnly=!labs.length&&prep.length?'<p class="note">Preparation written on the page: '+prep.map(p=>esc(p.text)).join("; ")+'</p>':"";
  const lab=section("tbl-labs",'Lab tests <span class="pill">'+labs.length+'</span>',
    (labs.length?'<div class="tbox"><table><thead><tr><th>#</th><th>Test (as written)</th><th>Standard name</th><th>Preparation</th><th>Why (linked to)</th><th>Check</th></tr></thead><tbody>'+labRows+'</tbody></table></div>'
      :'<p class="none" style="margin:4px 0">No lab test is written on this page.</p>')+prepOnly+unrecNote+unconfNote+droppedNote);
  const visRows=V.map(v=>'<tr><td>'+esc(v.where)+'</td><td>'+(v.date_text?esc(v.date_text):'<span class="none">no date read</span>')+'</td><td>'
    +(v.lab_tests.length?esc(v.lab_tests.join(", ")):'<span class="none">none</span>')+'</td><td>'+(v.follow_up?esc(v.follow_up):'<span class="none">—</span>')
    +'</td><td>'+(v.is_latest?'<span class="pill ok">latest — used above</span>':'<span class="pill">earlier</span>')+'</td></tr>').join("");
  const visits=V.length>1||(V.length===1&&V[0].date)?section("tbl-visits",'Dated visits found on the pages <span class="pill">'+V.length+'</span>',
    '<div class="tbox"><table><thead><tr><th>Where</th><th>Date (as written)</th><th>Lab tests</th><th>Follow-up</th><th></th></tr></thead><tbody>'+visRows+'</tbody></table></div>'
    +'<p class="note">The lab tests and the doctor booking above are those of the latest dated visit. An entry with no readable date is never ranked above one with a date.</p>'):"";
  return patient+org+doctor+booking+lab+visits;
}

// ---- progress, per patient (not per batch) --------------------------------------------------------------
function poll(){
  clearTimeout(TIMER);
  const active=JOBS.filter(x=>!x.finished);
  if(!active.length){ if(!SENDING) $("#hint").textContent=""; return; }
  if(!SENDING) $("#hint").textContent=active.length+(active.length===1?" prescription":" prescriptions")+" being read…";
  TIMER=setTimeout(async()=>{
    await Promise.all(active.map(refreshJob));
    renderJobs(); poll();
  }, 1500);
}
async function refreshJob(job){
  let j;
  try{ const r=await fetch("api/jobs/"+job.id); if(!r.ok) return; j=await r.json(); }catch(e){ return; }   // a blip: try again next time
  job.j=j;
  if(j.state==="done"||j.state==="review"){ job.finished=true; await loadResultsOf(job); }
  else if(j.state==="error"||j.state==="mismatch"){
    job.finished=true;
    job.err=j.error||j.documents.filter(d=>d.error).map(d=>d.filename+": "+d.error.replace(/^rescan: /,"")).join(" ")||"Processing stopped. Please try again.";
  }
}
function jobGrid(j){
  let h='<tr><th class="doc">Document</th>'+STAGES.map(s=>'<th>'+STAGE_LABEL[s]+'</th>').join("")+'<th>Done</th></tr>';
  h+=j.documents.map(d=>{
    const cells=STAGES.map(s=>{
      const st=d.stages[s]||"pending";
      const inner=st==="done"?TICK:st==="running"?'<span class="spin"></span>':st==="error"?'✕':'·';
      return '<td><span class="cell '+st+'">'+inner+'</span></td>';
    }).join("");
    const done=d.status==="done";
    return '<tr><td class="doc"><div class="fn">'+esc(d.filename)+'</div><div class="sub">'
      +(d.doc_type?esc(d.doc_type):'')+(d.facts?' · '+d.facts+' facts':'')+(d.seconds?' · '+d.seconds+'s':'')+'</div>'
      +(d.error?'<div class="sub" style="color:var(--danger,#b42318)">'+esc(d.error.replace(/^rescan: /,""))+'</div>':'')+'</td>'
      +cells+'<td>'+(done?'<span class="cell done">'+TICK+'</span>':d.status==="error"?'<span class="cell error">✕</span>':'<span class="cell running"><span class="spin"></span></span>')+'</td></tr>';
  }).join("");
  return h;
}
const JOB_CLOSED=new Set();
function renderJobs(){
  $("#jobs").innerHTML=JOBS.map(job=>{
    const j=job.j, running=!job.finished;
    const who=(j&&j.patient_name)||null;
    const state=job.err?'<span class="pill warn">stopped</span>':running?(j&&j.state==="running"?'<span class="pill">reading…</span>':'<span class="pill">'+(j&&j.queue_position!=null?(j.queue_position===0?'waiting · next':'waiting · '+j.queue_position+' ahead'):'waiting…')+'</span>'):'<span class="pill ok">done</span>';
    return '<details class="cf-det jobbox" data-j="'+esc(job.id)+'"'+(JOB_CLOSED.has(job.id)?"":" open")+'><summary><b>'+esc(who||"Reading the name…")+'</b> · '+esc(fmtPhone(job.phone))
      +' · token '+esc(job.token)+' '+state+'</summary><div class="det-body">'
      +(job.err?'<div class="jerr" role="alert">'+esc(job.err)+'</div>':'')
      +(j?'<div style="overflow-x:auto"><table class="emr-grid">'+jobGrid(j)+'</table></div>':'<div class="muted">sent — '+(j&&j.queue_position!=null?(j.queue_position===0?'it is next to be read':j.queue_position+' prescription'+(j.queue_position===1?'':'s')+' ahead of it (5 are read at a time)'):'waiting to start')+'…</div>')
      +(j?j.documents.filter(d=>d.document_id).map(d=>'<div style="margin-top:8px"><button type="button" class="btn btn-ghost btn-sm img-open" data-doc="'+esc(d.document_id)+'" data-pages="1" aria-label="View the uploaded image full screen">View image: '+esc(d.filename)+'</button></div>').join(""):"")
      +(job.finished&&!job.err?'<div style="margin-top:8px"><button type="button" class="btn btn-primary btn-sm job-go" data-job="'+esc(job.id)+'">Read: open in Extracted ›</button></div>':'')
      +'<div class="muted" style="margin-top:6px">'+esc(job.names.join(", "))+'</div></div></details>';
  }).join("");
}
$("#jobs").addEventListener("click",e=>{
  const b=e.target.closest(".job-go"); if(!b) return;
  const job=JOBS.find(j=>j.id===b.dataset.job); showTab("ex");
  if(job&&job.groupKey){ OPEN.add("g:"+job.groupKey); renderGroups(); const el=document.querySelector('details.grp[data-k="'+CSS.escape(job.groupKey)+'"]'); if(el) el.scrollIntoView({block:"nearest"}); }
});
$("#jobs").addEventListener("toggle",e=>{ const el=e.target; if(el instanceof HTMLDetailsElement&&el.dataset.j){ if(el.open) JOB_CLOSED.delete(el.dataset.j); else JOB_CLOSED.add(el.dataset.j); } },true);

// ---- the lab test mapping table -------------------------------------------------------------------------
let MAPROWS=null;
async function loadMap(){
  try{ const r=await fetch("api/mappings/lab"); if(!r.ok) throw 0; MAPROWS=(await r.json()).rows; }
  catch(e){ $("#maptbl").innerHTML='<tr><td class="none">The mapping table could not be loaded.</td></tr>'; return; }
  drawMap();
}
function drawMap(){
  if(!MAPROWS) return;
  const f=$("#mapfilter").value.trim().toLowerCase();
  const rows=MAPROWS.filter(r=>!f||r.alias.toLowerCase().includes(f)||r.canonical.toLowerCase().includes(f));
  const canon=new Set(MAPROWS.filter(r=>r.enabled).map(r=>r.canonical));
  $("#map-count").textContent="· "+MAPROWS.filter(r=>r.enabled).length+" written names → "+canon.size+" standard tests";
  $("#maptbl").innerHTML='<thead><tr><th>Written as</th><th></th><th>Standard test</th><th>Code</th><th>Source</th><th>Note</th><th>On</th></tr></thead><tbody>'
    +(rows.length?rows.map(r=>'<tr class="'+(r.enabled?"":"off")+'"><td>'+esc(r.alias)+'</td><td class="none">→</td><td><b>'+esc(r.canonical)+'</b></td><td>'+esc(r.loinc||"")+'</td><td>'+esc(r.source)+'</td><td>'+esc(r.note||"")
      +'</td><td><input type="checkbox" data-k="'+esc(r.alias_key)+'" aria-label="Use '+esc(r.alias)+'"'+(r.enabled?" checked":"")+'/></td></tr>').join("")
      :'<tr><td colspan="7" class="none">No row matches.</td></tr>')+'</tbody>';
}
$("#map-det").addEventListener("toggle",()=>{ if($("#map-det").open&&!MAPROWS) loadMap(); });
$("#mapfilter").addEventListener("input",drawMap);
$("#maptbl").addEventListener("change",async e=>{
  const c=e.target.closest("input[type=checkbox][data-k]"); if(!c) return;
  try{ const r=await fetch("api/mappings/lab/"+encodeURIComponent(c.dataset.k)+"/enabled",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enabled:c.checked})}); if(!r.ok) throw 0; }
  catch(err){ c.checked=!c.checked; $("#m-err").textContent="That change could not be saved."; return; }
  $("#m-err").textContent=""; loadMap();
});
$("#m-add").onclick=async()=>{
  const body={alias:$("#m-alias").value,canonical:$("#m-canon").value,loinc:$("#m-loinc").value,note:$("#m-note").value};
  let r; try{ r=await fetch("api/mappings/lab",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); }catch(e){ $("#m-err").textContent="We could not reach the server."; return; }
  if(!r.ok){ $("#m-err").textContent=await plainError(r); return; }
  $("#m-err").textContent=""; for(const id of ["#m-alias","#m-canon","#m-loinc","#m-note"]) $(id).value=""; loadMap();
};
gate(); renderGroups();
showTab(location.hash==="#extracted"?"ex":"up");
</script>
</body>
</html>
"""

ADMIN_PAGE = _HTML.replace("%%CSS%%", BRAND_CSS + _UPLOAD_CSS).replace("%%HEADER%%", _HEADER).replace("%%ICON%%", UPLOAD_ICON)

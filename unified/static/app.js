// ============================================================
// GenAI Studio - unified frontend
// ============================================================

const $ = (id) => document.getElementById(id);

// ---------- navigation ----------
function showPanel(name) {
    document.querySelectorAll(".panel-view").forEach((section) => {
        section.classList.toggle("active", section.id === "panel-" + name);
    });
    document.querySelectorAll(".nav-item").forEach((item) => {
        item.classList.toggle("active", item.dataset.panel === name);
    });
    if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => showPanel(item.dataset.panel));
});
document.querySelectorAll(".feature-card").forEach((card) => {
    card.addEventListener("click", () => showPanel(card.dataset.goto));
});

// ---------- helpers ----------
function toast(message, ok = false) {
    const el = $("toast");
    el.textContent = message;
    el.classList.toggle("ok", ok);
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), ok ? 4000 : 12000);
}

async function postJSON(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({ success: false, error: "Invalid server response" }));
    if (!data.success) throw new Error(data.error || "Request failed");
    return data;
}

async function withBusy(button, busyLabel, work) {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = busyLabel;
    try {
        return await work();
    } finally {
        button.disabled = false;
        button.textContent = original;
    }
}

function showImage(imgEl, phEl, dataUri) {
    imgEl.src = dataUri;
    imgEl.classList.add("on");
    if (phEl) phEl.style.display = "none";
}

// ---------- segmented controls ----------
function wireSegment(containerId, onChange) {
    const container = $(containerId);
    container.querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => {
            container.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            onChange(parseInt(btn.dataset.count, 10));
        });
    });
}

function wireSlider(rangeId, outId, formatter = (v) => v) {
    const range = $(rangeId);
    const out = $(outId);
    const sync = () => { out.textContent = formatter(range.value); };
    range.addEventListener("input", sync);
    sync();
}

// ---------- status ----------
function applyAvailability(name, info) {
    const note = $(name + "Avail");
    const button = $(name + "GenerateBtn");
    if (!note || !info) return;
    if (info.runnable === false) {
        note.textContent = "Unavailable: " + (info.note || "not runnable on this machine") + ".";
        note.classList.add("show");
        if (button) button.disabled = true;
    } else {
        note.classList.remove("show");
        note.textContent = "";
        if (button) button.disabled = false;
    }
}

async function refreshStatus() {
    try {
        const response = await fetch("/api/status");
        const data = await response.json();
        const status = data.status || {};
        const ready = Object.values(status).filter((item) => item && item.ready).length;
        $("serverDot").className = "dot on";
        $("serverLabel").textContent = `server online · ${ready} experiment(s) ready`;
        ["sdxl", "music", "video"].forEach((name) => applyAvailability(name, status[name]));
    } catch (error) {
        $("serverDot").className = "dot off";
        $("serverLabel").textContent = "server offline";
    }
}

// ============================================================
// EXPERIMENT 5 - VAE
// ============================================================
let vaeCount = 9;

wireSegment("vaeCountSeg", (count) => { vaeCount = count; });

// file upload
const vaeFile = $("vaeFile");
vaeFile.addEventListener("change", () => {
    if (vaeFile.files.length) previewInput(vaeFile.files[0]);
});

const dropZone = $("dropZone");
["dragover", "dragenter"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("hover"); }));
["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, () => dropZone.classList.remove("hover")));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) {
        vaeFile.files = e.dataTransfer.files;
        previewInput(e.dataTransfer.files[0]);
    }
});

function previewInput(file) {
    const reader = new FileReader();
    reader.onload = (event) => showImage($("vaeInputImage"), $("vaeInputPh"), event.target.result);
    reader.readAsDataURL(file);
}

$("vaeReconstructBtn").addEventListener("click", function () {
    if (!vaeFile.files.length) { toast("Please choose an MNIST digit image first."); return; }
    const button = this;
    withBusy(button, "Encoding…", async () => {
        const form = new FormData();
        form.append("image", vaeFile.files[0]);
        try {
            const response = await fetch("/api/vae/reconstruct", { method: "POST", body: form });
            const data = await response.json();
            if (!data.success) throw new Error(data.error);
            showImage($("vaeInputImage"), $("vaeInputPh"), "data:image/png;base64," + data.input);
            showImage($("vaeReconImage"), $("vaeReconPh"), "data:image/png;base64," + data.reconstruction);
            showImage($("vaeLatentGraph"), $("vaeLatentPh"), "data:image/png;base64," + data.latent_graph);
            $("vaeZ1").textContent = data.z1;
            $("vaeZ2").textContent = data.z2;
            refreshStatus();
        } catch (error) {
            toast("VAE reconstruction failed: " + error.message);
        }
    });
});

$("vaeGenerateBtn").addEventListener("click", function () {
    const button = this;
    withBusy(button, "Generating…", async () => {
        try {
            const data = await postJSON("/api/vae/generate", { count: vaeCount });
            const gallery = $("vaeGallery");
            gallery.innerHTML = "";
            data.images.forEach((item) => {
                const card = document.createElement("div");
                card.className = "item";
                card.innerHTML = `<img src="data:image/png;base64,${item.image}" alt="digit">
                                  <small>z₁ ${item.z1} · z₂ ${item.z2}</small>`;
                gallery.appendChild(card);
            });
            refreshStatus();
        } catch (error) {
            toast("VAE generation failed: " + error.message);
        }
    });
});

// decode point
wireSlider("vaeRangeZ1", "vaeOutZ1", (v) => Number(v).toFixed(2));
wireSlider("vaeRangeZ2", "vaeOutZ2", (v) => Number(v).toFixed(2));

$("vaeDecodeBtn").addEventListener("click", function () {
    const button = this;
    withBusy(button, "Decoding…", async () => {
        try {
            const data = await postJSON("/api/vae/decode", {
                z1: $("vaeRangeZ1").value,
                z2: $("vaeRangeZ2").value,
            });
            showImage($("vaeDecodedImage"), $("vaeDecodedPh"), "data:image/png;base64," + data.image);
        } catch (error) {
            toast("VAE decode failed: " + error.message);
        }
    });
});

// ============================================================
// EXPERIMENT 6 - GAN
// ============================================================
let ganCount = 4;
wireSegment("ganCountSeg", (count) => { ganCount = count; });

$("ganGenerateBtn").addEventListener("click", function () {
    const button = this;
    withBusy(button, "Generating…", async () => {
        try {
            const data = await postJSON("/api/gan/generate", { count: ganCount });
            const gallery = $("ganGallery");
            gallery.innerHTML = "";
            data.images.forEach((image) => {
                const card = document.createElement("div");
                card.className = "item";
                card.innerHTML = `<img src="data:image/png;base64,${image}" alt="face">`;
                gallery.appendChild(card);
            });
            refreshStatus();
        } catch (error) {
            toast("GAN generation failed: " + error.message);
        }
    });
});

// ============================================================
// EXPERIMENT 3 - FLAN-T5 QA
// ============================================================
wireSlider("qaTokens", "qaOutTokens");
document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => { $("qaQuestion").value = chip.textContent; });
});

$("qaAskBtn").addEventListener("click", function () {
    const button = this;
    withBusy(button, "Thinking…", async () => {
        try {
            const data = await postJSON("/api/qa/ask", {
                question: $("qaQuestion").value,
                max_tokens: parseInt($("qaTokens").value, 10),
            });
            $("qaAnswer").textContent = data.answer;
            refreshStatus();
        } catch (error) {
            toast("Question answering failed: " + error.message);
        }
    });
});

// ============================================================
// EXPERIMENT 4 - SDXL
// ============================================================
wireSlider("sdxlSteps", "sdxlOutSteps");
wireSlider("sdxlGuidance", "sdxlOutGuidance", (v) => Number(v).toFixed(1));

$("sdxlGenerateBtn").addEventListener("click", function () {
    const button = this;
    const hint = $("sdxlHint");
    withBusy(button, "Generating…", async () => {
        hint.textContent = "working · first run downloads the model";
        try {
            const data = await postJSON("/api/sdxl/generate", {
                prompt: $("sdxlPrompt").value,
                negative_prompt: $("sdxlNegative").value,
                steps: parseInt($("sdxlSteps").value, 10),
                guidance_scale: parseFloat($("sdxlGuidance").value),
                seed: parseInt($("sdxlSeed").value, 10),
            });
            showImage($("sdxlImage"), $("sdxlPh"), "data:image/png;base64," + data.image);
            hint.textContent = "done";
            refreshStatus();
        } catch (error) {
            hint.textContent = "failed";
            toast("Image generation failed: " + error.message);
        }
    });
});

// ============================================================
// EXPERIMENT 10 - VIDEO
// ============================================================
wireSlider("videoFrames", "videoOutFrames");
wireSlider("videoSteps", "videoOutSteps");

$("videoGenerateBtn").addEventListener("click", function () {
    const button = this;
    const hint = $("videoHint");
    withBusy(button, "Generating…", async () => {
        hint.textContent = "working · this can take a long time";
        try {
            const data = await postJSON("/api/video/generate", {
                prompt: $("videoPrompt").value,
                frames: parseInt($("videoFrames").value, 10),
                steps: parseInt($("videoSteps").value, 10),
            });
            const video = $("videoResult");
            video.src = data.video;
            video.classList.add("on");
            $("videoPh").style.display = "none";
            hint.textContent = "done";
            refreshStatus();
        } catch (error) {
            hint.textContent = "failed";
            toast("Video generation failed: " + error.message);
        }
    });
});

// ============================================================
// EXPERIMENT 7 - STYLEGAN (untrained demo)
// ============================================================
let styleganCount = 8;
wireSegment("styleganCountSeg", (count) => { styleganCount = count; });

$("styleganGenerateBtn").addEventListener("click", function () {
    const button = this;
    withBusy(button, "Generating…", async () => {
        try {
            const data = await postJSON("/api/stylegan/generate", { count: styleganCount });
            const gallery = $("styleganGallery");
            gallery.innerHTML = "";
            data.images.forEach((image) => {
                const card = document.createElement("div");
                card.className = "item";
                card.innerHTML = `<img src="data:image/png;base64,${image}" alt="stylegan sample">`;
                gallery.appendChild(card);
            });
            refreshStatus();
        } catch (error) {
            toast("StyleGAN generation failed: " + error.message);
        }
    });
});

// ============================================================
// EXPERIMENT 8 - MUSIC
// ============================================================
wireSlider("musicDuration", "musicOutDuration");

$("musicGenerateBtn").addEventListener("click", function () {
    const button = this;
    const hint = $("musicHint");
    withBusy(button, "Composing…", async () => {
        hint.textContent = "working · first run downloads the model";
        try {
            const data = await postJSON("/api/music/generate", {
                prompt: $("musicPrompt").value,
                duration: parseInt($("musicDuration").value, 10),
            });
            const audio = $("musicResult");
            audio.src = data.audio;
            audio.classList.add("on");
            $("musicPh").style.display = "none";
            hint.textContent = "done";
            refreshStatus();
        } catch (error) {
            hint.textContent = "failed";
            toast("Music generation failed: " + error.message);
        }
    });
});

// ============================================================
// EXPERIMENT 9 - RAG ASSISTANT
// ============================================================
let assistantDocument = "";
wireSlider("assistantTokens", "assistantOutTokens");

const assistantFile = $("assistantFile");
assistantFile.addEventListener("change", () => {
    if (assistantFile.files.length) uploadDocument(assistantFile.files[0]);
});

const assistantDrop = $("assistantDropZone");
["dragover", "dragenter"].forEach((evt) =>
    assistantDrop.addEventListener(evt, (e) => { e.preventDefault(); assistantDrop.classList.add("hover"); }));
["dragleave", "drop"].forEach((evt) =>
    assistantDrop.addEventListener(evt, () => assistantDrop.classList.remove("hover")));
assistantDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) {
        assistantFile.files = e.dataTransfer.files;
        uploadDocument(e.dataTransfer.files[0]);
    }
});

async function uploadDocument(file) {
    const form = new FormData();
    form.append("file", file);
    $("assistantDocInfo").textContent = "reading…";
    try {
        const response = await fetch("/api/assistant/upload", { method: "POST", body: form });
        const data = await response.json();
        if (!data.success) throw new Error(data.error);
        assistantDocument = data.text;
        $("assistantDocInfo").textContent = `${file.name} · ${data.characters} chars`;
        toast("Document loaded into the knowledge base.", true);
    } catch (error) {
        assistantDocument = "";
        $("assistantDocInfo").textContent = "no document";
        toast("Upload failed: " + error.message);
    }
}

$("assistantClearBtn").addEventListener("click", () => {
    assistantDocument = "";
    assistantFile.value = "";
    $("assistantDocInfo").textContent = "no document";
    $("assistantSources").innerHTML = "";
});

$("assistantAskBtn").addEventListener("click", function () {
    const button = this;
    const hint = $("assistantHint");
    withBusy(button, "Thinking…", async () => {
        hint.textContent = "working · model loads on first ask";
        try {
            const data = await postJSON("/api/assistant/ask", {
                question: $("assistantQuestion").value,
                document: assistantDocument,
                use_rag: $("assistantUseRag").checked,
                max_tokens: parseInt($("assistantTokens").value, 10),
            });
            $("assistantAnswer").textContent = data.answer;
            const sources = $("assistantSources");
            sources.innerHTML = "";
            (data.sources || []).forEach((source) => {
                const div = document.createElement("div");
                div.className = "source";
                div.textContent = source;
                sources.appendChild(div);
            });
            hint.textContent = "done";
            refreshStatus();
        } catch (error) {
            hint.textContent = "failed";
            toast("Assistant failed: " + error.message);
        }
    });
});

// ============================================================
// BOOT
// ============================================================
refreshStatus();
setInterval(refreshStatus, 8000);
const initial = (location.hash || "").replace("#", "");
if (initial) showPanel(initial);

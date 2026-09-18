/*
 * Review UI for the recognition backend.
 *
 * Plain ES modules, no build step and no dependencies: open /ui/ and it runs. The page is
 * served by the API itself, so it is same-origin and needs no CORS configuration.
 *
 * Two things shape the design:
 *
 *  1. Every /api/v1 route needs an `X-API-Key` header, which a plain <img src> cannot
 *     send. Images are therefore fetched as blobs and cached as object URLs. The key
 *     never goes into a URL — it would end up in logs and history.
 *  2. The backend deliberately refuses to guess. A plate it could not confirm, and a type
 *     that came from COCO rather than a reference match, are the interesting cases for a
 *     reviewer, so this page shows them prominently instead of hiding them.
 */

const API = "/api/v1";
const PAGE = 24;

/* All user-facing text lives here; translating the UI is editing this object. */
const T = {
  connected: "подключено",
  connecting: "проверка ключа…",
  badKey: "ключ не принят",
  offline: "сервер недоступен",
  noPlate: "номер не найден",
  rejected: "не подтверждён",
  types: {
    truck: "грузовик",
    truck_light: "лёгкий грузовик",
    tractor: "трактор",
    trailer: "прицеп",
    combine_harvester: "комбайн",
    wheel_loader: "погрузчик",
    car: "легковой",
    bus: "автобус",
    motorcycle: "мотоцикл",
    unidentified_vehicle: "техника не определена",
    unknown_vehicle: "неизвестно",
  },
  typeSource: {
    reference_gallery: "эталон",
    badge_text: "надпись",
    detector: "COCO — догадка",
    manual: "вручную",
  },
  plateStatus: {
    accepted: "принят",
    no_matching_plate_format: "не совпал с поддерживаемыми форматами",
    region_needs_review: "регион требует проверки",
    accepted_region: "номер и регион прочитаны раздельно",
    accepted_alpr: "принят",
    below_confidence_threshold: "ниже порога уверенности",
    no_text_recognised: "текст не распознан",
    empty_crop: "пустой фрагмент",
    manually_corrected: "исправлен вручную",
    manually_set: "рамка задана вручную",
    manually_cleared: "рамка снята вручную",
  },
  /* The API answers in English. These are the messages a reviewer can actually trigger;
     anything unmapped falls through verbatim rather than being swallowed. */
  apiErrors: {
    "Image has an active job; wait until it finishes":
      "Фотография уже в обработке — дождитесь окончания",
    "Supported image types: JPEG, PNG, WEBP": "Поддерживаются только JPEG, PNG и WEBP",
    "Declared MIME type does not match image contents":
      "Тип файла не совпадает с его содержимым",
    "Invalid or truncated image": "Файл повреждён или обрезан",
    "Animated or multi-frame images are not supported":
      "Анимированные и многостраничные файлы не поддерживаются",
    "Image exceeds MAX_UPLOAD_MB": "Файл больше разрешённого размера",
    "Image exceeds MAX_IMAGE_PIXELS": "Слишком много пикселей в изображении",
    "Normalized image exceeds MAX_UPLOAD_MB": "После нормализации файл слишком велик",
    "Insufficient image storage": "На диске сервера не хватает места",
    "Correction has inconsistent plate text/readability or invalid values":
      "Правка противоречива: пустой номер с отметкой «прочитан» или недопустимое значение",
    "Annotated image not available yet": "Разметка ещё не готова",
    "Image not found": "Фотография не найдена",
    "Detection not found": "Объект не найден",
    "Invalid or missing API key": "Ключ не принят",
  },
  jobStatus: {
    pending: "в очереди",
    processing: "обрабатывается",
    completed: "готово",
    failed: "ошибка",
    not_processed: "не обработано",
  },
};

const $ = (id) => document.getElementById(id);
const state = {
  key: "",
  rows: [],
  offset: 0,
  filter: "all",
  query: "",
  viewing: null,
  poll: null,
};

/* ---------------------------------------------------------------- transport */

async function api(path, options = {}) {
  const response = await fetch(API + path, {
    ...options,
    headers: { "X-API-Key": state.key, ...(options.headers || {}) },
  });
  if (response.status === 401) throw new Error(T.badKey);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch { /* a non-JSON error body is still just an HTTP status to the user */ }
    throw new Error(T.apiErrors[detail] || detail);
  }
  return response.status === 204 ? null : response.json();
}

/* Object URLs are revoked on eviction, otherwise a long review session leaks blobs. */
const imageCache = new Map();
const CACHE_LIMIT = 220;

async function authImage(path) {
  if (imageCache.has(path)) return imageCache.get(path);
  const promise = (async () => {
    const response = await fetch(API + path, { headers: { "X-API-Key": state.key } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return URL.createObjectURL(await response.blob());
  })();
  imageCache.set(path, promise);
  if (imageCache.size > CACHE_LIMIT) {
    const oldest = imageCache.keys().next().value;
    imageCache.get(oldest)?.then((url) => URL.revokeObjectURL(url)).catch(() => {});
    imageCache.delete(oldest);
  }
  return promise;
}

async function setImage(element, path) {
  try {
    element.src = await authImage(path);
  } catch {
    element.alt = "изображение недоступно";
  }
}

/* ------------------------------------------------------------------- helpers */

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const typeName = (value) => T.types[value] || value || "—";
const pct = (value) => (value == null ? "—" : `${Math.round(value * 100)}%`);

function toast(message, isError = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = "toast" + (isError ? " err" : "");
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, isError ? 6000 : 3000);
}

/** The plate actually reported, or the rejected reading with its reason. */
function plateInfo(det) {
  const plate = det.plate;
  if (det.plate_readable && det.license_plate) {
    return { text: det.license_plate, ok: true, reason: null, raw: plate?.text_raw || null };
  }
  if (plate?.text_raw) {
    return {
      text: plate.text_raw,
      ok: false,
      reason: T.plateStatus[plate.status] || plate.status || T.rejected,
      raw: plate.text_raw,
    };
  }
  return null;
}

/* --------------------------------------------------------------- connection */

async function connect() {
  const key = $("apiKey").value.trim();
  if (!key) return toast("Введите API-ключ", true);
  state.key = key;
  setStatus(T.connecting, "busy");
  try {
    await api("/images?limit=1");
    localStorage.setItem("fable.key", key);
    setStatus(T.connected, "ok");
    $("welcome").hidden = true;
    $("main").hidden = false;
    $("exportJson").hidden = $("exportCsv").hidden = false;
    state.offset = 0;
    state.rows = [];
    await load(true);
  } catch (error) {
    setStatus(error.message === T.badKey ? T.badKey : T.offline, "err");
    toast(error.message, true);
  }
}

function setStatus(text, kind) {
  const node = $("status");
  node.textContent = text;
  node.dataset.state = kind;
}

/* --------------------------------------------------------------- data + grid */

/* `/images` is ordered newest-first, which is what a reviewer wants after an upload;
   `/exports/results.json` carries the detections but is ordered oldest-first and takes no
   sort option. So: page the ids from one, fetch the bodies from the other, and restore the
   requested order client-side. Two cheap calls instead of one badly ordered list. */
async function load(reset = false) {
  if (reset) { state.offset = 0; state.rows = []; }
  const images = await api(`/images?limit=${PAGE}&offset=${state.offset}`);
  let page = [];
  if (images.length) {
    const query = images.map((image) => `image_id=${image.id}`).join("&");
    const bodies = await api(`/exports/results.json?${query}`);
    const byId = new Map(bodies.map((row) => [row.image_id, row]));
    page = images.map((image) => byId.get(image.id)).filter(Boolean);
  }
  state.rows = reset ? page : state.rows.concat(page);
  state.offset += images.length;
  $("loadMore").hidden = images.length < PAGE;
  render();
  schedulePoll();
}

/* Re-fetch while anything is still queued, then stop. No timer when idle. */
function schedulePoll() {
  clearTimeout(state.poll);
  const busy = state.rows.some((r) => r.status === "pending" || r.status === "processing");
  if (!busy) return;
  state.poll = setTimeout(async () => {
    try {
      const seen = state.offset;
      state.offset = 0;
      state.rows = [];
      await load(true);
      // Keep however many pages the reviewer had already opened.
      while (state.offset < seen && !$("loadMore").hidden) await load(false);
      if (state.viewing) openViewer(state.viewing, false);
    } catch { /* transient failure: the next user action will refresh anyway */ }
  }, 2500);
}

function matches(row) {
  const q = state.query.toLowerCase();
  if (q) {
    const haystack = [
      row.filename,
      ...row.detections.flatMap((d) => [
        d.license_plate, d.plate?.text_raw, typeName(d.vehicle_type), d.manufacturer,
      ]),
    ].filter(Boolean).join(" ").toLowerCase();
    if (!haystack.includes(q)) return false;
  }
  const dets = row.detections;
  switch (state.filter) {
    case "read": return dets.some((d) => d.plate_readable);
    case "rejected": return dets.some((d) => !d.plate_readable && d.plate?.text_raw);
    case "noplate": return dets.every((d) => !d.plate);
    case "guess": return dets.some((d) => d.vehicle_type_source === "detector");
    case "failed": return row.status === "failed";
    default: return true;
  }
}

function render() {
  const rows = state.rows.filter(matches);
  renderStats();
  const grid = $("grid");
  grid.innerHTML = "";
  $("empty").hidden = state.rows.length > 0;

  for (const row of rows) {
    const card = document.createElement("article");
    card.className = "card";
    card.tabIndex = 0;

    const plates = row.detections.map(plateInfo).filter(Boolean);
    const busy = row.status === "pending" || row.status === "processing";

    card.innerHTML = `
      <div class="thumbWrap">
        ${busy ? '<div class="spin">обрабатывается…</div>' : '<img alt="" loading="lazy">'}
      </div>
      <div class="cardBody">
        <div class="cardName" title="${esc(row.filename)}">${esc(row.filename)}</div>
        <div class="plates">
          ${plates.length
            ? plates.map((p) =>
                `<span class="plate ${p.ok ? "ok" : "no"}" title="${esc(p.reason || "принят")}">${esc(p.text)}</span>`
              ).join("")
            : `<span class="tag">${row.detections.length ? T.noPlate : "техника не найдена"}</span>`}
        </div>
        <div class="tags">
          ${row.status === "failed" ? `<span class="tag fail">${T.jobStatus.failed}</span>` : ""}
          ${busy ? `<span class="tag busy">${T.jobStatus[row.status]}</span>` : ""}
          ${row.detections.map((d) =>
            `<span class="tag src-${esc(d.vehicle_type_source || "detector")}">${esc(typeName(d.vehicle_type))}</span>`
          ).join("")}
        </div>
      </div>`;

    if (!busy) setImage(card.querySelector("img"), `/images/${row.image_id}/thumbnail`);
    card.addEventListener("click", () => openViewer(row.image_id));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openViewer(row.image_id); }
    });
    grid.appendChild(card);
  }
}

function renderStats() {
  const rows = state.rows;
  if (!rows.length) { $("stats").hidden = true; return; }
  const dets = rows.flatMap((r) => r.detections);
  const read = dets.filter((d) => d.plate_readable).length;
  const rejected = dets.filter((d) => !d.plate_readable && d.plate?.text_raw).length;
  const guessed = dets.filter((d) => d.vehicle_type_source === "detector").length;
  const failed = rows.filter((r) => r.status === "failed").length;

  $("stats").hidden = false;
  $("stats").innerHTML = [
    [rows.length, "фотографий"],
    [dets.length, "объектов техники"],
    [read, "номеров прочитано"],
    [rejected, "номеров отклонено"],
    [guessed, "тип — догадка COCO"],
    ...(failed ? [[failed, "ошибок обработки"]] : []),
  ].map(([value, label]) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`).join("");
}

/* ------------------------------------------------------------------- viewer */

async function openViewer(imageId, resetScroll = true) {
  const row = state.rows.find((r) => r.image_id === imageId);
  if (!row) return;
  state.viewing = imageId;
  $("viewer").hidden = false;
  $("viewerTitle").textContent = row.filename;
  if (resetScroll) $("side").scrollTop = 0;

  const img = $("stageImg");
  img.onload = () => drawOverlay(row);
  await setImage(img, `/images/${row.image_id}/file`);
  drawOverlay(row);
  renderSide(row);
}

function drawOverlay(row) {
  const img = $("stageImg");
  const svg = $("overlay");
  const width = img.clientWidth, height = img.clientHeight;
  if (!width || !height) return;

  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.left = `${img.offsetLeft}px`;
  svg.style.top = `${img.offsetTop}px`;

  const showVehicle = $("layerVehicle").checked;
  const showPlate = $("layerPlate").checked;
  const showQuad = $("layerQuad").checked;
  const showLabels = $("layerLabels").checked;
  const X = (v) => v * width, Y = (v) => v * height;
  const parts = [];

  row.detections.forEach((det, index) => {
    if (showVehicle && det.bbox) {
      const b = det.bbox;
      parts.push(`<rect x="${X(b.x1)}" y="${Y(b.y1)}" width="${X(b.x2 - b.x1)}" height="${Y(b.y2 - b.y1)}" stroke="var(--vehicle)" data-i="${index}"/>`);
      if (showLabels) {
        const label = [typeName(det.vehicle_type), det.manufacturer, det.license_plate]
          .filter(Boolean).join(" · ");
        parts.push(`<text x="${X(b.x1) + 4}" y="${Math.max(13, Y(b.y1) - 5)}" fill="var(--vehicle)">${esc(label)}</text>`);
      }
    }
    const plate = det.plate;
    if (showPlate && plate?.bbox) {
      const b = plate.bbox;
      const colour = det.plate_readable ? "var(--plate-ok)" : "var(--plate-no)";
      parts.push(`<rect x="${X(b.x1)}" y="${Y(b.y1)}" width="${X(b.x2 - b.x1)}" height="${Y(b.y2 - b.y1)}" stroke="${colour}"/>`);
      if (showLabels) {
        const text = det.plate_readable ? det.license_plate : `${plate.text_raw || "?"} — ${T.rejected}`;
        parts.push(`<text x="${X(b.x1)}" y="${Y(b.y2) + 15}" fill="${colour}">${esc(text)}</text>`);
      }
    }
    if (showQuad && plate?.quad?.length === 4) {
      const points = plate.quad.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");
      parts.push(`<polygon points="${points}"/>`);
    }
  });

  svg.innerHTML = parts.join("");
}

function renderSide(row) {
  const side = $("side");
  const job = [
    `<dt>Статус</dt><dd>${T.jobStatus[row.status] || row.status}</dd>`,
    row.error_message ? `<dt>Ошибка</dt><dd>${esc(row.error_message)}</dd>` : "",
    `<dt>Провайдер</dt><dd>${esc(row.provider || "—")}${row.is_mock ? " (mock!)" : ""}</dd>`,
    `<dt>Размер</dt><dd>${row.width}×${row.height}</dd>`,
  ].join("");

  const cards = row.detections.map((det) => detectionCard(det)).join("") ||
    `<p class="note">Техника на этом кадре не обнаружена.</p>`;

  side.innerHTML = `
    <dl class="kv">${job}</dl>
    ${cards}
    <div class="sideActions">
      <button class="ghost" data-act="reprocess">Обработать заново</button>
      <button class="ghost" data-act="annotated">Скачать разметку</button>
      <button class="ghost" data-act="delete">Удалить фото</button>
    </div>`;

  side.querySelectorAll(".crop img").forEach((img) => setImage(img, img.dataset.src));
  side.querySelector('[data-act="reprocess"]').onclick = () => reprocess(row.image_id);
  side.querySelector('[data-act="delete"]').onclick = () => removeImage(row.image_id);
  side.querySelector('[data-act="annotated"]').onclick = async () => {
    try {
      const url = await authImage(`/images/${row.image_id}/annotated`);
      Object.assign(document.createElement("a"), {
        href: url, download: `${row.filename}-annotated.jpg`,
      }).click();
    } catch { toast("Разметка ещё не готова", true); }
  };
  side.querySelectorAll("[data-save]").forEach((button) => {
    button.onclick = () => saveDetection(button.dataset.save, row.image_id);
  });
}

function detectionCard(det) {
  const plate = det.plate;
  const info = plateInfo(det);
  const source = det.vehicle_type_source || "detector";

  const crops = plate && (plate.crop_filename || plate.rectified_crop_filename)
    ? `<div class="crops">
         ${plate.crop_filename ? `<div class="crop">
           <img data-src="/detections/${det.detection_id}/plate-crop?variant=before" alt="">
           <span>до выравнивания</span></div>` : ""}
         ${plate.rectified_crop_filename ? `<div class="crop">
           <img data-src="/detections/${det.detection_id}/plate-crop?variant=after" alt="">
           <span>после выравнивания перспективы</span></div>` : ""}
       </div>`
    : "";

  const typeOptions = Object.keys(T.types)
    .map((key) => `<option value="${key}"${key === det.vehicle_type ? " selected" : ""}>${T.types[key]}</option>`)
    .join("");

  return `
    <div class="det" data-det="${det.detection_id}">
      <div class="detHead">
        <span class="detType">${esc(typeName(det.vehicle_type))}</span>
        <span class="tag src-${esc(source)}" title="чем определён тип">${T.typeSource[source] || source}</span>
        ${det.manufacturer ? `<span class="tag">${esc(det.manufacturer)}</span>` : ""}
        ${info ? `<span class="plate ${info.ok ? "ok" : "no"}">${esc(info.text)}</span>` : ""}
      </div>

      <dl class="kv">
        <dt>Техника</dt><dd>${pct(det.vehicle_confidence)} · COCO: ${esc(det.source_label || "—")}</dd>
        ${plate ? `<dt>Рамка номера</dt><dd>${pct(plate.detection_confidence)}</dd>
                   <dt>Чтение номера</dt><dd>${pct(plate.ocr_confidence)}${plate.ocr_engine ? ` · ${esc(plate.ocr_engine)}` : ""}${plate.rectified ? " · выпрямлен" : ""}</dd>
                   <dt>Формат</dt><dd>${esc(plate.format || "не определён")}${plate.region ? ` · ${esc(plate.region)}` : ""}</dd>` : ""}
        ${det.vehicle_type_confidence != null ? `<dt>${det.vehicle_type_source === "reference_gallery" ? "Сходство с эталоном" : det.vehicle_type_source === "badge_text" ? "Чтение марки" : "Оценка типа"}</dt><dd>${pct(det.vehicle_type_confidence)}</dd>` : ""}
      </dl>

      ${info && !info.ok
        ? `<p class="note"><b>Номер не подтверждён.</b> Прочитано «${esc(info.raw)}», причина: ${esc(info.reason)}.
             Проверьте предложенный номер по фрагменту фотографии и подтвердите кнопкой «Сохранить».</p>`
        : ""}
      ${!plate ? `<p class="note">Номерной знак на этой машине не найден.</p>` : ""}
      ${source === "detector"
        ? `<p class="note">Тип взят из класса COCO — в нём <b>нет сельхозтехники</b>.
             Добавьте эталонные фотографии этого ракурса, чтобы получить точный тип.</p>`
        : ""}

      ${crops}

      <div class="editRow">
        <input type="text" placeholder="номер, напр. 160 ALV 10"
               value="${esc(det.license_plate || plate?.text_raw || "")}" data-field="plate">
        <select data-field="type">${typeOptions}</select>
        <button class="save" data-save="${det.detection_id}">Сохранить</button>
      </div>
    </div>`;
}

/* ------------------------------------------------------------------ actions */

async function saveDetection(detectionId, imageId) {
  const card = document.querySelector(`[data-det="${detectionId}"]`);
  const text = card.querySelector('[data-field="plate"]').value.trim();
  const type = card.querySelector('[data-field="type"]').value;
  const body = { vehicle_type: type };
  // An empty box means "unreadable": the API clears the text when plate_readable is false.
  if (text) { body.license_plate = text; body.plate_readable = true; }
  else { body.plate_readable = false; }
  try {
    await api(`/detections/${detectionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    toast("Сохранено");
    await load(true);
    openViewer(imageId, false);
  } catch (error) {
    toast(`Не сохранено: ${error.message}`, true);
  }
}

async function reprocess(imageId) {
  try {
    await api(`/images/${imageId}/process`, { method: "POST" });
    toast("Поставлено в очередь");
    await load(true);
  } catch (error) { toast(error.message, true); }
}

async function removeImage(imageId) {
  if (!confirm("Удалить фотографию и все её результаты? Это необратимо.")) return;
  try {
    await api(`/images/${imageId}`, { method: "DELETE" });
    closeViewer();
    toast("Удалено");
    await load(true);
  } catch (error) { toast(error.message, true); }
}

function closeViewer() {
  $("viewer").hidden = true;
  state.viewing = null;
}

function step(delta) {
  const rows = state.rows.filter(matches);
  const index = rows.findIndex((r) => r.image_id === state.viewing);
  const next = rows[index + delta];
  if (next) openViewer(next.image_id);
}

/* ------------------------------------------------------------------- upload */

async function upload(files) {
  const list = $("uploadList");
  const accepted = [...files].filter((f) => /^image\/(jpeg|png|webp)$/.test(f.type));
  if (!accepted.length) return toast("Поддерживаются JPEG, PNG и WEBP", true);

  // The API rolls a whole batch back if one file is rejected, so send them one by one:
  // a single bad photo should not discard the rest of the drop.
  for (const file of accepted) {
    const row = document.createElement("div");
    row.className = "uploadRow";
    row.innerHTML = `<span class="name">${esc(file.name)}</span><span class="st">загрузка…</span>`;
    list.prepend(row);
    try {
      const form = new FormData();
      form.append("files", file, file.name);
      await api("/images?process=true", { method: "POST", body: form });
      row.classList.add("ok");
      row.querySelector(".st").textContent = "в очереди";
      setTimeout(() => row.remove(), 4000);
    } catch (error) {
      row.classList.add("err");
      row.querySelector(".st").textContent = error.message;
    }
  }
  await load(true);
}

/* --------------------------------------------------------------------- init */

function init() {
  const saved = localStorage.getItem("fable.key");
  if (saved) { $("apiKey").value = saved; connect(); }

  $("connect").onclick = connect;
  $("apiKey").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  $("refresh").onclick = () => load(true).catch((e) => toast(e.message, true));
  $("loadMore").onclick = () => load(false).catch((e) => toast(e.message, true));

  const zone = $("dropzone");
  const picker = $("fileInput");
  zone.onclick = () => picker.click();
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter") picker.click(); });
  picker.onchange = () => { upload(picker.files); picker.value = ""; };
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", (e) => upload(e.dataTransfer.files));

  $("search").addEventListener("input", (e) => { state.query = e.target.value; render(); });
  $("filters").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    $("filters").querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    state.filter = chip.dataset.filter;
    render();
  });

  $("viewerClose").onclick = closeViewer;
  $("prevImg").onclick = () => step(-1);
  $("nextImg").onclick = () => step(1);
  ["layerVehicle", "layerPlate", "layerQuad", "layerLabels"].forEach((id) => {
    $(id).onchange = () => {
      const row = state.rows.find((r) => r.image_id === state.viewing);
      if (row) drawOverlay(row);
    };
  });
  addEventListener("resize", () => {
    const row = state.rows.find((r) => r.image_id === state.viewing);
    if (row) drawOverlay(row);
  });
  addEventListener("keydown", (e) => {
    if ($("viewer").hidden) return;
    if (e.key === "Escape") closeViewer();
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });

  // Exports need the header too, so they are fetched and handed to the browser as blobs.
  for (const [id, path, name] of [
    ["exportJson", "/exports/results.json", "results.json"],
    ["exportCsv", "/exports/results.csv", "results.csv"],
  ]) {
    $(id).onclick = async (e) => {
      e.preventDefault();
      try {
        const response = await fetch(API + path, { headers: { "X-API-Key": state.key } });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const url = URL.createObjectURL(await response.blob());
        Object.assign(document.createElement("a"), { href: url, download: name }).click();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
      } catch (error) { toast(error.message, true); }
    };
  }
}

init();

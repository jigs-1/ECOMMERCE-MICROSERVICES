const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;
const OUT_DIR = "C:\\Users\\jigna\\OneDrive\\Documents\\Playground\\cameo\\outputs\\cameo_presentation";
const SCRATCH_DIR = path.join(OUT_DIR, "scratch");
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");

const COLORS = {
  ink: "#0E1726",
  text: "#243447",
  muted: "#617287",
  paper: "#F7F2E8",
  white: "#FFFFFF",
  emerald: "#0D8F83",
  emeraldDark: "#0A675F",
  orange: "#E38743",
  coral: "#DB6554",
  gold: "#D9B44A",
  sky: "#7AA7D8",
  lavender: "#B39AD9",
  line: "#D8D0C3",
  panel: "#FFFDF8",
  soft: "#F1ECE2",
  danger: "#B54E45",
  ok: "#2D8A58",
};

const FONT = {
  title: "Poppins",
  body: "Lato",
  mono: "Aptos Mono",
};

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
}

function addRect(slide, left, top, width, height, fill, line = COLORS.line, lineWidth = 0) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: lineWidth },
  });
}

function addLine(slide, left, top, width, height, fill, lineWidth = 0) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill, width: lineWidth },
  });
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: "#00000000",
    line: { style: "solid", fill: "#00000000", width: 0 },
  });
  shape.text = text;
  shape.text.typeface = opts.typeface || FONT.body;
  shape.text.fontSize = opts.fontSize || 22;
  shape.text.color = opts.color || COLORS.text;
  shape.text.bold = Boolean(opts.bold);
  shape.text.alignment = opts.alignment || "left";
  shape.text.verticalAlignment = opts.verticalAlignment || "top";
  shape.text.insets = { left: 0, right: 0, top: 0, bottom: 0 };
  if (opts.autoFit) shape.text.autoFit = opts.autoFit;
  return shape;
}

function addKicker(slide, text) {
  addText(slide, text, 74, 46, 400, 24, {
    typeface: FONT.mono,
    fontSize: 12,
    color: COLORS.emeraldDark,
    bold: true,
  });
}

function addTitle(slide, title, subtitle, accent = COLORS.emerald) {
  addLine(slide, 66, 76, 8, 92, accent);
  addText(slide, title, 92, 78, 760, 88, {
    typeface: FONT.title,
    fontSize: 30,
    color: COLORS.ink,
    bold: true,
  });
  if (subtitle) {
    addText(slide, subtitle, 94, 170, 860, 54, {
      typeface: FONT.body,
      fontSize: 17,
      color: COLORS.muted,
    });
  }
}

function card(slide, left, top, width, height, title, body, accent = COLORS.emerald) {
  addRect(slide, left, top, width, height, COLORS.panel, COLORS.line, 1);
  addLine(slide, left, top, 7, height, accent);
  addText(slide, title, left + 20, top + 18, width - 36, 26, {
    typeface: FONT.title,
    fontSize: 18,
    color: COLORS.ink,
    bold: true,
  });
  addText(slide, body, left + 20, top + 52, width - 34, height - 68, {
    typeface: FONT.body,
    fontSize: 15,
    color: COLORS.text,
    autoFit: "shrinkText",
  });
}

function twoColFormula(slide, left, top, label, formula, note, accent = COLORS.orange) {
  addRect(slide, left, top, 538, 126, COLORS.panel, COLORS.line, 1);
  addLine(slide, left, top, 6, 126, accent);
  addText(slide, label, left + 20, top + 16, 180, 22, {
    typeface: FONT.title,
    fontSize: 16,
    color: COLORS.ink,
    bold: true,
  });
  addText(slide, formula, left + 20, top + 46, 490, 28, {
    typeface: FONT.mono,
    fontSize: 18,
    color: COLORS.emeraldDark,
  });
  addText(slide, note, left + 20, top + 80, 494, 28, {
    typeface: FONT.body,
    fontSize: 14,
    color: COLORS.text,
    autoFit: "shrinkText",
  });
}

function sectionBadge(slide, text, left, top, fill = COLORS.soft, color = COLORS.emeraldDark) {
  addRect(slide, left, top, 220, 30, fill, "#00000000", 0);
  addText(slide, text, left + 12, top + 7, 200, 18, {
    typeface: FONT.mono,
    fontSize: 11,
    color,
    bold: true,
  });
}

function addNotes(slide, text) {
  slide.speakerNotes.setText(text);
}

function slide1(p) {
  const slide = p.slides.add();
  slide.background.fill = {
    colorStops: [
      { offset: 0, color: "#F7F2E8" },
      { offset: 65000, color: "#E7F4EF" },
      { offset: 100000, color: "#FDFBF7" },
    ],
    rotation: 18,
  };
  addRect(slide, 780, 70, 430, 580, "#FDF8F0CC", COLORS.line, 1);
  addLine(slide, 818, 116, 90, 90, COLORS.sky);
  addLine(slide, 930, 116, 90, 90, COLORS.emerald);
  addLine(slide, 1042, 116, 90, 90, COLORS.orange);
  addText(slide, "CAMEO", 78, 82, 540, 52, {
    typeface: FONT.mono,
    fontSize: 18,
    color: COLORS.emeraldDark,
    bold: true,
  });
  addText(slide, "Context-Aware Multimodal Emotion & Intent Engine", 78, 128, 660, 120, {
    typeface: FONT.title,
    fontSize: 34,
    color: COLORS.ink,
    bold: true,
    autoFit: "shrinkText",
  });
  addText(slide, "A compact, explainable multimodal system that takes caption + image, predicts emotion, intent, intensity, computes confidence, and generates a safe supportive response.", 82, 265, 620, 88, {
    typeface: FONT.body,
    fontSize: 19,
    color: COLORS.text,
    autoFit: "shrinkText",
  });
  card(slide, 80, 400, 290, 150, "Inputs", "Caption text\nImage upload\nOptional authenticated user");
  card(slide, 392, 400, 290, 150, "Outputs", "Emotion class\nIntent class\nIntensity score\nConfidence + response", COLORS.orange);
  card(slide, 810, 258, 360, 136, "Project Promise", "Not just a classifier: CAMEO is a full pipeline with fusion, safety-aware response generation, persistence, and trend analysis.", COLORS.gold);
  card(slide, 810, 418, 360, 168, "Main Claim", "Text and image together preserve more emotional context than either modality alone.\n\nCore architecture: projection -> attention + gating fusion -> multi-head prediction -> confidence -> calibration -> response.", COLORS.coral);
  addNotes(slide, "Cover slide. Emphasize that CAMEO is a multimodal system and not only a classifier. Mention the full user-facing workflow.");
}

function slide2(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "PROJECT OVERVIEW");
  addTitle(slide, "Problem Statement, Objective, and End-to-End Flow", "Why the project exists and what exactly happens from input to output.");
  card(slide, 72, 246, 350, 178, "Problem", "Human emotional expression is often split across modalities. Text may explicitly say 'I need help' while the image carries facial or scene context.", COLORS.coral);
  card(slide, 446, 246, 350, 178, "Objective", "Build a multimodal system that combines NLP and computer vision to predict emotion, intent, and intensity more reliably than weaker baselines.", COLORS.emerald);
  card(slide, 820, 246, 388, 178, "Why It Matters", "Single-modality systems can miss important cues. CAMEO reduces information loss by reasoning over both caption and image together.", COLORS.orange);
  sectionBadge(slide, "WORKFLOW", 72, 454);
  addRect(slide, 72, 492, 1136, 152, COLORS.panel, COLORS.line, 1);
  const steps = [
    "Caption + image input",
    "Text clean + tokenize",
    "Image resize + normalize",
    "Encode both modalities",
    "Project to shared 128-d space",
    "Fuse with attention + gating",
    "Predict + respond + store",
  ];
  steps.forEach((step, i) => {
    const x = 94 + i * 156;
    addRect(slide, x, 530, 128, 70, i % 2 === 0 ? "#EDF8F4" : "#FBF0E9", "#00000000", 0);
    addText(slide, step, x + 10, 548, 108, 42, {
      typeface: FONT.body,
      fontSize: 13,
      color: COLORS.ink,
      bold: true,
      alignment: "center",
      autoFit: "shrinkText",
    });
    if (i < steps.length - 1) {
      addLine(slide, x + 128, 563, 20, 4, COLORS.emerald);
    }
  });
  addNotes(slide, "Summarize the full system flow. This slide sets up the rest of the technical story.");
}

function slide3(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "ARCHITECTURE");
  addTitle(slide, "Neural Architecture from Projection to Final Output", "This slide is the core backbone: aligned modality vectors, fusion, heads, confidence, and response.");
  const boxY = 272;
  const positions = [78, 254, 430, 606, 782, 958];
  const labels = [
    ["Text pooled feature", "h_t"],
    ["Text projection", "z_t = W_t h_t + b_t"],
    ["Image projection", "z_i = W_i h_i + b_i"],
    ["Attention + gating fusion", "f = (alpha_t g_t)z_t + (alpha_i g_i)z_i"],
    ["Prediction heads", "emotion / intent / intensity"],
    ["Confidence + response", "safe supportive output"],
  ];
  labels.forEach(([title, body], idx) => {
    card(slide, positions[idx], boxY, 148, 132, title, body, idx < 3 ? COLORS.sky : idx === 3 ? COLORS.emerald : COLORS.orange);
    if (idx < labels.length - 1) addLine(slide, positions[idx] + 148, boxY + 62, 24, 4, COLORS.emeraldDark);
  });
  twoColFormula(slide, 78, 458, "Shared Projection", "z_t = W_t h_t + b_t   |   z_i = W_i h_i + b_i", "Both modalities are aligned into the same 128-dimensional latent space before fusion.", COLORS.sky);
  twoColFormula(slide, 664, 458, "Fusion Output", "f = (alpha_t * g_t) z_t + (alpha_i * g_i) z_i", "The model learns both relative importance and information flow control for text and image.", COLORS.emerald);
  addNotes(slide, "Explain that this slide begins where Member 3 ownership becomes strongest: projection, fusion, heads, and response logic.");
}

function slide4(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "FUSION LOGIC");
  addTitle(slide, "Why Attention and Gating Are Both Needed", "Attention compares modalities. Gating decides how much signal from each modality should pass.");
  card(slide, 78, 246, 264, 192, "Attention Solves", "Which modality matters more for this sample?\n\nIt produces relative weights alpha_t and alpha_i using softmax, so alpha_t + alpha_i = 1.", COLORS.sky);
  card(slide, 362, 246, 264, 192, "Gating Solves", "How much signal from each modality should actually pass?\n\nIt produces gate values g_t and g_i using sigmoid, so each gate stays between 0 and 1.", COLORS.emerald);
  card(slide, 646, 246, 264, 192, "Why Attention Alone Is Not Enough", "Softmax only gives a relative distribution. It cannot strongly suppress a noisy modality in an absolute way.", COLORS.orange);
  card(slide, 930, 246, 264, 192, "Why Gating Matters", "Gating filters weak or noisy signals. It lets the model reduce modality influence even if that modality still exists.", COLORS.coral);
  sectionBadge(slide, "KEY FORMULAS", 78, 470, "#EAF6F2");
  addRect(slide, 78, 508, 1116, 120, COLORS.panel, COLORS.line, 1);
  addText(slide, "alpha = softmax(W_a [z_t ; z_i] + b_a)", 106, 532, 500, 24, {
    typeface: FONT.mono,
    fontSize: 20,
    color: COLORS.emeraldDark,
  });
  addText(slide, "g = sigma(W_g [z_t ; z_i] + b_g)", 106, 566, 500, 24, {
    typeface: FONT.mono,
    fontSize: 20,
    color: COLORS.emeraldDark,
  });
  addText(slide, "Final multimodal vector:", 668, 520, 220, 22, {
    typeface: FONT.title,
    fontSize: 16,
    color: COLORS.ink,
    bold: true,
  });
  addText(slide, "f = (alpha_t * g_t) z_t + (alpha_i * g_i) z_i", 668, 554, 430, 28, {
    typeface: FONT.mono,
    fontSize: 20,
    color: COLORS.danger,
  });
  addNotes(slide, "This slide should be explained carefully. Attention = relative importance. Gating = information flow control.");
}

function slide5(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "PREDICTION + SAFETY");
  addTitle(slide, "Prediction Heads, Confidence, Calibration, and Response Engine", "From the fused vector to user-facing output.");
  card(slide, 76, 246, 256, 174, "Emotion Head", "Classification over 6 emotion classes:\nhappy, sad, angry, neutral, anxious, hopeful\n\nsoftmax -> probability distribution", COLORS.sky);
  card(slide, 352, 246, 256, 174, "Intent Head", "Classification over 6 intent classes:\ndistress, celebration, frustration, neutral, seeking_help, gratitude\n\nsoftmax -> probability distribution", COLORS.orange);
  card(slide, 628, 246, 256, 174, "Intensity Head", "Regression on a bounded 0 to 1 scale.\n\nsigmoid final layer keeps intensity interpretable and bounded.", COLORS.emerald);
  card(slide, 904, 246, 300, 174, "Confidence", "confidence = max(p_emotion) * max(p_intent)\n\nHigh confidence only when both classification branches are strong.", COLORS.coral);
  twoColFormula(slide, 76, 458, "Calibration Layer", "if strong text cues -> override raw prediction, confidence >= 0.82", "Keyword-aware correction improves robustness in obvious high-signal cases like distress or help-seeking language.", COLORS.gold);
  twoColFormula(slide, 664, 458, "Response Engine", "if distress -> rule-based   else -> template / optional FLAN", "Hybrid response logic keeps higher-risk outputs controlled while allowing supportive low-risk replies.", COLORS.coral);
  addNotes(slide, "Explain the multi-head structure, the confidence formula, and the hybrid safety-aware response engine.");
}

function slide6(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "DATASET");
  addTitle(slide, "Training and Testing Data: What We Used and Why", "A controlled synthetic multimodal dataset was used to validate the pipeline clearly and reproducibly.");
  card(slide, 78, 244, 360, 186, "Main Training / Holdout Dataset", "Generated inside the repo with presentation-oriented captions plus stylized face-like images.\n\nTotal rows: 240\nTrain: 180\nEval: 60", COLORS.emerald);
  card(slide, 460, 244, 360, 186, "Why Stylized Emoji-like Images?", "They give controlled visual emotion cues, balanced classes, stable demos, and easy text-image-label pairing.\n\nThis is a bootstrap dataset, not a claim of full real-world face recognition.", COLORS.orange);
  card(slide, 842, 244, 360, 186, "Extra Evaluation Sets", "12 realistic unseen cases\n6 response-safety prompts\n\nThese test generalization and output behavior beyond the neat holdout split.", COLORS.sky);
  sectionBadge(slide, "CLASS PAIRS", 78, 462, "#FBF0E9", COLORS.orange);
  addRect(slide, 78, 500, 1124, 134, COLORS.panel, COLORS.line, 1);
  const tags = [
    ["happy + celebration", COLORS.gold],
    ["sad + distress", COLORS.sky],
    ["angry + frustration", COLORS.coral],
    ["neutral + neutral", "#C9CBCF"],
    ["anxious + seeking_help", COLORS.lavender],
    ["hopeful + gratitude", "#94C3A2"],
  ];
  tags.forEach(([t, fill], i) => {
    const x = 96 + (i % 3) * 360;
    const y = 520 + Math.floor(i / 3) * 54;
    addRect(slide, x, y, 320, 36, fill, "#00000000", 0);
    addText(slide, t, x + 14, y + 9, 280, 18, {
      typeface: FONT.body,
      fontSize: 15,
      color: COLORS.ink,
      bold: true,
    });
  });
  addNotes(slide, "Be honest here: the dataset is synthetic, balanced, and presentation-oriented. Explain why that was useful and also its limitation.");
}

function slide7(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "TRAINING");
  addTitle(slide, "What Was Trained, Which Losses Were Used, and Why", "A lightweight training strategy was used to keep the project stable and compute-friendly.");
  card(slide, 74, 244, 368, 180, "Trainable Parts", "Projection layers\nAttention + gating fusion block\nEmotion / intensity head\nIntent head\n\nLarge pretrained encoders remain mostly frozen by default.", COLORS.emerald);
  card(slide, 462, 244, 368, 180, "Why Freeze Encoders?", "Full end-to-end fine-tuning is heavier and less stable on a small project dataset.\n\nFrozen encoders + trainable downstream layers are practical for a college-scale system.", COLORS.sky);
  card(slide, 850, 244, 356, 180, "Cached-head Training Path", "A faster CPU-oriented path caches frozen encoder outputs once and trains lighter layers on those cached features.", COLORS.orange);
  twoColFormula(slide, 74, 458, "Loss Functions", "loss = loss_emo + loss_intent + lambda * loss_intensity", "Emotion and intent use cross-entropy. Intensity uses MSE because it is a continuous target.", COLORS.coral);
  twoColFormula(slide, 662, 458, "Training Signals", "softmax -> class probabilities   |   sigmoid -> bounded intensity", "AdamW optimization, seeded runs, and regularization via dropout make training more stable and reproducible.", COLORS.gold);
  addNotes(slide, "This slide is for training logic. Mention cross-entropy for classification, MSE for intensity, and why frozen encoders were a practical choice.");
}

function slide8(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "EVALUATION");
  addTitle(slide, "How Accuracy Was Calculated and What the Results Show", "CAMEO is compared with a majority baseline and a text-only heuristic baseline.");
  const chartBg = addRect(slide, 60, 242, 690, 380, COLORS.panel, COLORS.line, 1);
  chartBg.position = { left: 60, top: 242, width: 690, height: 380 };
  const chart = slide.charts.add("bar");
  chart.position = { left: 86, top: 276, width: 620, height: 290 };
  chart.title = "Holdout Evaluation";
  chart.titleTextStyle.fontSize = 18;
  chart.titleTextStyle.typeface = FONT.title;
  chart.titleTextStyle.fill = COLORS.ink;
  chart.categories = ["Majority", "Heuristic", "CAMEO"];
  chart.barOptions.direction = "column";
  chart.barOptions.grouping = "clustered";
  chart.hasLegend = true;
  chart.legend.position = "bottom";
  chart.legend.textStyle.typeface = FONT.body;
  chart.legend.textStyle.fontSize = 12;
  chart.xAxis.textStyle.typeface = FONT.body;
  chart.xAxis.textStyle.fontSize = 12;
  chart.yAxis.textStyle.typeface = FONT.body;
  chart.yAxis.textStyle.fontSize = 12;
  chart.yAxis.minimum = 0;
  chart.yAxis.maximum = 1;
  chart.dataLabels.showValue = true;
  chart.dataLabels.position = "outEnd";
  chart.dataLabels.textStyle.typeface = FONT.body;
  chart.dataLabels.textStyle.fontSize = 11;
  const s1 = chart.series.add("Emotion Acc");
  s1.values = [0.1667, 0.8, 0.85];
  s1.fill = COLORS.sky;
  const s2 = chart.series.add("Intent Acc");
  s2.values = [0.1667, 0.8, 0.85];
  s2.fill = COLORS.emerald;
  const s3 = chart.series.add("1 - Intensity MAE");
  s3.values = [0.8025, 0.8377, 0.845];
  s3.fill = COLORS.orange;
  card(slide, 778, 246, 432, 148, "Metrics Logic", "Accuracy = correct predictions / total samples\n\nMAE = average |true intensity - predicted intensity|\n\nMacro-F1 gives equal importance to each class.", COLORS.coral);
  card(slide, 778, 414, 432, 96, "Core Results", "CAMEO: emotion acc 0.8500, intent acc 0.8500, intensity MAE 0.1550", COLORS.emerald);
  card(slide, 778, 528, 432, 94, "Interpretation", "CAMEO improves over both weaker baselines, so the multimodal architecture adds measurable value.", COLORS.gold);
  addNotes(slide, "Explain that emotion and intent use classification accuracy, intensity uses MAE, and macro-F1 is also reported to respect class balance.");
}

function slide9(p) {
  const slide = p.slides.add();
  slide.background.fill = COLORS.paper;
  addKicker(slide, "VALIDATION + PRODUCT");
  addTitle(slide, "Realistic Testing, Response Safety, and Product Layer", "The project was evaluated not only on holdout accuracy but also on output behavior and user-facing usability.");
  card(slide, 74, 248, 350, 166, "Realistic Unseen Cases", "12 hand-written unseen cases\nemotion accuracy = 0.6667\nintent accuracy = 0.6667\njoint accuracy = 0.6667", COLORS.sky);
  card(slide, 446, 248, 350, 166, "Safety Checks", "6 curated prompts\npass rate = 1.0000\nrequired supportive phrases present\nforbidden unsafe phrases absent", COLORS.emerald);
  card(slide, 818, 248, 390, 166, "Main Failure Pattern", "Strong negative wording can over-pull the model toward distress, especially when the better label is seeking_help or frustration.", COLORS.coral);
  card(slide, 74, 450, 350, 170, "Product Features", "FastAPI backend\nlogin / register\nsaved prediction history\ntrend analysis\nhealth + readiness endpoints", COLORS.orange);
  card(slide, 446, 450, 350, 170, "Trend Analysis Included?", "Yes. Stored predictions are aggregated into total checks, top emotion, top intent, average intensity, average confidence, and recent emotion/intent patterns.", COLORS.gold);
  card(slide, 818, 450, 390, 170, "Why This Matters", "CAMEO behaves like a usable system, not just a single-run model. Predictions, responses, storage, and analytics are all connected.", COLORS.emerald);
  addNotes(slide, "Use this slide to show that the project has both technical validation and usable product-layer behavior.");
}

function slide10(p) {
  const slide = p.slides.add();
  slide.background.fill = {
    colorStops: [
      { offset: 0, color: "#F8F3E8" },
      { offset: 100000, color: "#F2F7F4" },
    ],
    rotation: 0,
  };
  addKicker(slide, "CONCLUSION");
  addTitle(slide, "Strengths, Limitations, Future Work, and My Role", "A balanced final slide for viva: what is strong, what is limited, and what I contributed.");
  card(slide, 74, 248, 360, 170, "Strengths", "Multimodal reasoning\nshared latent alignment\nattention + gating fusion\nmulti-task prediction\nconfidence + calibration\nsafety-aware response engine", COLORS.emerald);
  card(slide, 456, 248, 360, 170, "Limitations", "Synthetic dataset\nsimplified visual modality\nrealistic unseen cases are harder\ncalibration can overreact to negative wording\nnot enterprise-scale deployment", COLORS.coral);
  card(slide, 838, 248, 370, 170, "Future Work", "More realistic data\nstronger multimodal fine-tuning\nbroader safety red-teaming\nricher deployment and monitoring\nbetter uncertainty estimation", COLORS.orange);
  addRect(slide, 74, 454, 1134, 168, COLORS.panel, COLORS.line, 1);
  addText(slide, "Member 3 Contribution", 98, 476, 260, 24, {
    typeface: FONT.title,
    fontSize: 18,
    color: COLORS.ink,
    bold: true,
  });
  addText(slide, "I handled the central multimodal reasoning pipeline: shared latent alignment, attention + gating fusion, prediction heads, confidence logic, calibration, response safety, and the main evaluation story.", 98, 512, 1080, 44, {
    typeface: FONT.body,
    fontSize: 17,
    color: COLORS.text,
    autoFit: "shrinkText",
  });
  addText(slide, "Best closing line: CAMEO is not just a classifier. It is a full multimodal system that converts caption and image into predictions, confidence, safe response, and product-level user history.", 98, 570, 1080, 34, {
    typeface: FONT.body,
    fontSize: 16,
    color: COLORS.emeraldDark,
    bold: true,
    autoFit: "shrinkText",
  });
  addNotes(slide, "Final slide. End with strengths plus honest limitations, then your role as Member 3.");
}

async function saveBlobToFile(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function renderAndExport(presentation) {
  await ensureDirs();
  for (let i = 0; i < presentation.slides.items.length; i += 1) {
    const slide = presentation.slides.items[i];
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    await saveBlobToFile(preview, path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
  }
  const pptx = await PresentationFile.exportPptx(presentation);
  const outPath = path.join(OUT_DIR, "output.pptx");
  await pptx.save(outPath);
  console.log(outPath);
}

async function main() {
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  p.theme.colorScheme = {
    name: "CameoTheme",
    themeColors: {
      accent1: COLORS.emerald,
      accent2: COLORS.orange,
      accent3: COLORS.sky,
      accent4: COLORS.gold,
      accent5: COLORS.coral,
      accent6: COLORS.lavender,
      bg1: COLORS.paper,
      bg2: COLORS.white,
      tx1: COLORS.ink,
      tx2: COLORS.text,
    },
  };
  slide1(p);
  slide2(p);
  slide3(p);
  slide4(p);
  slide5(p);
  slide6(p);
  slide7(p);
  slide8(p);
  slide9(p);
  slide10(p);
  await renderAndExport(p);
}

await main();

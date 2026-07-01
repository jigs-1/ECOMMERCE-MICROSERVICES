# CAMEO Neural Flow and Test Cases

This document is meant for project revision, viva preparation, and system understanding.

It combines:

- representative test cases
- actual model outputs
- system response behavior
- edge-case analysis
- the complete neural flow from projections to final heads
- mathematics, training logic, and evaluation logic

Primary reference file:
- [PROJECT_DOSSIER_READABLE.md](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/PROJECT_DOSSIER_READABLE.md)

Supporting implementation references:
- [pipeline.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/inference/pipeline.py)
- [encoders.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/encoders.py)
- [fusion.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/fusion.py)
- [heads.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/heads.py)
- [response.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/response.py)
- [train_multimodal.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/train_multimodal.py)
- [train_cached_heads.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/train_cached_heads.py)
- [evaluate_models.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/evaluate_models.py)

---

## 1. Six Core Test Cases

These six cases cover the six main emotion-intent pairs used in the project dataset.

Important note:
- the predictions below were collected from the current project model using `artifacts/cameo_cached_eval.pt`
- the response text is the actual system response
- the reasoning notes below are explanatory human reasoning, not internal chain-of-thought from the model

## 1.1 Celebration Case

**Image:** [happy_celebration_018.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/happy_celebration_018.png)  
**Input text:** `I got selected and I am so happy right now.`

**Expected**
- emotion: `happy`
- intent: `celebration`

**Model output**
- predicted emotion: `happy`
- predicted intent: `celebration`
- intensity: `0.35`
- confidence: `0.9730`
- response mode: `generative`

**System response**
> That is wonderful to hear. You have worked for this, and you deserve this moment. Keep going, one step at a time. This support is based on AI classification, so it may be imperfect and is not a medical diagnosis.

**Why this prediction makes sense**
- text contains explicit positive success words like `selected` and `happy`
- image belongs to the happy/celebration visual category
- calibration logic strongly supports celebration-type interpretation
- low-to-medium intensity is reasonable because the calibrated celebration intensity is around `0.35`

---

## 1.2 Distress Case

**Image:** [sad_distress_014.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/sad_distress_014.png)  
**Input text:** `I feel very low and I do not know how to handle this.`

**Expected**
- emotion: `sad`
- intent: `distress`

**Model output**
- predicted emotion: `sad`
- predicted intent: `distress`
- intensity: `0.82`
- confidence: `0.9727`
- response mode: `rule`

**System response**
> I'm sorry you're going through this. I'm here to listen. If you need urgent help, please contact a trusted person or local support line. This support is based on AI classification, so it may be imperfect and is not a medical diagnosis.

**Why this prediction makes sense**
- text contains strong negative emotional wording: `low`
- intent is not just sadness but distress-sensitive communication
- calibrated intensity is high at `0.82`
- because intent is `distress`, the response engine switches to the rule-based path

---

## 1.3 Frustration Case

**Image:** [angry_frustration_021.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/angry_frustration_021.png)  
**Input text:** `I am frustrated because nothing is going right.`

**Expected**
- emotion: `angry`
- intent: `frustration`

**Model output**
- predicted emotion: `angry`
- predicted intent: `frustration`
- intensity: `0.72`
- confidence: `0.9312`
- response mode: `generative`

**System response**
> That sounds really tough. You are not alone in this, and your effort still matters. Let's take one small step forward together. This support is based on AI classification, so it may be imperfect and is not a medical diagnosis.

**Why this prediction makes sense**
- the word `frustrated` is a direct calibration cue
- the emotional tone is angry/frustrated rather than sad/distress-like
- intensity is fairly high because frustration cases are modeled as strong but not always fully distress-level

---

## 1.4 Neutral Case

**Image:** [neutral_neutral_023.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/neutral_neutral_023.png)  
**Input text:** `Today was normal and nothing unusual happened.`

**Expected**
- emotion: `neutral`
- intent: `neutral`

**Model output**
- predicted emotion: `neutral`
- predicted intent: `neutral`
- intensity: `0.40`
- confidence: `0.9452`
- response mode: `generative`

**System response**
> Thanks for sharing this. You are doing your best, and that is meaningful. I am here to support you. This support is based on AI classification, so it may be imperfect and is not a medical diagnosis.

**Why this prediction makes sense**
- text is routine and low-arousal
- no clear distress, celebration, or frustration markers dominate
- neutral calibration defaults to around `0.40` intensity

---

## 1.5 Seeking Help Case

**Image:** [anxious_seeking_help_032.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/anxious_seeking_help_032.png)  
**Input text:** `I feel anxious and I need help deciding what to do next.`

**Expected**
- emotion: `anxious`
- intent: `seeking_help`

**Model output**
- predicted emotion: `anxious`
- predicted intent: `seeking_help`
- intensity: `0.62`
- confidence: `0.9177`
- response mode: `generative`

**System response**
> Reaching out for help is a strong step. We can break this into small, manageable actions and move forward together. This support is based on AI classification, so it may be imperfect and is not a medical diagnosis.

**Why this prediction makes sense**
- text explicitly contains `anxious`, `need help`, and `what to do next`
- these are direct help-seeking calibration signals
- intensity is moderately high but below distress-rule threshold behavior

---

## 1.6 Gratitude Case

**Image:** [hopeful_gratitude_020.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/hopeful_gratitude_020.png)  
**Input text:** `Thank you, I feel hopeful about what comes next.`

**Expected**
- emotion: `hopeful`
- intent: `gratitude`

**Model output**
- predicted emotion: `hopeful`
- predicted intent: `gratitude`
- intensity: `0.30`
- confidence: `0.9462`
- response mode: `generative`

**System response**
> Thank you for sharing that. Your gratitude and perspective are powerful, and it is great to see your positive momentum. This support is based on AI classification, so it may be imperfect and is not a medical diagnosis.

**Why this prediction makes sense**
- text directly contains `thank you` and `hopeful`
- gratitude and hopeful emotional tone are aligned
- intensity is low-to-moderate, which fits this calmer positive state

---

## 2. Important Edge Cases

These are especially useful for viva because they show you understand not only when the system works, but also where it can fail.

## 2.1 Negative Help-Seeking Can Drift to Distress

**Image:** [anxious_seeking_help_032.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/anxious_seeking_help_032.png)  
**Input text:** `I am overwhelmed by deadlines and need someone to help me decide what to do first.`

**Expected**
- emotion: `anxious`
- intent: `seeking_help`

**Model output**
- predicted emotion: `sad`
- predicted intent: `distress`
- intensity: `0.82`
- confidence: `0.9177`
- response mode: `rule`

**Why this happens**
- the words `overwhelmed` and the overall negative tone strongly activate distress calibration
- the text is indeed asking for help, but the negative lexical weight overpowers the help-seeking interpretation
- this is one of the main known failure patterns of the current system

**What this teaches**
- calibration helps robustness
- but heuristic calibration can also over-pull difficult cases toward distress

---

## 2.2 Celebration Sentence With Negative Word Fragment

**Image:** [happy_celebration_018.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/happy_celebration_018.png)  
**Input text:** `I finally got the offer letter and I cannot stop smiling.`

**Expected**
- emotion: `happy`
- intent: `celebration`

**Model output**
- predicted emotion: `sad`
- predicted intent: `distress`
- intensity: `0.82`
- confidence: `0.9730`
- response mode: `rule`

**Why this happens**
- the token `cannot` is part of the distress keyword set
- the calibration logic is lexical and does not fully understand that `cannot stop smiling` is actually positive
- this is a classic example of heuristic over-triggering

**What this teaches**
- rule-based calibration is practical but limited
- phrase-level semantics are more complex than keyword matching

---

## 2.3 Subtle Neutral Can Drift Positive

**Image:** [neutral_neutral_010.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/neutral_neutral_010.png)  
**Input text:** `Things are steady now. I am not excited or upset, just moving through the day.`

**Expected**
- emotion: `neutral`
- intent: `neutral`

**Model output**
- predicted emotion: `happy`
- predicted intent: `celebration`
- intensity: `0.35`
- confidence: `0.9452`
- response mode: `generative`

**Why this happens**
- the text avoids strong negative words
- the model/calibration may lean toward mild positivity instead of low-arousal neutrality
- subtle neutral phrasing is often harder than strong emotional phrasing

**What this teaches**
- neutral classification can be tricky
- low-arousal emotional states are often more ambiguous than high-signal cases

---

## 2.4 Low-Confidence Distress Prompt Type

**Image:** [sad_distress_029.png](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images/sad_distress_029.png)  
**Input text:** `I may be overreacting, but I feel low and alone.`

**Expected**
- emotion: `sad`
- intent: `distress`

**Model output**
- predicted emotion: `sad`
- predicted intent: `distress`
- intensity: `0.82`
- confidence: `0.9727`
- response mode: `rule`

**Why this is interesting**
- in the safety test dataset, a low-confidence distress case is used to ensure cautious wording exists when confidence is low
- in this actual model run, calibration raises confidence strongly because `low` and `alone` are strong distress cues
- so the final response still goes rule-first

**What this teaches**
- calibration can dominate the final confidence behavior
- distress-like text cues can override weaker uncertainty phrasing

---

## 3. Quick Pattern Summary Across Cases

### Cases the current system handles well

- clear celebration language
- obvious distress language
- direct frustration language
- direct help-seeking language
- explicit gratitude language
- many clean synthetic holdout examples

### Cases the current system struggles with

- negative wording inside otherwise positive sentences
- subtle neutral statements
- help-seeking mixed with strong distress-style words
- cases where heuristics are too literal

---

## 4. Full Neural Flow

This section explains the full model from projection onward, which is the most important part for Member 3.

## 4.1 Inputs to the Member 3 Core

Before your main module begins, the system already has:

- pooled text feature `h_t`
- pooled image feature `h_i`

These are produced by:
- transformer text encoder
- ResNet image encoder

But they are not fused directly.

---

## 4.2 Projection to Shared Latent Space

Text projection:

`z_t = W_t h_t + b_t`

Image projection:

`z_i = W_i h_i + b_i`

Where:
- `W_t`, `b_t` are learned projection weights for text
- `W_i`, `b_i` are learned projection weights for image

After projection:
- `z_t in R^128`
- `z_i in R^128`

### Why projection is needed

- text and image come from different encoder spaces
- raw text and raw image vectors are not naturally aligned
- projection puts them into the same dimension and a more comparable learned space

### What problem projection solves

- representation mismatch
- dimensional mismatch
- fusion readiness

---

## 4.3 Fusion Input

The model concatenates the projected vectors:

`u = [z_t ; z_i]`

Since each is 128-dimensional:

`u in R^256`

Important:
- the model does not stay in 256 dimensions permanently
- `256` is the temporary fusion-input size used to compute modality control signals

---

## 4.4 Attention Logic

Attention weights are computed as:

`alpha = softmax(W_a u + b_a)`

This gives:
- `alpha_t`
- `alpha_i`

and:

`alpha_t + alpha_i = 1`

### What attention solves

Attention answers:

“Which modality matters more for this sample?”

It gives relative importance between text and image.

---

## 4.5 Gating Logic

Gate values are computed as:

`g = sigma(W_g u + b_g)`

This gives:
- `g_t`
- `g_i`

with:

`0 <= g_t <= 1`

`0 <= g_i <= 1`

### What gating solves

Gating answers:

“How much signal from each modality should actually pass through?”

It gives per-modality filtering strength.

---

## 4.6 Why Both Attention and Gating Are Used

Attention and gating solve different but complementary problems.

### Attention

- relative weighting
- which modality should matter more

### Gating

- signal filtering
- how much of each modality should pass

### Why attention alone is not enough

Softmax forces a relative distribution even when both modalities may be weak or noisy.

### Why gating is needed

Gating allows the system to suppress a modality more absolutely when needed.

---

## 4.7 Final Fused Vector

The fusion output is:

`f = (alpha_t * g_t) z_t + (alpha_i * g_i) z_i`

This gives:

`f in R^128`

### Why it returns to 128 dimensions

- `z_t` is 128-dimensional
- `z_i` is 128-dimensional
- weighted sum of two 128-dimensional vectors is still 128-dimensional

So the model temporarily uses a 256-dimensional concatenated vector to compute attention and gates, but the final fused representation returns to 128 dimensions.

---

## 4.8 Prediction Heads

The fused vector `f` goes into task-specific heads.

## 4.8.1 Emotion Head

This is classification over 6 classes:

- happy
- sad
- angry
- neutral
- anxious
- hopeful

Probability output:

`p_e = softmax(y_e)`

Final emotion:

`emotion = argmax(p_e)`

## 4.8.2 Intent Head

This is classification over 6 classes:

- distress
- celebration
- frustration
- neutral
- seeking_help
- gratitude

Probability output:

`p_n = softmax(y_n)`

Final intent:

`intent = argmax(p_n)`

## 4.8.3 Intensity Head

This is bounded regression:

`y_s = sigma(W_s h + b_s)`

So:

`0 <= intensity <= 1`

### Why intensity uses sigmoid

- keeps output within an interpretable range
- avoids impossible negative or over-1 scores

---

## 4.9 Shared Hidden Trunk in EmotionIntensityHead

The project uses a shared hidden trunk before final emotion and intensity outputs:

- Linear `128 -> 64`
- `ReLU`
- `Dropout(0.2)`

### Why ReLU is used

ReLU:

`ReLU(x) = max(0, x)`

It adds non-linearity, so the head can learn more expressive patterns than a purely linear mapping.

### Why shared trunk is used

Emotion category and emotional strength are related, so sharing hidden processing before splitting into separate outputs is efficient and meaningful.

---

## 4.10 Confidence Logic

The project computes confidence as:

`confidence = max(p_emotion) * max(p_intent)`

### What problem confidence solves

- uncertainty awareness
- more cautious response wording
- better output trust interpretation

### Why multiplication was used

Multiplication is stricter than averaging.

It keeps confidence high only when both emotion and intent branches are confident.

---

## 4.11 Calibration Layer

After raw prediction, the pipeline may apply text-based calibration.

If strong lexical cues are found, the system can override:

- emotion
- intent
- intensity

and set:

`confidence = max(confidence, 0.82)`

### What calibration solves

- instability on obvious text cases
- weak behavior in strong high-signal captions
- demo unreliability

### Limitation

It can over-trigger when the text contains misleading negative words, such as:

`I cannot stop smiling`

---

## 4.12 Response Engine

The final response logic is hybrid.

### Distress rule

If:
- intent is `distress`
- or emotion is `sad` and intensity >= `0.7`

then the system uses the rule-based safe response path.

### Otherwise

It uses:
- optional FLAN generation if enabled
- otherwise deterministic supportive templates

### Low-confidence behavior

If confidence is low enough, the system can prepend:

`I may be wrong, but ...`

### Ethics note

Every response ends with:
- AI-based disclaimer
- not a medical diagnosis disclaimer

---

## 5. Training Logic

## 5.1 What Was Trained

By default, the project trains mainly:

- text projection layer
- image projection layer
- fusion block
- emotion/intensity head
- intent head

The large pretrained encoders are mostly kept frozen.

## 5.2 Why Frozen Encoders Were Used

- lower compute
- better stability on small data
- practical for college-project scale

---

## 5.3 Loss Functions

Emotion loss:
- cross-entropy

Intent loss:
- cross-entropy

Intensity loss:
- mean squared error

Combined loss:

`loss = loss_emo + loss_intent + lambda_intensity * loss_intensity`

### Why this design makes sense

- classification tasks use cross-entropy
- continuous scalar task uses regression loss
- weighted combination balances the multi-task objective

---

## 5.4 Cached-Head Training Path

The project also includes a faster training path:

- frozen encoder outputs are cached once
- projection, fusion, and heads are trained on cached features

This is especially useful for:
- CPU evaluation runs
- faster experimentation

---

## 6. Evaluation Logic

## 6.1 Accuracy

Emotion accuracy:

`accuracy = correct emotion predictions / total samples`

Intent accuracy:

`accuracy = correct intent predictions / total samples`

## 6.2 MAE

Intensity uses:

`MAE = average |true_intensity - predicted_intensity|`

## 6.3 Macro-F1

Macro-F1:
- compute F1 for each class
- average equally across classes

This is useful because it respects per-class behavior rather than only overall accuracy.

---

## 6.4 Baselines

The project compares:

### Majority baseline

- always predicts most common train emotion
- always predicts most common train intent
- always predicts mean train intensity

### Heuristic baseline

- uses text calibration only
- no learned multimodal reasoning

### CAMEO model

- learned multimodal projection + fusion + heads pipeline

---

## 6.5 Holdout Results

From [eval_metrics_clean.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/eval_metrics_clean.json):

| Model | Emotion Acc | Intent Acc | Intensity MAE |
|---|---:|---:|---:|
| Majority baseline | 0.1667 | 0.1667 | 0.1975 |
| Heuristic baseline | 0.8000 | 0.8000 | 0.1623 |
| CAMEO | 0.8500 | 0.8500 | 0.1550 |

### Interpretation

- CAMEO clearly beats majority baseline
- CAMEO improves over the text-only heuristic baseline
- intensity error also improves

---

## 6.6 Realistic and Safety Testing

### Realistic unseen cases

From [realistic_eval_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/realistic_eval_results.json):

- 12 cases
- emotion accuracy = `0.6667`
- intent accuracy = `0.6667`
- joint accuracy = `0.6667`

### What joint accuracy means

Both emotion and intent must be correct at the same time.

### Safety checks

From [response_safety_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/response_safety_results.json):

- 6 curated prompts
- pass rate = `1.0000`

This means:
- correct mode switching
- required supportive phrases present
- forbidden unsafe phrases absent

---

## 7. Final Viva-Useful Takeaways

## 7.1 Strongest technical story for Member 3

Your strongest technical story is:

- shared latent projection
- attention + gating fusion
- multi-head prediction
- confidence logic
- calibration layer
- hybrid safety-aware response engine
- baseline-driven evaluation

## 7.2 Strongest honest limitation story

Your strongest honest limitation story is:

- the dataset is synthetic and presentation-oriented
- stylized images help the architecture demonstration but do not prove full real-world generalization
- calibration improves robustness but can over-trigger on strong negative words

## 7.3 Best one-paragraph description

“CAMEO is a multimodal emotion and intent understanding system. It separately encodes text and image, projects both into a shared 128-dimensional latent space, fuses them using attention and gating, predicts emotion, intent, and intensity through task-specific heads, computes confidence, optionally calibrates strong text-driven cases, and then generates a safe supportive response. It is wrapped inside a usable system with history, trend analysis, and explicit evaluation against baselines.”

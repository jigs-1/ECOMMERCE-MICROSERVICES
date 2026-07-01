# CAMEO Project Dossier

## 1. Project Identity

**Full name:** Context-Aware Multimodal Emotion & Intent Engine  
**Short name:** CAMEO

### 1.1 Problem Statement

Human emotional expression is rarely purely textual or purely visual. A caption may say one thing while the image conveys something else. CAMEO is designed to:

- take both text and image as input
- estimate emotion, intent, and emotional intensity
- generate a supportive response
- provide a usable product layer with login, saved history, and trend analysis

### 1.2 Core Objective

Build a multimodal deep learning system that combines NLP and computer vision to improve affect-aware inference over single-modality baselines.

---

## 2. Exact Team Division for 3 Members

This is the final recommended split for viva and presentation.

The goal is:

- Member 3 remains the owner of the most important technical intelligence in the project
- Member 1 and Member 2 still sound like they contributed across the complete pipeline
- the panel should feel that all three members understand the system end to end
- the speaking order should naturally build from input modules to the main multimodal brain

### 2.1 Final role positioning

### Member 1: Text Pipeline + Data Understanding + Input Preparation

Member 1 should not sound like "only preprocessing." This member should sound responsible for how raw user language becomes a meaningful machine-readable representation and how text contributes to downstream multimodal reasoning.

Owns:

- problem framing from the text side
- why captions matter in emotion and intent understanding
- text preprocessing
- tokenization
- attention masks
- transformer-based text encoding
- pooled text embedding
- text projection into shared latent space
- text-related data patterns and limitations
- examples where text carries explicit intent better than image

Should sound involved in:

- input design
- data cleaning decisions
- feature readiness before fusion
- explaining why text features are critical to the final system

Must explain confidently:

- why text cleaning is necessary
- what tokenization does
- what `input_ids` and `attention_mask` mean
- how contextual embeddings differ from bag-of-words or TF-IDF
- why pretrained language models are useful in low-data settings
- why text is still important even in a multimodal system
- why text is projected before fusion

### Member 2: Vision Pipeline + API/Product Layer + Usability

Member 2 should sound like the owner of the complete image path plus the product-facing intelligence that makes the project usable in real life, not just a model on paper.

Owns:

- why image modality matters
- image preprocessing
- image normalization and resizing
- CNN-based image feature extraction
- image projection into shared latent space
- request flow from frontend to backend
- API behavior
- authentication and session flow
- persistent history
- trend analysis
- dashboard behavior and saved results
- deployment-oriented explanation of the usable system

Should sound involved in:

- making model inputs reliable
- converting raw uploaded images into model-ready tensors
- connecting the model to the user-facing application
- making the project look like a real product rather than only an academic model

Must explain confidently:

- why resizing and normalization are needed
- how a CNN extracts semantic visual features
- why pretrained image models are useful
- why image features are projected before fusion
- how API input/output flow works
- why login and persistent storage improve the project
- how trend analysis is computed from past predictions

### Member 3: Core Multimodal Intelligence, Prediction Logic, Safety, and Evaluation

This should be explained by you. This is the "brain of the project" role.

Owns:

- overall architecture
- multimodal reasoning
- shared latent representation
- attention and gating fusion
- fused representation logic
- prediction heads
- emotion classification
- intent classification
- intensity regression
- confidence logic
- calibration behavior
- response generation
- safety logic
- distress handling
- evaluation
- baseline comparison
- strengths
- limitations
- future improvements
- overall integration story

Must explain confidently:

- why multimodal beats single-modality systems
- why shared latent projection is used
- why attention and gating are both used
- why emotion and intent are separate tasks
- why intensity is modeled as regression
- how confidence is computed
- why the response engine is hybrid and safety-oriented
- how the full system turns raw text and image into a safe supportive response

### 2.2 What each member should say about contribution

Use this wording style so Members 1 and 2 sound genuinely involved throughout the build:

- Member 1 can say: "I worked on the text pipeline and on preparing language features that feed the final multimodal model."
- Member 2 can say: "I worked on the image pipeline and on the product layer that connects the model to real users."
- Member 3 can say: "I integrated the multimodal core, prediction heads, response engine, and evaluation strategy."

This creates a balanced impression:

- Member 1 prepared one major half of the intelligence input
- Member 2 prepared the other major half and productized the system
- Member 3 integrated the central reasoning and final outputs

### 2.3 Member-wise preparation packs

These packs are designed so each person's topics come together in one place.

## 2A. Member 1 Preparation Pack

### Member 1 topic identity

`Text understanding -> tokenization -> transformer encoding -> text projection -> contribution to final multimodal reasoning`

### Member 1 exact topic list

- project problem from the text perspective
- why text matters in emotional AI
- text preprocessing in CAMEO
- tokenization
- `input_ids`
- `attention_mask`
- transformer text encoder
- contextual embeddings
- first-token pooled representation
- text projection layer
- text-side strengths and limitations
- examples where text reveals emotion or intent clearly

### Member 1 should open with this idea

"In many real cases, the image alone is not enough. The caption often carries direct emotion, intent, or situational context such as help-seeking, gratitude, frustration, or exhaustion. So in CAMEO, we first convert raw user text into a clean and contextual feature representation that can later be fused with image information."

### Member 1 core explanation flow

1. Text matters because captions can contain explicit emotional and situational information.
2. Raw text is noisy, especially in social-style captions.
3. So CAMEO first cleans text by lowercasing, removing URLs, mentions, hashtags, clutter, and repeated whitespace.
4. Then tokenization converts text into model-readable token IDs and attention masks.
5. The transformer encoder converts the token sequence into contextual hidden states.
6. The pooled text vector represents the text meaning in dense form.
7. A projection layer maps it into a shared dimension so it can be fused with image features later.

### Member 1 minor but useful details to mention

- preprocessing improves consistency
- removing noise helps inference stability
- attention mask prevents the model from treating padding as real content
- contextual embeddings change based on surrounding words
- pretrained transformers help because project datasets are limited
- text can capture intent better than image in many cases
- the text branch is not isolated; it directly supports the final fusion stage

### Member 1 mathematics

Let tokenized text be:

`X = [x_1, x_2, ..., x_n]`

Transformer hidden states:

`H = [h_1, h_2, ..., h_n]`

Pooled text vector:

`h_t = H[0]`

Projected text vector:

`z_t = W_t h_t + b_t`

### Member 1 likely viva questions and answers

#### Q1. Why is text preprocessing needed if transformers are already powerful?

Because noisy text still creates unnecessary variation. Removing URLs, mentions, hashtags, and clutter improves consistency and reduces irrelevant tokens, which helps stable inference.

#### Q2. What is tokenization?

Tokenization converts raw text into model-readable units. In transformers, it produces token IDs and attention masks that let the model process the sequence properly.

#### Q3. What is the purpose of `attention_mask`?

It tells the model which positions are actual tokens and which are padding. This prevents the model from attending to padded positions.

#### Q4. Why use a pretrained transformer instead of training from scratch?

Because pretrained models already capture rich language structure. For a project-scale dataset, this gives much stronger performance and stability than training a text encoder from scratch.

#### Q5. Why does text still matter in a multimodal project?

Because image may show scene context, but text often reveals explicit intent and emotion such as "I need help," "I am exhausted," or "I got selected."

#### Q6. What is the output of the text module?

The output is a dense text feature vector, which is then projected into a shared latent space for fusion with image features.

#### Q7. Why project text to a smaller shared dimension?

Projection makes the text representation compatible with the image representation, reduces computation, and prepares both modalities for structured fusion.

#### Q8. What is one limitation of the text branch?

Text can be ambiguous, sarcastic, or incomplete. Also, emotional meaning may depend on visual context that text alone cannot capture.

### Member 1 ultra-short closing line

"So my part ensures that raw language is converted into a contextual, model-ready feature representation that contributes directly to the final multimodal decision."

## 2B. Member 2 Preparation Pack

### Member 2 topic identity

`Image understanding -> image projection -> API and dashboard flow -> history and trend intelligence`

### Member 2 exact topic list

- why image matters in emotional AI
- image preprocessing in CAMEO
- resizing
- normalization
- tensor conversion
- ResNet-based image encoding
- visual semantic feature extraction
- image projection layer
- frontend-to-backend flow
- FastAPI role
- authentication
- session behavior
- saved history
- trend analytics
- dashboard usability
- why productization makes the project stronger

### Member 2 should open with this idea

"In emotional understanding, the image provides facial, scene, and visual context that text may miss. In CAMEO, we convert uploaded images into normalized tensors, extract high-level visual features using a pretrained CNN, and then connect the model to a usable product through API, login, history, and analytics."

### Member 2 core explanation flow

1. Images carry facial and scene context.
2. Raw images are inconsistent in size and scale.
3. So CAMEO preprocesses images through resizing and normalization.
4. A pretrained CNN backbone extracts semantic visual features.
5. Those image features are projected into the same latent space used by the text branch.
6. Beyond the model, the product layer lets users log in, run predictions, save results, and see history and trends.

### Member 2 minor but useful details to mention

- resizing gives a consistent input size to the CNN
- normalization makes pixel values compatible with pretrained model expectations
- pretrained CNNs are useful because they already learn general visual patterns
- image embeddings alone are useful but incomplete without text in many cases
- FastAPI provides structured endpoints and clear backend organization
- SQLite is sufficient for local demo persistence
- saved history shows repeated use, not one-time inference
- trend analysis makes the system feel application-oriented

### Member 2 mathematics

Let input image be:

`I`

CNN image feature:

`h_i = CNN(I)`

Projected image vector:

`z_i = W_i h_i + b_i`

### Member 2 likely viva questions and answers

#### Q1. Why resize images before inference?

Because CNNs expect consistent spatial dimensions. Resizing ensures stable tensor shapes and predictable feature extraction.

#### Q2. Why normalize images?

Normalization aligns pixel distributions with what the pretrained backbone expects, improving inference stability and performance.

#### Q3. Why use ResNet?

ResNet is a strong and established pretrained CNN architecture. It balances performance, simplicity, and reliability well for a project setting.

#### Q4. What does the image encoder output?

It outputs a dense visual feature vector representing important semantic patterns from the image.

#### Q5. Why is image projection required?

Projection maps visual features into the same shared latent dimension as text features, enabling structured multimodal fusion.

#### Q6. Why include login and saved history in an AI project?

Because the project becomes a usable system rather than only a one-time model demo. It supports repeated use, persistence, and longitudinal observation.

#### Q7. How is trend analysis computed?

Trend analysis uses stored past predictions, especially emotion and intensity outputs over time, to provide a simple view of how outcomes change across sessions.

#### Q8. Why use FastAPI and SQLite?

FastAPI is good for typed backend APIs and easy demo deployment. SQLite is lightweight and practical for local persistence without complex setup.

#### Q9. What is one limitation of the image branch?

Images can be ambiguous, and emotional meaning may not be visually explicit. Scene context alone can be misleading without the caption.

### Member 2 ultra-short closing line

"So my part handles the visual understanding pipeline and the user-facing product flow that makes the model practical, persistent, and demo-ready."

## 2C. Member 3 Preparation Pack

### Member 3 topic identity

`shared latent space -> attention and gating fusion -> prediction heads -> confidence -> response engine -> safety -> evaluation`

### Member 3 exact topic list

- overall end-to-end architecture
- why multimodal learning is needed
- shared latent representation
- text and image projection alignment
- attention fusion
- gating fusion
- fused vector construction
- emotion head
- intent head
- intensity head
- classification versus regression design
- confidence score
- calibration logic
- response engine
- hybrid rule-plus-template behavior
- safety-first distress handling
- evaluation metrics
- baseline comparison
- strengths, limitations, future improvements
- integration of complete system

### Member 3 should open with this idea

"My role is the central multimodal intelligence of the project. After text and image are converted into aligned feature vectors, CAMEO combines them through attention and gating, predicts emotion, intent, and intensity, computes confidence, and then produces a safe supportive response."

### Member 3 core explanation flow

1. Text and image are first encoded separately.
2. Their features are projected into a shared latent space.
3. Attention computes relative importance across modalities.
4. Gating controls how much information should pass through from each modality.
5. The fused representation becomes the shared multimodal understanding.
6. Separate heads predict emotion, intent, and intensity.
7. Confidence is derived from prediction certainty.
8. The response engine uses prediction outputs plus safety logic to generate a helpful response.
9. Evaluation shows improvement over weaker baselines.

### Member 3 minor but useful details to mention

- shared latent space makes cross-modal interaction easier
- attention is not the same as full explanation but gives useful weighting
- gating provides multiplicative filtering
- multi-task learning captures different but related outputs
- emotion and intent are related but not identical
- intensity uses regression because it is continuous
- safety-sensitive cases should not rely on unconstrained generation
- product features strengthen deployment realism, but the core novelty is still multimodal intelligence

### Member 3 mathematics

Projected text vector:

`z_t in R^128`

Projected image vector:

`z_i in R^128`

Attention-style weighting idea:

`alpha = softmax([s_t, s_i])`

Weighted combination:

`u = alpha_t z_t + alpha_i z_i`

Gate:

`g = sigma(W_g [z_t ; z_i] + b_g)`

Fused vector:

`f = g odot u`

Emotion logits:

`y_e = W_e f + b_e`

Intent logits:

`y_n = W_n f + b_n`

Intensity score:

`y_s = sigma(W_s f + b_s)`

Simple confidence idea:

`confidence = max(p_emotion) * max(p_intent)`

### Member 3 likely viva questions and answers

#### Q1. Why use multimodal learning here?

Because emotional meaning is distributed across text and image. Combining both reduces information loss and improves understanding compared with single-modality approaches.

#### Q2. Why project both modalities into the same latent dimension?

Because the original feature spaces of text and image are different. Projection aligns them into a comparable shared space and makes fusion computationally manageable.

#### Q3. Why use both attention and gating?

Attention provides relative weighting of modalities, while gating gives multiplicative control over information flow. Together they create more flexible fusion than simple concatenation.

#### Q4. Why not just concatenate text and image features?

Concatenation is simple but static. It does not explicitly model dynamic weighting or selective information flow between modalities.

#### Q5. Why separate emotion and intent heads?

Because they represent different targets. Emotion answers what is being felt, while intent answers what kind of communicative situation the user is expressing.

#### Q6. Why treat intensity as regression?

Because emotional intensity is continuous and naturally fits a bounded scalar prediction rather than a hard class label.

#### Q7. How is confidence computed?

Confidence is based on how certain the model is in its emotion and intent predictions, so final trust reflects both tasks rather than only one.

#### Q8. Why is the response engine hybrid?

Because safety-sensitive cases require deterministic handling, while lower-risk situations can use supportive templated or guided generation. This makes the system safer and more controllable.

#### Q9. Why is distress handling important?

Because emotionally sensitive systems must avoid careless outputs in high-risk contexts. A safety-first approach improves reliability and responsibility.

#### Q10. What do your results show?

The model outperforms weaker baselines on holdout evaluation and achieves full pass rate on response safety checks, which supports both predictive quality and safer output behavior.

#### Q11. What are your main limitations?

- synthetic and presentation-oriented dataset
- harder realistic unseen cases
- imperfect calibration on strongly negative wording
- limited explainability
- demo-grade deployment stack rather than production-scale deployment

#### Q12. What would you improve next?

- more realistic data
- stronger fine-tuning
- better calibration
- broader safety testing
- stronger deployment infrastructure

### Member 3 ultra-short closing line

"So my part is the integrated intelligence layer that turns prepared text and image features into final predictions, confidence, and safe supportive response behavior."

### 2.4 How to make the whole team sound balanced

Use these rules during viva:

- Member 1 should occasionally mention that the text vector is later fused with image features
- Member 2 should occasionally mention that image features are prepared for the common fusion stage
- Member 3 should explicitly credit Member 1 and Member 2 by saying the core module depends on strong text and image representations
- no member should speak as if their module works alone
- all three should use the same end-to-end vocabulary: input, encoding, projection, fusion, prediction, response, storage

### 2.5 What each member must remember at minimum

### Member 1 minimum memory set

- one reason text matters
- preprocessing steps
- meaning of tokenization
- meaning of attention mask
- transformer output idea
- pooled text vector
- text projection purpose
- one limitation

### Member 2 minimum memory set

- one reason image matters
- resizing and normalization
- CNN feature extraction idea
- image projection purpose
- API flow
- login/history purpose
- trend analysis purpose
- one limitation

### Member 3 minimum memory set

- why multimodal
- why shared latent projection
- why attention plus gating
- why three heads
- why confidence is needed
- why safety layer is needed
- evaluation numbers
- limitations and future work

### 2.6 Full project ownership notes for lead presenter

This section is for you specifically.

Use it to prepare like the person who deeply understands the whole build from implementation to product behavior.

Important viva stance:

- do not say other members did nothing
- do say that you handled the central integration, multimodal logic, response design, evaluation, and final system behavior
- do show that you understand every module, even if another member presents part of it

### 2.6.1 One-line ownership statement you can use

"I handled the main multimodal integration of the project end to end, including aligning the text and image branches, designing the fusion and prediction logic, building the safety-aware response flow, connecting model behavior to the API and product layer, and validating the system through evaluation and baselines."

### 2.6.2 Full implementation details you should know

#### A. Project objective and system shape

You should know that the project is not just a classifier.

It is a full multimodal pipeline that:

- accepts caption plus image
- predicts emotion
- predicts intent
- predicts intensity
- computes confidence
- generates a supportive response
- optionally stores predictions for authenticated users
- exposes history and trend analysis
- provides frontend plus backend usability

#### B. Text preprocessing details

The text cleaning logic is simple but deliberate.

It does:

- lowercasing
- URL removal
- mention removal
- hashtag removal
- non-alphanumeric clutter removal except selected punctuation
- whitespace collapsing

Implementation-level details:

- URL regex: `https?://\S+`
- mention regex: `@\w+`
- hashtag regex: `#\w+`
- non-alphanumeric filter keeps lowercase letters, digits, spaces, and `.,!?`
- the cleaning is tuned for social-caption style text

Why this matters:

- reduces noise
- stabilizes tokenization
- prevents random social metadata from dominating the text branch

#### C. Tokenization details

The tokenizer produces:

- `input_ids`
- `attention_mask`

Important details:

- max length used is `128`
- padding is `max_length`
- truncation is enabled
- special tokens are included
- fast tokenizer path is used

If asked what tokens look like, say:

- they are vocabulary IDs understood by the pretrained transformer
- the attention mask marks real tokens with `1` and padding with `0`

#### D. Text encoder details

The text encoder uses:

- Hugging Face `AutoConfig`
- Hugging Face `AutoModel`
- pretrained model `microsoft/deberta-v3-base`

Important implementation details:

- hidden states are not fully fine-tuned by default
- encoder parameters are frozen unless `trainable=True`
- the first token representation is used as pooled text feature
- that pooled vector is then projected into the shared `128`-dimensional space

Technical statement:

- raw transformer hidden size is larger than the fusion size
- projection compresses the text representation to a manageable common latent dimension

#### E. Image preprocessing details

The image pipeline:

- converts image to RGB
- resizes to `224 x 224`
- converts to tensor
- normalizes using ImageNet mean and standard deviation

Normalization values:

- mean = `[0.485, 0.456, 0.406]`
- std = `[0.229, 0.224, 0.225]`

Additional detail:

- image transform builder uses caching through `lru_cache`
- this avoids rebuilding the same transform repeatedly

#### F. Image encoder details

The image encoder uses:

- `resnet50`
- torchvision pretrained weights `IMAGENET1K_V2`

Implementation behavior:

- the final fully connected layer is removed
- backbone features are pooled
- if features are four-dimensional, they are flattened
- the resulting feature vector is projected into `128` dimensions

Extra detail:

- the encoder class also supports a timm path, but the default system uses torchvision ResNet-50

#### G. Shared latent space details

This is one of the key technical decisions.

Both text and image are projected to the same dimension:

- text vector -> `128`
- image vector -> `128`

Why:

- text and image original features live in different spaces
- common dimensionality makes fusion mathematically simpler
- smaller latent dimension reduces computation
- it avoids blindly mixing unmatched raw features

#### H. Fusion module details

The fusion module is `AttentionGatingFusion`.

Implementation details:

- text and image features are concatenated first
- concatenated size becomes `2D`
- one linear layer produces `2` attention logits
- softmax converts them into modality attention weights
- another linear layer produces `2` gate values
- sigmoid converts them into values between `0` and `1`
- final text contribution is `attention_text * gate_text`
- final image contribution is `attention_image * gate_image`
- fused vector is weighted text feature plus weighted image feature

Why this is strong:

- attention gives comparative importance
- gating gives multiplicative filtering
- combined weighting is more expressive than concatenation alone

#### I. Prediction head details

There are two prediction modules:

- `EmotionIntensityHead`
- `IntentHead`

EmotionIntensityHead details:

- input dimension `128`
- hidden layer `64`
- `ReLU`
- `Dropout(0.2)`
- separate emotion linear layer
- separate intensity branch with sigmoid output

IntentHead details:

- input dimension `128`
- hidden layer `64`
- `ReLU`
- `Dropout(0.2)`
- final linear layer for intent logits

Why this matters:

- emotion and intensity share some internal representation
- intent is modeled separately because communicative purpose is a distinct task

#### J. Label space details

Emotion labels:

- `happy`
- `sad`
- `angry`
- `neutral`
- `anxious`
- `hopeful`

Intent labels:

- `distress`
- `celebration`
- `frustration`
- `neutral`
- `seeking_help`
- `gratitude`

#### K. Output activation details

Emotion and intent:

- logits are converted with softmax
- top class is selected with argmax

Intensity:

- intensity branch uses sigmoid
- output is a bounded scalar between `0` and `1`

Confidence:

- final confidence is `max emotion probability * max intent probability`

Interpretation:

- confidence is high only when both classification branches are confident

#### L. Calibration layer details

This is one of the most important "practical intelligence" additions.

The pipeline has `_calibrate_from_text(raw_text)`.

What it does:

- lowercases and strips text
- extracts alphabetic tokens
- checks keyword sets for celebration, distress, frustration, neutral, help-seeking, and gratitude
- checks help phrases like:
  `what should i do`, `need help`, `can you help`, `do next`
- if no keyword group is activated, it returns `None`
- otherwise it returns a calibrated override for emotion, intent, and intensity

Examples of heuristic effects:

- celebration -> `happy`, around `0.35`
- distress -> `sad`, around `0.82`
- frustration -> `angry`, around `0.72`
- seeking help -> `anxious`, around `0.62`
- gratitude -> `hopeful`, around `0.30`
- neutral -> `neutral`, around `0.40`

Extra detail:

- when calibration triggers, confidence is raised to at least `0.82`

Why this exists:

- small project-scale datasets can make raw neural predictions unstable
- keyword-aware calibration improves reliability on obvious cases
- it is useful for demo robustness

#### M. Response engine details

The response engine is hybrid.

It contains:

- rule-based response path
- optional generative path
- deterministic fallback templates
- ethics note appended to responses

Safety constants:

- `DISTRESS_THRESHOLD = 0.7`

Distress rule behavior:

- if intent is `distress`, use rule-based support
- or if emotion is `sad` and intensity >= `0.7`, also use rule-based support

LLM path details:

- optional via `CAMEO_USE_FLAN=1`
- model name from `CAMEO_FLAN_MODEL`
- default is `google/flan-t5-small`
- disabled by default for deterministic offline demos

Generation prompt behavior:

- asks for a brief warm supportive reply
- avoids mentioning probabilities or image details
- uses deterministic generation style with beam search

Fallback behavior by intent:

- celebration -> congratulatory supportive message
- gratitude -> appreciative positive message
- frustration -> calming supportive message
- seeking_help -> action-oriented supportive message
- sad/angry fallback -> gentle emotional acknowledgment
- general fallback -> neutral support

Uncertainty behavior:

- if confidence < `0.45`, response begins with "I may be wrong, but ..."

Ethics behavior:

- every final response includes a note that the output is AI-based and not a medical diagnosis

#### N. API behavior details

The backend uses FastAPI and exposes:

- `POST /predict`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `GET /history`
- `GET /trends`
- `GET /health`
- `GET /ready`

Predict endpoint details:

- expects `text` as form input
- expects `image` as uploaded file
- accepts optional bearer token header
- rejects empty caption
- enforces maximum caption length
- only accepts PNG and JPEG/JPG images
- checks image byte size
- validates uploaded file as real image
- cleans text before tokenization
- preprocesses image before inference
- records prediction only if a valid logged-in user token is present

Response payload includes:

- emotion
- intensity
- intent
- confidence
- attention weights
- gates
- response text
- response mode

#### O. Security and middleware details

The API includes several practical safety features:

- request ID generation
- response processing time header
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: same-origin`
- centralized exception handler

Rate limiting details:

- in-memory bucketed rate limiter
- keyed by client host
- defaults from environment variables
- designed to prevent burst abuse on `/predict`

API key behavior:

- if `CAMEO_API_KEY` is set, `/predict` requires matching `X-API-Key`

CORS behavior:

- only enabled when origins are configured

Warm-start behavior:

- if `CAMEO_WARM_START=1`, model assets load during startup

Readiness behavior:

- `/ready` reports whether model, tokenizer, weights, and built frontend are available

#### P. Data storage details

Persistence uses SQLite with three tables:

- `users`
- `sessions`
- `predictions`

User handling details:

- usernames are normalized to lowercase
- minimum username length is `3`
- minimum password length is `6`
- password hashing uses `pbkdf2_hmac` with `sha256`
- iteration count is `120000`
- hash uses random salt
- password verification uses `hmac.compare_digest`

Session details:

- session token is generated with `secrets.token_urlsafe(32)`

Prediction storage details:

- stores cleaned text
- stores emotion
- stores intent
- stores intensity
- stores confidence
- stores final response text
- stores created timestamp

#### Q. History and trend logic details

History behavior:

- returns latest saved predictions first
- default history limit is `12`

Trend behavior:

- calculates over latest `50` predictions
- total checks
- top emotion
- top intent
- average intensity
- average confidence
- recent emotions
- recent intents

Implementation detail:

- top labels are computed with `Counter(...).most_common(1)`
- averages are rounded to `4` decimal places

#### R. Configuration details

The configuration loader reads environment variables for:

- app version
- device
- text model
- weights path
- database path
- warm start
- require weights
- local files only
- CORS origins
- API key
- rate limit request count
- rate limit window
- maximum text characters
- maximum image bytes

Important defaults:

- app name = `CAMEO API`
- default text model = `microsoft/deberta-v3-base`
- device defaults to `cuda` if available, else `cpu`
- default weights path = `artifacts/cameo.pt`
- default database path = `artifacts/cameo.db`
- max text chars = `1200`
- max image bytes = `8 MB`

Path handling detail:

- if weights path or database path is relative, it is resolved from the repository root

#### S. Evaluation and testing details

Evaluation artifacts include:

- holdout metrics JSON
- realistic case evaluation results
- response safety results

Important result numbers:

- CAMEO emotion accuracy = `0.8500`
- CAMEO intent accuracy = `0.8500`
- CAMEO intensity MAE = `0.1550`
- response safety pass rate = `100%`

What this proves:

- model beats majority baseline
- model beats heuristic text-only style baseline
- output safety behavior was explicitly checked, not only assumed

#### T. Training and experimentation details

The repository also includes:

- lightweight multimodal training script
- faster cached-head training path for CPU evaluation
- evaluation script
- realistic-case evaluation script
- response safety check script
- CLI prediction script
- ONNX export script

This is useful in viva because it shows the project is not a one-file prototype.

It has:

- training path
- inference path
- evaluation path
- export path
- API path
- frontend path

#### U. Frontend and product details

The UI is built with React and Vite.

Product-facing value:

- login experience
- dashboard-style interaction
- ability to run CAMEO repeatedly
- saved history display
- trend display
- stronger demo readiness

Backend static-serving detail:

- if `ui/dist` exists, FastAPI mounts it and serves the frontend

### 2.6.3 If the panel asks "What exactly did you do?"

Use a version like this:

"I worked on the core integration of the full system. I aligned the text and image encoders into a shared 128-dimensional latent space, designed the attention-plus-gating fusion block, built separate heads for emotion, intent, and intensity, added confidence estimation and a text-based calibration layer for robustness, designed the hybrid safety-aware response engine, and connected the pipeline to the FastAPI backend with authentication, prediction storage, history, and trend analytics. I also worked through evaluation, baselines, and deployment-oriented settings like warm start, offline model loading, and readiness checks."

### 2.6.4 If the panel asks "What is your strongest technical contribution?"

Best answer:

"My strongest contribution is the central multimodal reasoning pipeline. The core novelty is not just that we have text and image, but that we align them into a shared space, fuse them dynamically using attention and gating, predict multiple affect-related outputs together, and then use those outputs inside a safety-aware response system."

### 2.6.5 If the panel asks "Did you understand only your part or the whole project?"

Best answer:

"I understand the whole project flow end to end, including preprocessing, encoding, fusion, prediction, response generation, API behavior, storage, analytics, and evaluation. My main ownership is the central multimodal intelligence, but I can also explain the text branch, image branch, and product layer because they directly affect the final integrated system."

---

## 3. End-to-End Architecture

### 3.1 High-Level Flow

1. User provides caption and image.
2. Text is cleaned and tokenized.
3. Image is resized and normalized.
4. Text encoder generates text features.
5. Image encoder generates image features.
6. Fusion layer combines both using attention and gating.
7. Prediction heads estimate:
   - emotion
   - intent
   - intensity
8. Confidence is computed.
9. Response engine generates supportive output.
10. If logged in, result is stored for history and trend analysis.

### 3.2 One-Line Architecture

`Text -> Transformer Encoder -> Text Projection`

`Image -> CNN Encoder -> Image Projection`

`Projected Text + Projected Image -> Attention + Gating Fusion -> Shared Fused Vector`

`Fused Vector -> Emotion Head + Intent Head + Intensity Head`

`Predictions + Confidence -> Response Engine -> Safe Supportive Reply`

---

## 4. Technology Stack and Why We Used It

### Backend

- **Python**
  - standard for deep learning pipelines
  - strong ecosystem for NLP, CV, and API development

- **PyTorch**
  - flexible dynamic graph framework
  - good for custom multimodal architectures
  - easy to inspect tensors and implement fusion layers

- **Transformers (Hugging Face)**
  - easy access to pretrained language models
  - well-tested tokenization and encoder APIs

- **TorchVision**
  - standard CNN backbones and image transforms

- **FastAPI**
  - fast, typed, easy-to-document backend framework
  - useful for demo deployment and clean API endpoints

- **SQLite**
  - simple local persistence
  - perfect for localhost demo without external DB setup

### Frontend

- **React + Vite**
  - fast development and production build
  - modular UI
  - easy state-driven dashboard

### Why this stack fits this project

- fast to prototype
- strong research-to-product bridge
- easy to explain in a panel
- supports both model experimentation and demo-ready productization

---

## 5. NLP Basics and Our Text Module

### 5.1 Why text matters

Text often carries:

- explicit emotional content
- intent markers such as help-seeking, gratitude, frustration
- situational context not visible in image

Example:

- “I need help figuring this out”
- “I got selected”
- “I feel exhausted”

### 5.2 Text preprocessing used in CAMEO

Implemented in [text.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/text.py).

Steps:

- lowercase the caption
- remove URLs
- remove mentions like `@user`
- remove hashtags
- remove non-alphanumeric clutter except selected punctuation
- collapse repeated whitespace

### 5.3 Why preprocessing is needed

- reduces noise
- improves consistency
- avoids distracting tokens
- makes inference more stable for social-caption style text

### 5.4 Tokenization basics

Tokenization converts raw text into model-readable units.

For transformers, tokenization outputs:

- `input_ids`
- `attention_mask`

#### `input_ids`

Integer IDs corresponding to tokens in the model vocabulary.

#### `attention_mask`

Binary mask:

- `1` for real tokens
- `0` for padding

This prevents the transformer from attending to padding positions.

### 5.5 Why transformer-based text encoding

Classical methods like TF-IDF:

- treat words mostly independently
- lose contextual nuance

Transformers:

- produce contextual embeddings
- meaning of a word depends on surrounding words
- better for emotion and intent understanding

### 5.6 Text encoder used

Implemented in [encoders.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/encoders.py).

Current model:

- `microsoft/deberta-v3-base`

### 5.7 Mathematical representation of text pipeline

Let tokenized text be:

`X = [x_1, x_2, ..., x_n]`

Transformer encoder outputs contextual hidden states:

`H = [h_1, h_2, ..., h_n]`, where `h_i in R^d`

In this project:

- pooled text vector is taken as first-token representation:
  `h_t = H[0]`

Then linear projection is applied:

`z_t = W_t h_t + b_t`

where:

- `h_t in R^768` approximately for DeBERTa-base
- `z_t in R^128`

### 5.8 Why project text into 128 dimensions

- creates a shared latent space with image features
- reduces computational cost
- makes multimodal fusion easier

---

## 6. Computer Vision Basics and Our Image Module

### 6.1 Why image matters

Images provide:

- facial expression cues
- posture and scene information
- contextual signals not available in text

### 6.2 Image preprocessing used in CAMEO

Implemented in [image.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/image.py).

Steps:

- convert to RGB
- resize to `224 x 224`
- convert to tensor
- normalize using ImageNet statistics

Normalization values:

- mean = `[0.485, 0.456, 0.406]`
- std = `[0.229, 0.224, 0.225]`

### 6.3 Why resize to 224 x 224

- standard input size for many pretrained CNN backbones
- stable inference cost
- simpler batching

### 6.4 Why normalization matters

- pretrained CNNs expect normalized input
- stabilizes scale of activations
- improves feature extraction consistency

### 6.5 Image encoder used

Implemented in [encoders.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/encoders.py).

Current backbone:

- `resnet50`

### 6.6 Why ResNet-50

- strong pretrained visual baseline
- well understood in academic settings
- balanced accuracy and computational cost
- easy to justify and explain

### 6.7 Mathematical representation of image pipeline

Let input image be `I`.

After preprocessing:

`I' = preprocess(I)`

CNN backbone extracts a visual representation:

`h_i = CNN(I')`

Then projection:

`z_i = W_i h_i + b_i`

where:

- `h_i in R^2048` typically for ResNet-50 pooled features
- `z_i in R^128`

### 6.8 Why image projection is necessary

- aligns dimensionality with text branch
- enables common fusion space
- reduces overparameterization downstream

---

## 7. Multimodal Fusion: Core of the Project

Implemented in [fusion.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/fusion.py).

This is the most important part of the project.

### 7.1 Why not just concatenate and classify?

Simple concatenation assumes:

- both modalities are equally useful
- all samples need the same combination rule

That is rarely true.

Example:

- caption may be strong, image weak
- image may be strong, caption weak
- modalities may conflict

So we need dynamic fusion.

### 7.2 Fusion inputs

- text feature: `z_t in R^128`
- image feature: `z_i in R^128`

Concatenation:

`s = [z_t ; z_i] in R^256`

### 7.3 Attention computation

The model computes modality scores:

`a = W_a s + b_a`

This gives two logits:

- one for text
- one for image

Then softmax:

`alpha = softmax(a)`

So:

- `alpha_t + alpha_i = 1`
- both are in `[0,1]`

Interpretation:

- `alpha_t` = relative importance of text
- `alpha_i` = relative importance of image

### 7.4 Gating computation

The model also computes gates:

`g = sigmoid(W_g s + b_g)`

So:

- `g_t in [0,1]`
- `g_i in [0,1]`

Interpretation:

- gate controls how much signal from each modality is allowed through

### 7.5 Final fused representation

The code computes:

`f = (alpha_t * g_t) * z_t + (alpha_i * g_i) * z_i`

This is a weighted, gated sum of the two modalities.

### 7.6 Why both attention and gating

Attention answers:

- which modality matters more relatively

Gating answers:

- how much signal from each branch should pass through absolutely

Using both improves flexibility.

### 7.7 Important viva point

If asked why both are needed:

Say:

“Attention gives relative importance between text and image, while gating provides an additional multiplicative control over signal flow. This allows the model to both rank modalities and selectively suppress or amplify them.”

---

## 8. Prediction Heads

Implemented in [heads.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/heads.py).

### 8.1 Why separate heads

Emotion, intent, and intensity are related but not identical tasks.

- emotion = what is felt
- intent = what kind of communicative situation this is
- intensity = how strongly the affect is expressed

### 8.2 Emotion head

Architecture:

- linear layer: `128 -> 64`
- ReLU
- dropout
- linear classifier to emotion classes

Emotion logits:

`y_e = W_e h + b_e`

Emotion probabilities:

`p_e = softmax(y_e)`

### 8.3 Intent head

Architecture:

- linear layer: `128 -> 64`
- ReLU
- dropout
- linear classifier to intent classes

Intent probabilities:

`p_int = softmax(y_int)`

### 8.4 Intensity head

Architecture:

- linear layer to scalar
- sigmoid activation

So:

`s = sigmoid(W_s h + b_s)`

This constrains intensity to `[0,1]`.

### 8.5 Label spaces used

Emotions:

- happy
- sad
- angry
- neutral
- anxious
- hopeful

Intents:

- distress
- celebration
- frustration
- neutral
- seeking_help
- gratitude

---

## 9. Confidence Logic

Implemented in [pipeline.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/inference/pipeline.py).

Confidence is computed as:

`confidence = max(emotion_probs) * max(intent_probs)`

### Why this makes sense

- if emotion prediction is uncertain, confidence should drop
- if intent prediction is uncertain, confidence should also drop
- multiplying them penalizes disagreement or ambiguity

### Interpretation

High confidence means:

- strong emotion confidence
- strong intent confidence

Low confidence means:

- one or both heads are uncertain

---

## 10. Text Calibration Layer

Implemented inside [pipeline.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/inference/pipeline.py).

### Why it exists

Pure neural predictions can be unstable on small, synthetic, or sparse datasets.

So we add a rule-based text calibration layer using keyword sets for:

- celebration
- distress
- frustration
- neutral
- seeking_help
- gratitude

### How it works

The system counts keyword overlaps and chooses the strongest matching intent.

Then it assigns:

- calibrated intent
- corresponding emotion
- default intensity

### Why we added it

- improves demo reliability
- strengthens performance on known emotional phrasing
- gives a stronger baseline over random head behavior

### Limitation

It can over-trigger on certain wording patterns and is not a replacement for a large-scale properly fine-tuned model.

---

## 11. Response Engine and Safety Design

Implemented in [response.py](C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/response.py).

### 11.1 Why a response engine is needed

A prediction system alone is incomplete for a user-facing emotional support application.

We need:

- a response style
- safety-aware behavior
- uncertainty-aware phrasing

### 11.2 Hybrid approach

CAMEO uses:

- rule-based responses for distress
- deterministic supportive templates otherwise
- optional FLAN-T5 generative fallback if enabled

### 11.3 Why distress is rule-first

Distress is a sensitive category.

Rule-first behavior gives:

- predictability
- safer wording
- easier auditing

### 11.4 Distress trigger

Rule path is used if:

- `intent == distress`
- or `(emotion == sad and intensity >= 0.7)`

### 11.5 Uncertainty behavior

If confidence `< 0.45`, reply starts with:

`I may be wrong, but ...`

This is important because:

- it avoids overclaiming
- it signals uncertainty
- it is good human-centered AI behavior

---

## 12. Product Layer: Auth, History, Trends

### 12.1 Why product layer was added

A strong project is not only a model; it is also a usable system.

So we added:

- local account registration
- login
- session-based dashboard
- saved user history
- trend analytics

### 12.2 Why this improves the project

- makes it look like an actual deployable product
- allows repeated use
- supports personal analytics over time
- gives a stronger final demo

### 12.3 Trend analysis logic

Computed from stored predictions:

- total checks
- top emotion
- top intent
- average intensity
- average confidence
- recent emotion sequence
- recent intent sequence

### 12.4 Mathematics for trend analysis

Let history entries be:

`H = {h_1, h_2, ..., h_n}`

Then:

- `top_emotion = mode(emotion_i)`
- `top_intent = mode(intent_i)`
- `avg_intensity = (1/n) * sum(intensity_i)`
- `avg_confidence = (1/n) * sum(confidence_i)`

---

## 13. Base Papers and Academic References

These are strong references to cite in the panel.

### NLP / Transformer references

1. **Attention Is All You Need**  
   Vaswani et al., 2017  
   Why relevant:
   - foundation of transformer architecture
   - explains self-attention and sequence modeling

2. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
   Devlin et al., 2018  
   Why relevant:
   - established pretrained transformer language representation learning

3. **DeBERTa: Decoding-enhanced BERT with Disentangled Attention**  
   He et al., 2020  
   Why relevant:
   - direct lineage for the text encoder family used in this project

### Vision references

4. **Deep Residual Learning for Image Recognition**  
   He et al., 2015  
   Why relevant:
   - introduced ResNet
   - foundational for our image branch

### Multimodal fusion references

5. **Multimodal Machine Learning: A Survey and Taxonomy**  
   Baltrusaitis, Ahuja, Morency, 2018  
   Why relevant:
   - broad foundation for multimodal learning concepts

6. **VisualBERT: A Simple and Performant Baseline for Vision and Language**  
   Li et al., 2019  
   Why relevant:
   - useful reference for multimodal representation ideas

7. **Gated Multimodal Units for Information Fusion**  
   Arevalo et al., 2017  
   Why relevant:
   - directly supports our use of gating in multimodal fusion

### Generative response / instruction tuning reference

8. **Scaling Instruction-Finetuned Language Models** (FLAN family)  
   Chung et al., 2022  
   Why relevant:
   - basis for optional FLAN-T5 supportive response path

---

## 14. Base Idea of Our Model

The closest conceptual base is:

- transformer text branch
- CNN image branch
- shared projection space
- multimodal fusion with gating/attention
- multi-head prediction

This is not a direct copy of one paper. It is a practical hybrid system combining:

- pretrained NLP
- pretrained CV
- lightweight multimodal fusion
- user-facing safe response generation

---

## 15. Improvements We Added Beyond a Basic Base Model

### Improvement 1: Shared latent projection

Why:

- text and image features come from different dimensions and distributions
- projection aligns them into one fusion-ready space

### Improvement 2: Attention + gating fusion

Why:

- better than fixed concatenation
- dynamic modality weighting
- more interpretable output

### Improvement 3: Multi-task output

Why:

- emotion alone is insufficient
- intent adds communicative context
- intensity adds severity/strength information

### Improvement 4: Hybrid response engine

Why:

- end-to-end usability
- safety for distress cases
- uncertainty-aware responses

### Improvement 5: Calibration layer

Why:

- stabilizes performance on project-scale data
- improves demo-time reliability

### Improvement 6: Productization features

Why:

- transforms model into usable system
- helps submission stand out

---

## 16. Evaluation Summary

From [EVALUATION_REPORT.md](C:/Users/jigna/OneDrive/Documents/Playground/cameo/EVALUATION_REPORT.md):

### Holdout split results

| Model | Emotion Acc | Intent Acc | Intensity MAE |
|---|---:|---:|---:|
| Majority baseline | 0.1667 | 0.1667 | 0.1975 |
| Heuristic baseline | 0.8000 | 0.8000 | 0.1623 |
| CAMEO | 0.8500 | 0.8500 | 0.1550 |

### Meaning

CAMEO improves over:

- majority baseline
- heuristic text-only style baseline

This is important to say in viva:

“We did not just build a model; we compared it against weaker baselines and showed measurable improvement.”

### Safety evaluation

- response safety pass rate = `100%`

---

## 17. Known Limitations

Be honest. This increases credibility.

### Data limitation

- dataset is synthetic and presentation-oriented

### Generalization limitation

- realistic unseen cases are harder than holdout split

### Calibration limitation

- strong negative wording can over-push toward distress-like interpretation

### Explainability limitation

- attention and gating are internal signals, not perfect human explanations

### Product limitation

- local SQLite and localhost setup are good for demo, but not enterprise-scale deployment

---

## 18. Possible Viva Questions and Strong Answers

### Q1. Why multimodal instead of only text?

Because emotional meaning can be distributed across modalities. Text may state the event, while image conveys facial or scene context. Multimodal fusion reduces information loss compared with single-modality systems.

### Q2. Why not only image?

Images often miss explicit semantic context such as “lost my job,” “need help,” or “thank you.” Text contributes intent and situation understanding.

### Q3. Why transformer for text?

Transformers produce contextual embeddings, which are stronger than classical lexical features for subtle semantic and emotional language patterns.

### Q4. Why ResNet for image?

ResNet is a strong, established pretrained backbone that balances accuracy, explainability, and implementation simplicity.

### Q5. Why projection to 128 dimensions?

It creates a common latent space, reduces computation, and makes fusion tractable.

### Q6. Why both attention and gating?

Attention gives relative weighting, while gating gives multiplicative control over how much information from each modality passes through.

### Q7. Why separate emotion and intent?

They answer different questions:

- emotion: what is being felt
- intent: what type of communicative situation is being expressed

### Q8. Why sigmoid for intensity?

Because intensity is modeled on a bounded scale from 0 to 1.

### Q9. Why softmax for classification?

Softmax converts logits into a valid probability distribution over classes.

### Q10. Why confidence multiplication?

Because final trust should depend on both emotion certainty and intent certainty.

### Q11. Why rule-based distress responses?

Because distress is safety-sensitive, so deterministic handling is more reliable and auditable than unconstrained generation.

### Q12. Why is login/history useful in an AI project?

It makes the model usable as a real system, supports longitudinal analysis, and shows product thinking beyond isolated inference.

### Q13. Why not train everything end-to-end?

Given project scale and demo constraints, frozen pretrained encoders plus lightweight trainable heads are more efficient and stable.

### Q14. What would you improve next?

- real-world dataset
- stronger multimodal fine-tuning
- better calibration
- more rigorous safety red-teaming
- stronger deployment layer

---

## 19. Sample 3-Person Speaking Order

### Speaker 1: Text Module

Start from:

- why text matters
- preprocessing
- tokenization
- transformer encoding
- text projection

### Speaker 2: Image + Product Layer

Then explain:

- image preprocessing
- ResNet encoding
- image projection
- API flow
- authentication
- saved history
- trend analytics

### Speaker 3: Core Multimodal Logic

Finish with:

- complete architecture
- fusion mathematics
- prediction heads
- response engine
- evaluation and improvements
- limitations and future work

This order feels natural and technically balanced.

---

## 20. Very Short Expert Summary

CAMEO is a multimodal affective AI system that combines a pretrained transformer text encoder and a pretrained ResNet image encoder, projects both into a shared latent space, fuses them using attention and gating, predicts emotion, intent, and intensity via task-specific heads, and produces uncertainty-aware supportive responses through a hybrid safety-first response engine. It is wrapped inside a usable product layer with authentication, persistent history, and trend analytics.

---

## 21. Final Message to the Team

For panel success:

- do not memorize blindly
- understand the role of each layer
- each person should know:
  - input
  - output
  - one equation
  - one design reason
  - one limitation

If that is done, the project will sound real, technically grounded, and team-balanced.

# CAMEO Project Dossier: Readable Study Format

## 1. What This Document Is For

This version is written for preparation and revision.

It does not remove the project details.  
It reorganizes them so you can:

- read them more comfortably
- revise speaker-wise
- understand the complete project flow
- answer viva questions like someone who genuinely built the system

This document keeps the explanations, but presents them in a cleaner order.

---

## 2. Project Identity

**Full name:** Context-Aware Multimodal Emotion & Intent Engine  
**Short name:** CAMEO

### What problem the project solves

Human emotion is often expressed through both caption and image at the same time.

If we only read the text, we may miss facial or scene context.  
If we only see the image, we may miss explicit intent such as:

- asking for help
- expressing gratitude
- showing frustration
- describing a life event

So this project takes both modalities together and tries to understand:

- emotion
- intent
- emotional intensity

Then it produces a supportive response and wraps the whole thing inside a usable application with login, history, and trend analysis.

### Core objective

The main technical goal is to build a multimodal deep learning system that combines NLP and computer vision so the final system performs better than weaker single-modality or naive baselines.

---

## 3. Full End-to-End Story of the Project

This is the best mental model to remember.

### What happens from input to output

1. The user enters a caption and uploads an image.
2. The caption is cleaned.
3. The cleaned caption is tokenized.
4. The image is resized and normalized.
5. A pretrained transformer converts text into a dense text representation.
6. A pretrained CNN converts image into a dense visual representation.
7. Both features are projected into the same shared latent space.
8. A fusion module combines them using attention and gating.
9. The fused representation is passed to prediction heads.
10. The system predicts:
    - emotion
    - intent
    - intensity
11. Confidence is computed from the classification branches.
12. A calibration layer can adjust obviously strong text-driven cases.
13. A hybrid response engine generates a safe supportive reply.
14. If the user is logged in, the result is stored.
15. The stored results are later used for history and trend analysis.

### One-line architecture

`Text -> Cleaning -> Tokenization -> Transformer Encoder -> Text Projection`

`Image -> Resize/Normalize -> CNN Encoder -> Image Projection`

`Projected Text + Projected Image -> Attention + Gating Fusion -> Fused Vector`

`Fused Vector -> Emotion Head + Intent Head + Intensity Head`

`Predictions + Confidence + Calibration -> Response Engine -> Safe Supportive Output`

`Saved Predictions -> History + Trends`

---

## 4. Team Division in a Natural Viva Format

The division should make the panel feel that:

- all three members contributed meaningfully
- Member 3 handled the central intelligence and integration
- Member 1 and Member 2 were still involved in real technical parts of the project

### How to sound natural while speaking

Do not speak as if you are reading a report.

Try to sound like this:

- explain the idea first
- then explain the reason
- then mention the technical term

For example:

Instead of saying:

"Text preprocessing removes noisy social metadata and stabilizes tokenization."

Say:

"Before sending the caption to the model, we clean it a little. We remove noisy things like links, mentions, and hashtags so the text becomes more consistent and easier for the model to understand."

This style sounds more natural in viva.

### Simple spoken-English rule

When possible, use this pattern:

- "What we do is..."
- "The reason we do that is..."
- "Technically, this is called..."

That makes even technical answers sound clear and confident.

### Member 1

**Main identity:** text pipeline and language-side input preparation

This member should sound like the owner of how raw text becomes useful machine-readable information.

Topics:

- why text matters in emotion and intent understanding
- text preprocessing
- tokenization
- attention mask
- transformer text encoding
- contextual embeddings
- pooled text vector
- text projection into shared latent space
- text-side limitations

What this member should sound like:

"I worked on how raw language is cleaned, represented, and converted into features that later contribute to the final multimodal system."

Natural speaking version:

"My part was the text side of the project. I worked on how the raw caption is cleaned, tokenized, and converted into useful language features that later go into the multimodal model."

### Member 2

**Main identity:** image pipeline plus product and API usability

This member should sound like the owner of the visual branch and the practical application layer.

Topics:

- why image matters
- image preprocessing
- resizing and normalization
- CNN feature extraction
- image projection into shared latent space
- API flow
- authentication
- history
- trend analysis
- dashboard and product usability

What this member should sound like:

"I worked on the image understanding path and the user-facing product layer that makes the system usable beyond a one-time model demo."

Natural speaking version:

"My part was the image pipeline and the product layer. I worked on how the uploaded image is processed, how visual features are extracted, and how the full system is connected to login, history, trends, and the frontend."

### Member 3

**Main identity:** core multimodal brain of the project

This should be your role.

Topics:

- shared latent space
- fusion logic
- attention and gating
- prediction heads
- confidence
- calibration
- response engine
- safety behavior
- baselines
- evaluation
- overall integration

What you should sound like:

"I handled the core multimodal reasoning and system integration, where both text and image are aligned, fused, predicted on, converted into a safe response, and evaluated as a full working pipeline."

Natural speaking version:

"My role was the core brain of the project. I worked on how the text and image features are brought together, how the final predictions are made, how confidence and safety are handled, and how the complete system is evaluated."

---

## 5. Member 1 Study Pack: Text Module

### How Member 1 should sound overall

Member 1 should sound clear, simple, and language-focused.

Best tone:

- "The text tells us what the user is actually saying."
- "So first we clean it, then we tokenize it, then we encode it."
- "After that, the text features are sent to the shared multimodal layer."

## 5.1 Why text matters

Text is important because emotional meaning is often stated explicitly in the caption.

For example, a caption may say:

- "I need help"
- "I feel exhausted"
- "I got selected"
- "I am really frustrated"

These are signals that an image may not communicate clearly on its own.

So the text branch is not optional decoration. It carries:

- explicit emotional content
- communicative intent
- situational context

### Good viva line

"The text branch helps capture what the user is actually saying, not just what is visually present."

More natural speaking version:

"The main value of the text branch is that it captures what the user is directly expressing, not just what we can guess from the image."

## 5.2 Text preprocessing

Raw user text is usually noisy, especially if it looks like social media or casual caption text.

So the project first cleans the text before giving it to the transformer.

### What exactly is done

- convert to lowercase
- remove URLs
- remove mentions like `@user`
- remove hashtags
- remove extra non-alphanumeric clutter except selected punctuation
- collapse repeated whitespace

### Why this is done

This improves consistency.

If we leave raw clutter in the text:

- tokenization becomes noisier
- random metadata may distract the model
- inference becomes less stable

Spoken version:

"If we leave the text completely raw, the model may waste attention on things like links or hashtags that do not really help emotion understanding. So cleaning makes the input more consistent."

### Small implementation-level details

The cleaning logic uses regex patterns such as:

- `https?://\S+` for URLs
- `@\w+` for mentions
- `#\w+` for hashtags

The cleaning keeps lowercase letters, digits, spaces, and punctuation like:

- `.`
- `,`
- `!`
- `?`

## 5.3 Tokenization

The transformer cannot directly read raw strings.  
It needs tokenized input.

So tokenization converts the cleaned sentence into model-readable pieces.

### Outputs of tokenization

- `input_ids`
- `attention_mask`

### What `input_ids` means

These are integer IDs representing tokens in the vocabulary of the pretrained transformer.

### What `attention_mask` means

This tells the model which positions are real tokens and which positions are only padding.

- `1` means real token
- `0` means padding

This is important because the transformer should not attend to empty padding positions.

Spoken version:

"The attention mask basically tells the model which tokens are real and which are just padding added for fixed input length."

### Tokenization settings used

- maximum length = `128`
- padding = `max_length`
- truncation = enabled
- special tokens = included

## 5.4 Text encoder

The text encoder is based on a pretrained transformer.

### Model used

- `microsoft/deberta-v3-base`

### Why this model choice makes sense

Pretrained language models already learn useful representations from large corpora.

That means:

- we do not need to train language understanding from scratch
- performance is stronger in low-data project settings
- contextual meaning is preserved better than older lexical methods

Spoken version:

"We used a pretrained transformer because training a language model from scratch is unrealistic for a project like this. A pretrained model already understands a lot about language, so we can use that strength directly."

### What the transformer produces

It generates contextual hidden states.

This means the representation of a word depends on surrounding words.  
That is a major advantage over bag-of-words or TF-IDF.

### Implementation detail

The first-token representation is used as the pooled text feature, and then that feature is projected to the common latent dimension used later in fusion.

## 5.5 Text projection

The raw hidden size of the transformer is larger than what we want to use for fusion.

So we apply a projection layer.

### Why projection is important

- reduces computation
- makes text and image features compatible
- places both modalities in the same shared latent space

### Mathematical form

If the pooled text feature is:

`h_t`

then the projected text vector is:

`z_t = W_t h_t + b_t`

## 5.6 What Member 1 must be ready to answer

### Why preprocess text if transformers are powerful?

Because noisy input still causes unnecessary variation. Cleaning removes irrelevant social noise and improves stability.

### Why use a pretrained transformer?

Because it gives strong contextual language understanding without needing huge project-specific data.

### Why does text matter in a multimodal model?

Because text often expresses intent and emotion directly, while image may only provide indirect context.

### One limitation of the text branch

Text can be ambiguous, sarcastic, incomplete, or dependent on visual context.

## 5.7 Member 1 file ownership and what each file does

These are the main files Member 1 should know by name and function.

### [cameo/core/preprocess/text.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/text.py)

This file contains the text cleaning and tokenization utilities.

It holds the logic for:

- regex-based cleaning
- lowercase conversion
- URL, mention, and hashtag removal
- whitespace cleanup
- tokenization helper flow
- returning `input_ids`, `attention_mask`, and token list

### [cameo/core/models/encoders.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/encoders.py)

Member 1 should know the text side of this file.

It holds the logic for:

- loading transformer configuration
- loading pretrained text model
- freezing encoder parameters by default
- extracting the first-token pooled representation
- projecting text features into the shared latent dimension

### [cameo/core/data/dataset.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/data/dataset.py)

This file is important because it connects raw labeled data to the model.

Member 1 should know that it helps:

- read the manifest rows
- load text and image per sample
- apply text preprocessing
- apply tokenization
- package tensors and labels for training

### Text-related data files Member 1 should know

- [data/sample_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/sample_manifest.csv)
- [data/presentation_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_manifest.csv)
- [data/presentation_train_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_train_manifest.csv)
- [data/presentation_eval_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_eval_manifest.csv)

These manifest files define examples using fields like:

- text
- image path
- emotion label
- intensity value
- intent label

### What Member 1 can claim from the file level

"I understand the full text path from raw caption cleaning up to contextual representation generation and preparation of text features for multimodal fusion."

## 5.8 Member 1 must understand the text-data source story

If the panel asks where the text training and testing data came from, Member 1 should be able to explain it clearly.

### Main text-data source

The main training and holdout-evaluation text data was not collected from a public benchmark dataset in this repo.

Instead, it was generated as a controlled presentation-oriented synthetic dataset.

This happens through:

- [scripts/generate_presentation_data.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/generate_presentation_data.py)

That script defines a `TEXT_BANK` with grouped caption examples for six emotion-intent combinations:

- happy + celebration
- sad + distress
- angry + frustration
- neutral + neutral
- anxious + seeking_help
- hopeful + gratitude

### Why synthetic text was used

Synthetic text was used because this project needed:

- balanced class coverage
- clearly interpretable captions
- stable training behavior
- presentation-friendly examples
- easy mapping between text, image, and labels

In a college project, this is a practical way to demonstrate the full multimodal pipeline without depending on difficult-to-collect sensitive emotional data.

### What this means academically

This is a strength and a limitation at the same time.

Strength:

- the model gets clean class-balanced examples
- each label pairing is easy to explain
- the demo becomes stable

Limitation:

- it is not the same as real-world noisy social data
- performance on synthetic-style examples will usually look better than performance on messy real-life data

### How the text labels were created

The dataset script maps text examples into numeric labels using:

- `EMOTION_TO_LABEL`
- `INTENT_TO_LABEL`

So the manifest stores integer IDs rather than raw label words.

---

## 6. Member 2 Study Pack: Image Module and Product Layer

### How Member 2 should sound overall

Member 2 should sound practical and system-oriented.

Best tone:

- "The image gives visual context."
- "So we standardize it before sending it to the CNN."
- "Then we connect the model to a real usable application."

## 6.1 Why image matters

Images contribute:

- facial expression clues
- body language
- scene context
- visual mood

Sometimes the caption is short and the image provides most of the emotional cue.  
At other times, image and text complement each other.

### Good viva line

"The image branch helps capture visual context that text alone cannot fully express."

More natural speaking version:

"The image branch adds visual context, which is important because emotion is not always fully expressed in words."

## 6.2 Image preprocessing

Raw uploaded images are inconsistent in size and format, so they must be standardized before being passed to the CNN.

### What exactly is done

- convert image to RGB
- resize to `224 x 224`
- convert to tensor
- normalize using ImageNet statistics

### Normalization values

- mean = `[0.485, 0.456, 0.406]`
- std = `[0.229, 0.224, 0.225]`

### Why resizing is needed

CNN backbones expect a consistent input size for stable feature extraction.

### Why normalization is needed

The pretrained vision backbone was trained under normalized input conditions, so matching that input distribution improves reliability.

Spoken version:

"Because the backbone was originally trained on normalized images, we also normalize our inputs so the model sees data in the format it expects."

### Small implementation detail

The image transform builder is cached using `lru_cache`, which avoids rebuilding the same transform repeatedly.

## 6.3 Image encoder

The project uses a pretrained CNN backbone.

### Model used

- `resnet50`

### Pretraining source

- ImageNet weights: `IMAGENET1K_V2`

### Why ResNet-50 is a good choice

- strong and well-established architecture
- easy to explain in viva
- reliable for feature extraction
- practical for project-scale use

### What the encoder does

The final classification layer is removed, and the backbone is used as a feature extractor.

Those features are then flattened if needed and projected into the same shared latent space used by the text branch.

Spoken version:

"We do not use ResNet as a final image classifier here. We use it as a feature extractor, and then we send those features to the multimodal fusion stage."

## 6.4 Image projection

Just like text, image features also need projection before fusion.

### Why this is important

Text and image feature spaces are originally different.

Projection helps:

- align dimensionality
- simplify fusion
- reduce computation
- create a common comparison space

### Mathematical form

If the CNN feature is:

`h_i`

then the projected image vector is:

`z_i = W_i h_i + b_i`

## 6.5 Product layer and API

This project is not only a model notebook or research pipeline.  
It is packaged like a usable system.

### Backend framework

- FastAPI

### Why FastAPI was useful

- clean endpoint design
- typed request/response handling
- easy product-style backend behavior
- good fit for deployment demos

### Main endpoints

- `POST /predict`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `GET /history`
- `GET /trends`
- `GET /health`
- `GET /ready`

### What the predict endpoint checks

- non-empty caption
- max text length
- supported image type
- image size limit
- valid image decoding

### What the predict endpoint returns

- emotion
- intent
- intensity
- confidence
- attention weights
- gate values
- response text
- response mode

## 6.6 Authentication, history, and trends

These features make the project look like a real product.

### Why authentication was included

If the user can register and log in, predictions can be linked to an identity.

That enables:

- saved history
- repeated usage
- longitudinal pattern viewing

### Storage details

SQLite stores:

- users
- sessions
- predictions

### Important implementation details

- usernames are normalized to lowercase
- passwords are hashed using `pbkdf2_hmac` with `sha256`
- session tokens are generated securely

### History behavior

The history endpoint returns recent saved predictions.

### Trend behavior

The trend endpoint summarizes recent prediction behavior across past sessions, including:

- top emotion
- top intent
- average intensity
- average confidence
- recent emotions
- recent intents

### Good viva line

"The product layer shows that the project is not only capable of prediction but also supports real interaction, persistence, and simple personal analytics."

More natural speaking version:

"This part makes the project feel like a real system, not just a model output on one sample."

### One limitation of the image/product branch

Image can still be ambiguous, and the current deployment layer is demo-friendly rather than enterprise-scale.

## 6.7 Member 2 file ownership and what each file does

These are the main files Member 2 should know by name and function.

### [cameo/core/preprocess/image.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/preprocess/image.py)

This file contains the image preprocessing pipeline.

It holds the logic for:

- converting image to RGB
- building reusable image transforms
- resizing to `224 x 224`
- tensor conversion
- ImageNet normalization
- loading image from path when needed

### [cameo/core/models/encoders.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/encoders.py)

Member 2 should know the image side of this file.

It holds the logic for:

- loading the ResNet backbone
- removing the final classification layer
- extracting pooled visual features
- flattening feature maps when needed
- projecting image features into the shared latent dimension

### [cameo/api/main.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/api/main.py)

This is one of the most important product-layer files.

It holds the logic for:

- FastAPI app creation
- request handling
- `/predict` endpoint
- auth endpoints
- history and trends endpoints
- health and readiness endpoints
- middleware
- rate limiting
- optional API key protection
- frontend static serving

### [cameo/api/store.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/api/store.py)

This file contains the persistence logic.

It holds:

- SQLite table setup
- user creation
- password hashing and verification
- session token storage
- prediction recording
- history retrieval
- trend aggregation

### [cameo/api/config.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/api/config.py)

This file contains runtime configuration loading.

It holds:

- environment-variable parsing
- path resolution for weights and database
- rate-limit settings
- text and image input limits
- device selection
- local/offline behavior flags

### Frontend files Member 2 should know

### [ui/src/App.tsx](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/ui/src/App.tsx)

This is the main UI logic file.

It handles:

- text input state
- image upload state
- preset demo scenarios
- service status checking
- auth forms
- prediction calls
- history loading
- trend loading
- result rendering
- session token persistence in local storage

### [ui/src/api.ts](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/ui/src/api.ts)

This file wraps frontend API calls.

It handles:

- predict request
- health request
- readiness request
- register/login requests
- current-user fetch
- history fetch
- trends fetch

### [ui/src/types.ts](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/ui/src/types.ts)

This file stores TypeScript response and entity types for:

- prediction payload
- user payload
- auth payload
- history records
- trends payload
- health payload
- readiness payload

### [ui/src/main.tsx](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/ui/src/main.tsx)

This file mounts the React app.

### [ui/src/styles.css](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/ui/src/styles.css)

This file controls the UI appearance and visual presentation.

### Data and artifact files Member 2 should know

- [artifacts/cameo.db](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/cameo.db)
- [ui/dist/index.html](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/ui/dist/index.html)

These show that:

- predictions and accounts are actually stored
- the frontend is built and ready to serve

### What Member 2 can claim from the file level

"I understand the complete image preprocessing and visual feature path, and I also understand how the backend and frontend connect the model to real users through prediction, login, persistence, and trends."

## 6.8 Member 2 must understand the image-data source story

This is a very important viva topic because panel members may ask why the images look simple or emoji-like.

### Where the training images came from

The main training and holdout-evaluation image data was generated inside the project using:

- [scripts/generate_presentation_data.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/generate_presentation_data.py)

The script creates images in:

- [data/presentation_images](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_images)

### What kind of images were created

They are simple face-like, icon-like, or emoji-like images.

They are not random decorative pictures.

They were intentionally designed as controlled visual emotion cues.

The script draws:

- a colored background based on emotion
- a face outline
- two eyes
- a different mouth pattern depending on the emotional category

Examples:

- happy gets a smiling arc
- sad gets a downward arc
- angry gets a slanted line
- neutral/anxious/hopeful use simpler line-based mouth cues

### Why emoji-like or stylized face images were used

This was done for practical project reasons:

1. The project needed a consistent visual signal for each emotional category.
2. Real emotional image datasets are harder to collect, clean, label, and ethically justify.
3. Controlled stylized images reduce background noise and keep the training signal focused.
4. They make it easier to test whether the multimodal pipeline works technically.
5. They support a stable classroom/demo environment.

### The most honest way to defend this in viva

You should not pretend these are natural images.

The best answer is:

"We used controlled stylized face-like images as a bootstrap visual dataset to create a stable and balanced presentation-time multimodal benchmark. The goal was to validate the pipeline architecture and product flow clearly. We treat this as a controlled demo dataset, not as a claim of real-world visual generalization."

### Why these images were still useful

Even though they are simple, they still serve an engineering purpose:

- they create modality variation
- they let the image branch produce non-identical tensors
- they allow the CNN path, projection path, and fusion path to be tested end to end
- they support label-aligned multimodal examples

### Very important implementation detail

The dataset script itself says:

- it is a small synthetic multimodal dataset
- it is for presentation-time fine-tuning
- it is not a production dataset
- it is meant to make demo behavior stable

### What Member 2 should say if challenged

"The visual training data was intentionally simplified because the main objective of this project was to demonstrate a full multimodal architecture, not to claim state-of-the-art real-world facial-affect recognition from uncontrolled photographs."

---

## 7. Member 3 Study Pack: Core Multimodal Brain

This is the most important technical part of the project and should be your strongest area.

### How Member 3 should sound overall

You should sound like the person who understands the full system behavior from input alignment to final response.

Best tone:

- "Both branches first work separately."
- "Then we align them in a shared space."
- "Then we fuse them and make the final predictions."
- "After that, we add confidence, calibration, and safety logic."

## 7.1 Why multimodal learning was necessary

Emotion and intent are often distributed across both text and image.

If we use only text:

- we may miss facial or scene context

If we use only image:

- we may miss explicit emotional wording
- we may miss help-seeking language
- we may miss situational context

So the multimodal design is justified because the two branches complement each other.

Natural speaking version:

"The main idea is simple: text and image each tell part of the story, so using both gives us a more complete understanding than using only one."

## 7.2 Shared latent space

Before combining modalities, text and image must be aligned.

That is why both projected representations are mapped to the same size:

- text -> `128`
- image -> `128`

### Why shared projection is important

- makes fusion mathematically manageable
- avoids mixing incomparable raw feature spaces
- reduces computational cost
- creates a structured common representation

Spoken version:

"Before combining them, we need both modalities in a comparable form. That is why we project them into the same latent dimension."

## 7.3 Fusion logic

The project does not use simple concatenation alone.

It uses a fusion module that combines:

- attention
- gating

### Why not simple concatenation?

Concatenation is static.

It joins vectors, but it does not explicitly decide:

- which modality matters more in this case
- how much information should pass from each modality

### Attention in this project

Attention gives relative weighting across text and image.

In simple terms, it helps answer:

"For this input, which modality should matter more?"

### Gating in this project

Gating gives multiplicative control over information flow.

That means even if a branch exists, the model can decide to reduce or increase how much it influences the final fused vector.

### Implementation-level flow

1. Concatenate text and image projected vectors.
2. Produce 2 attention logits through a linear layer.
3. Apply softmax to get attention weights.
4. Produce 2 gate values through another linear layer.
5. Apply sigmoid to keep gate values between 0 and 1.
6. Multiply attention and gating contributions for each modality.
7. Form the fused vector as weighted text plus weighted image.

### Mathematical idea

If projected text and image are:

`z_t` and `z_i`

attention weights are:

`alpha = softmax([s_t, s_i])`

gate values are:

`g = sigma(W_g [z_t ; z_i] + b_g)`

the fused feature is a weighted combination of both branches.

### Why both attention and gating were used together

Attention gives relative importance.  
Gating gives information filtering.  
Together they make fusion more flexible than a single static merge.

Spoken version:

"Attention tells us which modality matters more, and gating tells us how much information from that modality should actually pass through. Using both gives us a more adaptive fusion step."

## 7.4 Prediction heads

The fused representation is passed into separate output heads.

### Why separate heads were used

The project predicts more than one thing, and each output answers a different question.

Spoken version:

"We kept separate heads because emotion, intent, and intensity are related, but they are not exactly the same problem."

### Emotion head

This predicts the emotional class.

Classes:

- happy
- sad
- angry
- neutral
- anxious
- hopeful

### Intent head

This predicts the communicative context.

Classes:

- distress
- celebration
- frustration
- neutral
- seeking_help
- gratitude

### Intensity head

This predicts the strength of the affect on a bounded scale from 0 to 1.

### Why intensity is regression

Intensity is naturally continuous, so it is better modeled as a scalar prediction rather than a rigid discrete class.

### Small architecture details

Emotion and intensity share a small internal trunk:

- linear layer
- ReLU
- dropout

Intent is predicted through a separate branch because communicative purpose is related to, but not identical to, emotional state.

## 7.5 Confidence logic

Confidence is computed from the certainty of the emotion and intent predictions.

### Exact idea

Take:

- maximum emotion probability
- maximum intent probability

Multiply them:

`confidence = max(p_emotion) * max(p_intent)`

### Why this makes sense

The system should only appear highly confident when both classification branches are confident.

## 7.6 Calibration layer

The project includes a practical text-based calibration step.

This is a strong detail to mention because it shows engineering judgment.

### What the calibration does

It checks the cleaned raw text for strong keywords and phrases associated with:

- celebration
- distress
- frustration
- neutral
- seeking help
- gratitude

It also checks useful phrases such as:

- `what should i do`
- `need help`
- `can you help`
- `do next`

### What happens when it triggers

If a strong keyword signal is found, the system can override the raw model prediction with a more reliable intent, emotion, and intensity estimate.

It also raises confidence to at least `0.82`.

### Why this was added

In small project datasets, purely learned outputs can be unstable on obvious high-signal captions.

So calibration improves:

- practical robustness
- demo reliability
- handling of obvious emotionally loaded phrases

Spoken version:

"We added this calibration layer as a practical correction step, especially for captions that contain very strong and obvious emotional language."

## 7.7 Response engine and safety

The project does not stop at classification.

It produces a supportive response, but it does so carefully.

### Why a response engine was needed

If the system predicts emotion and intent but gives no usable output, it feels incomplete as a product.

### Why the response engine is hybrid

The response engine combines:

- rule-based behavior for distress-sensitive cases
- deterministic supportive templates
- optional generative behavior using FLAN-T5

This hybrid design is safer than unconstrained generation everywhere.

Spoken version:

"We did not want the system to generate everything freely, especially in sensitive cases. So we used a hybrid design, where higher-risk cases are handled more carefully."

### Distress rule

The system uses a distress-first rule path when:

- intent is `distress`
- or emotion is `sad` and intensity is at least `0.7`

### Why this matters

High-risk emotional cases should be handled predictably and carefully.

### Optional generative path

An optional FLAN-T5 path exists and can be enabled through environment variables.

But by default the project keeps this path disabled for deterministic offline demos.

That is a good design choice because:

- outputs stay stable during presentations
- the demo remains offline-friendly
- the system avoids unnecessary variability

### Uncertainty handling

If the model confidence is low, the response starts with:

`I may be wrong, but ...`

This is a very useful safety detail because the system does not overclaim certainty.

### Ethics note

Each final response includes a note that the output is AI-based and not a medical diagnosis.

This is another strong point in viva because it shows responsible AI thinking.

## 7.8 API integration and system behavior

As lead presenter, you should know how the model is connected to the application.

### Predict flow

When `/predict` is called:

1. the caption is validated
2. the image type is validated
3. model assets are loaded
4. text is cleaned
5. text is tokenized
6. image bytes are read and validated
7. the image is converted into a tensor
8. the multimodal pipeline runs
9. if a logged-in user exists, the prediction is stored
10. the response payload is returned

### Readiness flow

The `/ready` endpoint checks:

- whether the model is loaded
- whether tokenizer is loaded
- whether weights exist
- whether weights are loaded
- whether the frontend build exists

This is useful in deployment and demo situations.

## 7.9 Storage, history, and trend logic

Even if Member 2 presents this, you should know it clearly.

### Database tables

- `users`
- `sessions`
- `predictions`

### User handling

- usernames are normalized
- passwords are hashed securely
- sessions use generated tokens

### Prediction record stores

- cleaned caption
- emotion
- intent
- intensity
- confidence
- response text
- timestamp

### Trend computation

The system reads recent stored predictions and computes:

- most common emotion
- most common intent
- average intensity
- average confidence
- recent emotion sequence
- recent intent sequence

This gives a small but meaningful analytics layer.

## 7.10 Configuration and deployment-related details

The system uses environment-based settings for flexibility.

### Important configurable elements

- model path
- database path
- device
- warm start
- local-files-only mode
- API key
- CORS origins
- rate limit settings
- text length limit
- image size limit

### Why these settings matter

They make the project more realistic and deployment-aware.

For example:

- warm start loads model assets early
- local-files-only mode supports offline behavior
- rate limits protect the prediction endpoint
- readiness checks make debugging easier

## 7.11 Evaluation

This is where you prove the model is not just built, but assessed.

### Main result numbers

| Model | Emotion Acc | Intent Acc | Intensity MAE |
|---|---:|---:|---:|
| Majority baseline | 0.1667 | 0.1667 | 0.1975 |
| Heuristic baseline | 0.8000 | 0.8000 | 0.1623 |
| CAMEO | 0.8500 | 0.8500 | 0.1550 |

### What this means

- CAMEO beats the majority baseline
- CAMEO beats the text-only heuristic-style baseline
- intensity error is also improved

### Safety result

- response safety pass rate = `100%`

### Why this matters in viva

You can say:

"We did not stop at building the model. We also compared it against weaker baselines and evaluated response safety behavior explicitly."

More natural speaking version:

"We did not just train a model and stop there. We compared it with simpler baselines and also checked whether the final response behavior was safe."

## 7.12 Limitations

This section is important because honest limitations increase credibility.

### Data limitation

The dataset is synthetic and presentation-oriented.

### Generalization limitation

Realistic unseen cases are harder than controlled holdout cases.

### Calibration limitation

Strong negative wording can over-influence interpretation.

### Explainability limitation

Attention and gating are useful internal signals, but they are not perfect human explanations.

### Product limitation

The current storage and deployment style is suitable for demo-scale use, not enterprise production.

## 7.13 Future improvements

- better real-world dataset
- stronger multimodal fine-tuning
- better calibration
- broader safety testing
- stronger deployment infrastructure

## 7.14 Training data, testing data, and why they were used

As lead presenter, you should know the full data story very clearly.

### Main training dataset

The main training dataset comes from:

- [data/presentation_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_manifest.csv)

This dataset is generated by:

- [scripts/generate_presentation_data.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/generate_presentation_data.py)

### What the generated dataset contains

The script creates:

- 6 emotion-intent categories
- 40 examples per category
- a total of 240 multimodal rows

Each row contains:

- text
- image path
- emotion label
- intensity value
- intent label

### Why this generated dataset was used

Because the project needed:

- balanced class representation
- easy-to-explain multimodal examples
- paired text-image-label structure
- stable CPU-friendly experimentation
- reliable demo behavior

### What categories were used

- `happy` + `celebration`
- `sad` + `distress`
- `angry` + `frustration`
- `neutral` + `neutral`
- `anxious` + `seeking_help`
- `hopeful` + `gratitude`

### How intensity values were assigned

The script does not use a single fixed intensity.

Instead, it samples intensity ranges by class:

- happy: lower to medium range
- neutral: medium-low range
- negative/help-seeking classes: medium-high to high range

This was done so the regression head learns variation rather than a single constant per class.

### Train/eval split source

The balanced evaluation split is created by:

- [scripts/create_eval_split.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/create_eval_split.py)

That script groups rows by class and creates:

- [data/presentation_train_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_train_manifest.csv)
- [data/presentation_eval_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_eval_manifest.csv)

### How the split works

It groups by `(emotion, intent)` and then, with a fixed seed:

- takes a small number per class for evaluation
- keeps the rest for training

This keeps the holdout set balanced across categories.

### Why this split strategy was used

Because in a small synthetic dataset, random splitting could create imbalance.

Balanced per-class splitting gives:

- fairer evaluation
- more stable comparison
- easier explanation during viva

### Realistic testing dataset

Beyond the balanced holdout split, the project also uses:

- [data/realistic_eval_cases.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/realistic_eval_cases.csv)

This file contains hand-written unseen cases with:

- text
- image path
- expected emotion
- expected intent
- notes

### Why realistic cases were added

Because only testing on neat synthetic holdout data would be too easy and not fully honest.

The realistic cases were added to check:

- whether the model generalizes beyond templated training captions
- whether negative wording causes over-prediction of distress
- whether the pipeline still behaves sensibly on less controlled examples

### Response safety testing dataset

The project also uses:

- [data/response_safety_cases.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/response_safety_cases.csv)

This dataset is different from training data.

It is designed specifically to test whether the response engine:

- chooses the correct mode
- contains required safe language
- avoids forbidden unsafe language

### Why three different data uses exist

The project really uses data in three roles:

1. training/fine-tuning data
2. prediction-quality evaluation data
3. response-safety validation data

That is a strong point to mention because it shows the project was not assessed in only one way.

## 7.15 Best viva explanation for the emoji-like images

If someone asks, "Why did you use emoji-like images instead of real photos?" use this answer:

"We used controlled face-like synthetic images as a bootstrap visual dataset because our goal was to validate the full multimodal architecture in a stable, explainable, and balanced setting. Real emotional-image collection would require more complex data sourcing, labeling, and ethical handling. The stylized images gave us a clean visual signal per class, allowed consistent multimodal pairing, and made the demo and evaluation pipeline reproducible. We clearly treat this as a controlled project dataset, not as a claim of full real-world affect recognition."

## 7.16 What this means as a strength and a limitation

### Strength

- easy to control
- balanced classes
- reproducible
- easier to debug
- good for proving architecture and pipeline flow

### Limitation

- visually much simpler than real-world emotional imagery
- weaker claim of generalization
- performance on realistic data is naturally harder

### Best honest conclusion

"The dataset choice was suitable for a demonstration-focused multimodal systems project, but future work should use more realistic, diverse, and carefully labeled real-world data."

## 7.17 Training logic, loss functions, and why they were chosen

As lead presenter, you should know not only that the model was trained, but how it was trained.

### Main training script

The standard training path is:

- [scripts/train_multimodal.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/train_multimodal.py)

### What it trains

By default, the project does not fully fine-tune the whole pretrained transformer and CNN backbone.

Instead, it mainly trains:

- text projection layer
- image projection layer
- fusion block
- emotion/intensity head
- intent head

### Why that design was used

Because for a college-scale project:

- full end-to-end fine-tuning is heavier
- it requires more compute and time
- it can become unstable on small synthetic data

So freezing the large pretrained encoders and training the lighter downstream layers is a practical and stable choice.

### Loss functions used

The model is multi-task, so training combines three losses.

#### Emotion loss

Emotion is a classification task, so it uses:

- cross-entropy loss

#### Intent loss

Intent is also a classification task, so it uses:

- cross-entropy loss

#### Intensity loss

Intensity is a continuous value, so it uses:

- mean squared error loss

### Combined training objective

The total loss is:

`loss = loss_emo + loss_intent + lambda_intensity * loss_intensity`

where:

- `loss_emo` = cross-entropy for emotion
- `loss_intent` = cross-entropy for intent
- `loss_intensity` = MSE for intensity
- `lambda_intensity` controls the weight of the intensity regression part

### Why intensity is weighted

If all losses are added blindly, one task may dominate the others.

The scaling factor helps balance the regression objective against the two classification objectives.

### Optimization details

The training scripts use:

- `AdamW`
- weight decay
- optional mixed precision on CUDA

### Reproducibility details

The training scripts set random seeds for:

- Python random
- NumPy
- PyTorch
- CUDA if available

This improves repeatability of the results.

## 7.18 Why cached-head training was used

There is a second training path:

- [scripts/train_cached_heads.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/train_cached_heads.py)

### What it does

Instead of repeatedly recomputing frozen encoder outputs during every epoch, it:

1. runs the frozen text and image encoders once
2. caches pooled features
3. trains projection, fusion, and heads on those cached features

### Why this was useful

This path is especially useful for:

- CPU-based experimentation
- faster evaluation preparation
- repeated training of lightweight downstream layers

### Honest viva explanation

"We added a cached-feature training path to speed up experiments when the large pretrained encoders were frozen. That made CPU evaluation runs more practical."

## 7.19 How evaluation metrics were calculated

This is one of the most important additions for viva.

The evaluation logic is implemented in:

- [scripts/evaluate_models.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/evaluate_models.py)

### Emotion accuracy

Emotion accuracy is calculated as:

`number of correct emotion predictions / total emotion predictions`

In code logic:

- compare predicted emotion label and true emotion label
- count matches
- divide by total number of samples

### Intent accuracy

Intent accuracy is calculated the same way:

`number of correct intent predictions / total intent predictions`

### Intensity MAE

Intensity uses regression, so the project reports:

- Mean Absolute Error (MAE)

Formula:

`MAE = average of |true_intensity - predicted_intensity|`

### Why MAE was used for intensity

Because intensity is continuous.

MAE is easy to interpret: it tells us the average absolute distance between prediction and ground truth.

### Macro-F1

The evaluation script also calculates macro-F1 for classification tasks.

### What macro-F1 means

For each class:

- calculate precision
- calculate recall
- calculate class-wise F1

Then average those F1 values equally across classes.

### Why macro-F1 is useful

Accuracy alone can hide imbalance effects.

Macro-F1 is useful because it gives equal importance to every class rather than letting big classes dominate the score.

### The exact summary metrics reported

For each evaluated system, the script reports:

- emotion accuracy
- emotion macro-F1
- intent accuracy
- intent macro-F1
- intensity MAE
- sample count

## 7.20 What models were compared during evaluation

The project does not evaluate only the final model.

It compares three systems:

1. majority baseline
2. heuristic baseline
3. CAMEO model

### Majority baseline

This baseline predicts:

- the most common emotion from training data
- the most common intent from training data
- the mean training intensity

### Why majority baseline was included

Because every model should be compared against a very weak baseline to prove it learns something non-trivial.

### Heuristic baseline

This baseline uses the calibration logic directly on text, without the full multimodal neural pipeline.

If no keyword signal appears, it falls back to:

- neutral emotion
- neutral intent
- intensity `0.4`

### Why heuristic baseline was included

Because this project has strong text cues in some cases.

So it is useful to compare the full multimodal model against a text-driven rule-style baseline, not only against a majority guesser.

### CAMEO model

This is the learned multimodal system using:

- text encoder
- image encoder
- shared projection
- attention-plus-gating fusion
- prediction heads
- calibration
- response engine

## 7.21 How prediction labels are produced during evaluation

During evaluation:

- text is cleaned
- text is tokenized
- image is loaded and preprocessed
- the model produces emotion probabilities, intent probabilities, and intensity
- the top emotion class is chosen using argmax
- the top intent class is chosen using argmax
- intensity is taken as the scalar regression output

The predicted label indices are then compared with the manifest ground-truth labels.

## 7.22 How realistic-case results were calculated

The realistic-case evaluation is implemented in:

- [scripts/evaluate_realistic_cases.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/evaluate_realistic_cases.py)

### What it measures

For each unseen hand-written case, it checks:

- whether predicted emotion matches expected emotion
- whether predicted intent matches expected intent

Then it reports:

- emotion accuracy
- intent accuracy
- joint accuracy

### What joint accuracy means

Joint accuracy means:

- both emotion and intent must be correct at the same time

This is stricter than checking them separately.

## 7.23 How response safety pass rate was calculated

The safety evaluation is implemented in:

- [scripts/run_response_safety_checks.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/run_response_safety_checks.py)

### What each safety case checks

For each row in the safety cases file, the script checks:

- whether the output mode matches expected mode
- whether all required phrases are present
- whether forbidden phrases are absent

### When a case passes

A safety case passes only if:

- mode is correct
- required content is present
- forbidden content is absent

### Pass rate calculation

`pass_rate = passed_cases / total_cases`

### Why this metric matters

Because for a safety-sensitive responder, classification quality alone is not enough.  
The actual final text behavior also needs evaluation.

## 7.24 How the main artifact files were produced

You should know where the output files came from.

### [artifacts/cameo.pt](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/cameo.pt)

Produced by the main training path and stores trained model weights.

### [artifacts/cameo_cached_eval.pt](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/cameo_cached_eval.pt)

Produced by the cached-head training path and used for evaluation-oriented runs.

### [artifacts/eval_metrics_clean.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/eval_metrics_clean.json)

Produced by the model evaluation script and stores:

- dataset info
- majority baseline metrics
- heuristic baseline metrics
- CAMEO model metrics

### [artifacts/realistic_eval_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/realistic_eval_results.json)

Produced by the realistic-case evaluation script and stores:

- per-case predictions
- correctness flags
- summary accuracies

### [artifacts/response_safety_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/response_safety_results.json)

Produced by the response safety script and stores:

- per-case response checks
- summary pass count
- pass rate

## 7.25 Why two tests are skipped by default

In the current test suite, two tests are skipped unless:

- `CAMEO_RUN_INTEGRATION=1`

This behavior is defined in:

- [tests/test_pipeline_shapes.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_pipeline_shapes.py)

### Why they are skipped by default

Because they are heavier integration-style tests that may:

- require model assets
- take longer to run
- be unnecessary for every quick local test cycle

### Why this is acceptable

It allows:

- fast default testing for development
- optional deeper testing when needed

This is a practical engineering choice, not a missing-feature problem.

## 7.26 Deployment and runtime choices you should be able to justify

### Why `CAMEO_WARM_START=1` was used

To load model assets at startup rather than paying the load cost on the first live request.

This improves demo smoothness.

### Why `CAMEO_REQUIRE_WEIGHTS=1` was used

To fail closed if trained weights are missing.

That avoids silently running an untrained or incomplete system in front of judges.

### Why `CAMEO_LOCAL_FILES_ONLY=1` was used

To avoid runtime dependency on network downloads and keep the demo stable and offline-friendly.

### Why `/health` and `/ready` are both present

Because they answer different questions:

- `/health` asks: is the API process alive?
- `/ready` asks: is the system actually ready to serve prediction traffic?

## 7.27 Best concise answer if asked "How did you calculate accuracy?"

Use this:

"For emotion and intent, accuracy was calculated as the number of correct predictions divided by total evaluated samples. For intensity, since it is a regression output, we used mean absolute error, which averages the absolute difference between predicted and true intensity values. We also reported macro-F1 for the classification tasks so that each class contributes equally to the evaluation."

## 7.28 Small design choices and how to justify them

These are the small constants and hyperparameters that panel members sometimes ask about.

### Why was the shared feature dimension set to 128?

The projected text and image features are both mapped to `128` dimensions.

This is a practical middle ground because:

- it is large enough to preserve useful information
- it is much smaller than raw pretrained encoder feature sizes
- it reduces computation
- it makes fusion and downstream heads lighter

So `128` was chosen as a compact but expressive shared latent size for a project-scale multimodal model.

### Why was text max length set to 128?

The tokenization path uses `max_length = 128`.

This was chosen because:

- project captions are short to medium length
- longer sequence lengths increase compute and memory
- most emotional or intent-carrying information in this dataset appears early enough

So `128` is a practical tradeoff between retaining enough text context and keeping inference efficient.

### Why were images resized to 224 x 224?

This size is standard for many pretrained CNN backbones such as ResNet.

It was chosen because:

- the pretrained model expects this style of input size
- it keeps the image pipeline simple and stable
- it reduces computational load compared with larger images

### Why were there 6 emotion classes and 6 intent classes?

The label spaces were designed to cover the major communicative patterns needed for the project demo.

Emotion classes:

- happy
- sad
- angry
- neutral
- anxious
- hopeful

Intent classes:

- distress
- celebration
- frustration
- neutral
- seeking_help
- gratitude

Why this set was chosen:

- it covers positive, negative, and neutral affect
- it separates inner feeling from outward communicative purpose
- it includes safety-relevant classes like distress and seeking help
- it is broad enough to be meaningful but still manageable for a small project dataset

### Why was dropout set to 0.2?

The prediction heads use `Dropout(0.2)`.

This was chosen as a moderate regularization setting:

- strong enough to reduce overfitting risk
- not so strong that the small downstream heads lose too much signal

For a compact project model, `0.2` is a common and reasonable stabilization choice.

### Why was the uncertainty threshold set around confidence 0.45?

The response engine adds the phrase:

`I may be wrong, but ...`

when confidence is below `0.45`.

Why this is reasonable:

- it avoids overclaiming certainty in low-confidence cases
- it still allows the system to respond supportively
- it creates a practical boundary between stronger and weaker predictions

This should be explained as an engineering threshold rather than a theoretically perfect constant.

### Why was the distress threshold set to 0.7?

The response engine uses:

- `DISTRESS_THRESHOLD = 0.7`

for the sadness-plus-intensity rule.

Why this was chosen:

- values above `0.7` represent clearly elevated negative intensity on a 0 to 1 scale
- it gives a conservative rule trigger for emotionally sensitive outputs
- it reduces the chance of casual mild sadness being treated as a higher-risk distress case

### Why was max text length in the API set to 1200 characters?

The API config uses:

- `CAMEO_MAX_TEXT_CHARS = 1200`

Why this is useful:

- prevents extremely large user inputs from stressing the system
- keeps demo requests practical
- still allows reasonably detailed captions

### Why was password minimum length set to 6?

This is a demo-level application, not a production identity platform.

So the rule was kept simple:

- enough to block extremely weak trivial passwords
- simple enough for fast demo account creation

### Why was the random seed set to 7?

The training and split-generation scripts use:

- `seed = 7`

The exact number is not special mathematically.

Its purpose is reproducibility:

- stable data split
- stable sampling
- more repeatable experiments

### Why was the cached training script set to 6 epochs by default?

The cached training path uses a slightly longer default because:

- only the lighter layers are being trained
- cached features make repeated epochs cheaper
- a few extra epochs help lightweight layers settle better on small datasets

## 7.29 Exact dataset sizes and what they mean

From the evaluation artifact:

- training rows = `180`
- evaluation rows = `60`

This comes from the generated 240-row presentation dataset split into:

- 30 train samples per class across 6 class-pairs
- 10 eval samples per class across 6 class-pairs

Why this matters:

- the split is balanced
- each category contributes equally to evaluation
- the accuracy numbers are easier to interpret fairly

## 7.30 Best answers for tiny-but-common viva questions

### Why not use a larger shared dimension than 128?

Because a larger dimension would increase compute and parameter count without guaranteeing better performance on a small project dataset.

### Why not use a smaller shared dimension than 128?

Because too much compression may discard useful multimodal information before fusion.

### Why not use real images from the start?

Because the project prioritized a stable, reproducible, and explainable multimodal demo pipeline first. Real-image expansion is a future improvement.

### Why not fine-tune the full transformer and CNN?

Because for limited data and compute, freezing pretrained encoders and training lighter layers is more practical and stable.

### Why use macro-F1 in addition to accuracy?

Because macro-F1 gives equal importance to every class and is more informative when class-specific behavior matters.

### Why include both health and readiness endpoints?

Because a service can be alive but still not ready to serve real prediction traffic.

### Why include a heuristic baseline if the final model is multimodal?

Because some captions contain strong explicit cues, so a text-based heuristic is a meaningful non-neural comparison point.

## 7.31 Member 3 file ownership and what each file does

These are the core files you should know best.

### [cameo/core/inference/pipeline.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/inference/pipeline.py)

This is the central multimodal orchestration file.

It holds the logic for:

- assembling text encoder, image encoder, fusion, heads, and response engine
- defining label spaces
- running the end-to-end forward pass
- producing attention and gate outputs
- converting logits into final predictions
- computing confidence
- applying text-based calibration
- invoking the response engine
- returning the final prediction object

### [cameo/core/models/fusion.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/fusion.py)

This file contains the attention-plus-gating fusion block.

It holds the logic for:

- concatenating modality features
- generating attention logits
- applying softmax for attention weights
- generating gate values
- applying sigmoid for bounded gates
- building the fused representation

### [cameo/core/models/heads.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/heads.py)

This file contains the output heads.

It holds:

- emotion classification branch
- intensity regression branch
- intent classification branch
- shared trunk logic for emotion and intensity

### [cameo/core/models/response.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/core/models/response.py)

This file contains the response engine and safety logic.

It holds:

- distress threshold definition
- rule-first distress behavior
- optional FLAN-T5 generation path
- deterministic fallback supportive responses
- confidence-aware uncertainty phrasing
- ethics note injection

### [cameo/__init__.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/cameo/__init__.py)

This file exposes the top-level pipeline class and helps package-level access stay clean.

### Training and experimentation files you should know

### [scripts/train_multimodal.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/train_multimodal.py)

This file contains the main lightweight training path.

It handles:

- manifest loading
- dataloader creation
- training loop
- loss computation for emotion, intent, and intensity
- optimizer setup
- saving trained weights

Important training detail:

- by default it trains projection layers, fusion, and heads rather than fully fine-tuning all pretrained encoders

### [scripts/train_cached_heads.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/train_cached_heads.py)

This file provides a faster CPU-friendly training path for evaluation runs by using cached features.

### [scripts/evaluate_models.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/evaluate_models.py)

This file contains evaluation logic for the main model and baselines and writes metrics JSON.

### [scripts/evaluate_realistic_cases.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/evaluate_realistic_cases.py)

This file checks how the model behaves on more realistic hand-written unseen cases.

### [scripts/run_response_safety_checks.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/run_response_safety_checks.py)

This file validates safety behavior of the response engine against curated prompts.

### [scripts/predict_cli.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/predict_cli.py)

This file offers a command-line inference path outside the web UI.

### [scripts/export_onnx.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/export_onnx.py)

This file exports model components for deployment-oriented ONNX usage.

### [scripts/generate_presentation_data.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/generate_presentation_data.py)

This file supports generation of the larger presentation-oriented synthetic dataset.

### [scripts/create_eval_split.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/scripts/create_eval_split.py)

This file helps create the evaluation split structure used during testing and reporting.

### Evaluation and testing files you should know

### [tests/test_core_units.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_core_units.py)

This file tests:

- attention weights form a proper distribution
- gates stay between 0 and 1
- response engine mode behavior
- calibration detects help-like language

### [tests/test_pipeline_shapes.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_pipeline_shapes.py)

This file checks tensor and integration behavior of the pipeline shape flow.

### [tests/test_response_safety.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_response_safety.py)

This file verifies that response outputs satisfy required safety conditions and avoid forbidden content patterns.

### [tests/test_api.py](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/tests/test_api.py)

This file tests:

- health endpoint
- readiness behavior
- valid prediction path
- invalid image rejection
- oversized upload rejection
- missing weights behavior
- API key enforcement
- rate limiting
- register/login/history/trends flow

### Data and artifact files you should know as lead presenter

### Main data files

- [data/presentation_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_manifest.csv)
- [data/presentation_train_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_train_manifest.csv)
- [data/presentation_eval_manifest.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/presentation_eval_manifest.csv)
- [data/realistic_eval_cases.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/realistic_eval_cases.csv)
- [data/response_safety_cases.csv](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/data/response_safety_cases.csv)

### What they are for

- presentation manifests drive training and holdout evaluation
- realistic cases measure generalization beyond neat balanced data
- response safety cases measure whether the responder behaves responsibly

### Main artifact files

- [artifacts/cameo.pt](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/cameo.pt)
- [artifacts/cameo_cached_eval.pt](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/cameo_cached_eval.pt)
- [artifacts/eval_metrics_clean.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/eval_metrics_clean.json)
- [artifacts/realistic_eval_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/realistic_eval_results.json)
- [artifacts/response_safety_results.json](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/artifacts/response_safety_results.json)

### What they prove

- trained weights exist
- evaluation was run
- unseen-case checking was run
- safety checking was run

### Docs and deployment files you should know

- [README.md](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/README.md)
- [CAMEO_IMPLEMENTATION_GUIDE.md](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/CAMEO_IMPLEMENTATION_GUIDE.md)
- [EVALUATION_REPORT.md](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/EVALUATION_REPORT.md)
- [DEMO_PROMPTS.md](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/DEMO_PROMPTS.md)
- [Dockerfile](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/Dockerfile)
- [compose.yaml](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/compose.yaml)
- [requirements.txt](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/requirements.txt)
- [pyproject.toml](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/pyproject.toml)
- [.env.example](/C:/Users/jigna/OneDrive/Documents/Playground/cameo/.env.example)

### Why these matter

They show the project is not just model code. It also includes:

- implementation explanation
- evaluation reporting
- demo planning
- environment configuration
- dependency definition
- containerized deployment path

### What you can claim from the file level

"I understand not only the central multimodal architecture files, but also the training, evaluation, testing, deployment, and supporting artifacts that make the project complete."

---

## 8. Full Lead-Presenter Ownership Notes

This section is written for you to sound like the person who deeply knows the whole project.

### Best speaking style for you

Do not sound defensive or over-rehearsed.

Try to speak in this order:

1. big picture
2. technical design
3. why that design was chosen
4. limitations
5. future improvement

That order sounds confident and mature in viva.

### If asked "What exactly did you do?"

You can answer:

"I handled the core integration of the full system. I worked on aligning the text and image branches into a shared latent space, designing the fusion block with attention and gating, building the prediction logic for emotion, intent, and intensity, adding confidence estimation and text-based calibration, designing the safety-aware response engine, and understanding how the model is exposed through the API, storage, history, and trend analysis. I also prepared the evaluation logic and baseline comparison so the project could be justified technically."

### If asked "What is your strongest contribution?"

You can answer:

"My strongest contribution is the central multimodal reasoning layer. The most important part of the project is not just that it uses text and image, but that it aligns them, fuses them dynamically, predicts multiple affect-related outputs, and then uses those outputs in a safety-aware response system."

### If asked "Did you understand only your part or the full project?"

You can answer:

"I understand the whole project end to end. My main role is the central multimodal intelligence, but I can also explain preprocessing, tokenization, image encoding, API behavior, storage, analytics, evaluation, and deployment-related settings because they all affect the final integrated system."

---

## 9. Quick Viva Questions With Full Answers

### Why multimodal instead of only text?

Because emotional meaning is often distributed across caption and image together. Text may describe the situation explicitly, while image provides facial or scene context. Using both reduces information loss.

### Why not only image?

Images often miss explicit meaning such as help-seeking or gratitude. Text contributes direct semantic intent.

### Why transformer for text?

Because transformers generate contextual embeddings. That means the meaning of a token depends on surrounding words, which is much stronger than older lexical methods for emotion and intent understanding.

### Why ResNet-50 for image?

Because it is a reliable pretrained CNN backbone that is accurate, explainable enough for viva, and practical for project-scale implementation.

### Why project both modalities to 128 dimensions?

Because the original text and image feature spaces are different. Projection aligns them into a shared latent space and keeps fusion computationally manageable.

### Why use both attention and gating?

Attention gives relative importance, while gating controls how much information from each modality should pass forward. Together they provide dynamic fusion behavior.

### Why separate emotion and intent?

Because they answer different questions. Emotion describes what is being felt, while intent describes the communicative context.

### Why use regression for intensity?

Because intensity is naturally continuous and should be represented on a bounded scale from 0 to 1 instead of a hard class.

### Why compute confidence from both emotion and intent?

Because the system should only appear very confident when both classification branches are confident.

### Why use a rule-first response for distress?

Because distress-sensitive outputs should be handled predictably and safely instead of depending only on unconstrained generation.

### Why include login, history, and trends?

Because that turns the project into a usable system rather than only a one-time prediction demo.

---

## 10. Final Revision Advice

Do not try to memorize every line.

Instead, remember the project in layers:

1. input preparation
2. encoding
3. projection
4. fusion
5. prediction
6. confidence and calibration
7. response safety
8. API and storage
9. evaluation

If you can explain those layers smoothly, you will sound like the person who built the project.

### Final speaking tip

Whenever you answer a question, try to end with one sentence that shows design reasoning.

For example:

- "We chose that because it was more stable for a small project dataset."
- "We added that because it made the system safer."
- "We kept it this way because it was a practical tradeoff between quality and simplicity."

That final sentence makes your answer sound more original and thoughtful.

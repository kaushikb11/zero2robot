# P3: Data Engineering at Scale

*A practitioner reading module. No artifact, no exercises, no wall-clock. This one you read, then go read the real repos. It is deliberately cheap to update: the code is theirs, on `main`, un-pinned; the SHAs move, the ideas don't.*

## The job is the messy middle

Nobody tells you this at the start, so here it is: the person who ships a robot policy spends far less time on the network than on the *pile of data the network eats*. The glamorous part (the architecture, the loss) is a weekend. The rest of the quarter is teleop rigs that drift, a dataset format that has to survive a million episodes without melting the filesystem, a quality filter that decides which demonstrations are real, and a co-training mixture whose weights you will tune more than any hyperparameter. This is the messy middle, and it is where practitioners actually live.

You have already met every one of these problems in miniature. Chapter 0.4 wrote your first dataset in the real LeRobot format (two inputs, one canonical writer) because the format is a contract. Chapter 3.7 mixed a 2-dim pusher with a 6-dim bimanual rig and made you feel, by hand, that normalization is per-embodiment and that what crosses the robot gap is *shape*, not *semantics*. It also built a from-scratch MimicGen and *measured* that more valid data beats a better network. That chapter is the production reality reproduced on tiny data. This module is the production reality itself: same problems, five orders of magnitude larger, and this time you read the code that solves them instead of writing it.

## The format that survives a million episodes

Start with the thing you already touched. `LeRobotDataset v3.0` is the format ch0.4's `record.py` writes into, and its whole design is a scar from v2.1. In v2.1 every episode was its own Parquet file and its own MP4. That is fine for a hobby dataset and catastrophic at scale: a million episodes is a million files, and a filesystem asked to `stat` a million files at dataset-init time falls over long before your GPU sees a batch.

v3.0 fixes it by **decoupling storage from the API**. Three pillars:

- **Tabular data**, the low-dimensional, high-frequency signals (state, action, timestamps), lives in **Apache Parquet** under `data/`, memory-mapped or streamed.
- **Visual data** (camera frames) is concatenated and encoded into **MP4** shards under `videos/`, sharded per camera so files stay a practical size.
- **Metadata** (schema, fps, normalization stats, and crucially *episode segmentation*) lives in JSON/JSONL/Parquet under `meta/`.

The move that buys the scale: **many episodes per file**. Tabular rows and video frames from many episodes are concatenated into `data/chunk-*/file-*.parquet` and `videos/.../file-*.mp4`, and episode boundaries are reconstructed *from metadata, not filenames*. `meta/info.json` holds the canonical schema and the path templates; `meta/stats.json` the global normalization statistics; `meta/tasks.jsonl` the natural-language task strings mapped to integer IDs; and `meta/episodes/` holds per-episode records (lengths, tasks, byte/frame offsets) as *chunked Parquet* so even the index scales. The v2.1→v3.0 converter is exactly this aggregation: `episode-0000.parquet, episode-0001.parquet, … → file-0000.parquet`, then rewrite the offsets. Fewer, larger files; faster init; Hub-native streaming so you can train off a dataset you never fully download. (v3.0 ships in `lerobot >= 0.4.0`; on older installs the layout differs, so check the `CODEBASE_VERSION` stamp before you trust a path.)

This is why ch0.4 insisted the browser never write the format itself and instead handed Python a small interchange bundle: the format is a moving target owned by a pinned library, and reimplementing quantile stats and chunk offsets in TypeScript would be a standing bug farm. The lesson generalizes: at scale, the format *is* infrastructure, and you inherit it, you do not reinvent it.

## The corpus everyone shares, and the co-training lesson

Open X-Embodiment is the standard corpus. The 2023 collaboration pooled data from **22 robot embodiments** across 21 institutions into **1M+ trajectories** (roughly 60 constituent datasets, demonstrating 527 skills / 160k+ tasks, and the corpus keeps growing, so treat any single count as a snapshot). The storage standard underneath is **RLDS** (the Reinforcement Learning Data Standard), episodes served through TensorFlow Datasets. It is a different lineage from LeRobot's Parquet-and-MP4, and reconciling the two is real practitioner work; there is no universal robot data format, only the two big ones and a lot of glue.

Its co-equal is **DROID** (Distributed Robot Interaction Dataset, RSS 2024): 76k in-the-wild teleoperation trajectories across 564 scenes and 86 tasks, all collected on one hardware stack (the Franka Panda arm), which makes it the clean single-embodiment counterweight to OXE's 22-embodiment sprawl. It is the other corpus the LeRobot v3.0 format was built to hold, and it ships trained: openpi's `pi0-DROID` and `pi0.5-DROID` checkpoints are fine-tuned on it. Between them, OXE and DROID are the two corpora a practitioner is actually expected to know.

The result that made OXE matter is the co-training result, and it is the ch3.7 thesis at scale. Train RT-1's architecture on the *mixture* instead of its own robot's data and you get **RT-1-X, about +50% success** over the single-robot baseline; scale to the VLM-based **RT-2-X** and you get roughly **3× on emergent skills**, spatial relations it was never explicitly taught. The policy barely changed; the *data* did. That is ch1.2's "the data is the policy" and ch3.7's measured augmentation delta, one corpus larger.

But read it honestly, because the honest version is the useful one: **a mixture beats any single dataset, but not monotonically.** You cannot pour every dataset in at weight 1.0. Some robots have tens of thousands of near-duplicate episodes and will drown the mixture; some tiny diverse datasets carry most of the generalization and need up-weighting. So the mixtures are *hand-tuned sampling weights*, and they are the most honest artifact in this whole subject. Go read them:

- **OpenVLA**: `prismatic/vla/datasets/rlds/oxe/mixtures.py` in `openvla/openvla`. The `OXE_NAMED_MIXTURES` dict is a registry of `(dataset_name, sample_weight)` tuples. In `oxe_magic_soup`: `("bridge_orig", 1.0)`, `("fractal20220817_data", 0.54087122203)`, `("taco_play", 2.0)`. Those are not round numbers: `0.54087122203` is somebody's deliberate down-weight of Google's giant RT-1 set so it does not swamp everything else. Read the whole dict as a table of judgment calls.
- **Octo**: `octo/data/oxe/oxe_dataset_mixes.py` in `octo-models/octo`, which defines `OXE_MAGIC_SOUP`, `OXE_FLEX_ACT_SOUP`, `RT_X_MIX` and friends, with the same `fractal` down-weight and diverse-dataset up-weights (the paper describes doubling "more diverse" datasets and down-weighting repetitive ones; you can see that policy in the numbers).

Diff the two mixtures. Two serious labs, same corpus, *different weights*, because the right mixture is empirical, not derivable, and that disagreement is the actual state of the art. Both registries are 2024-era exemplars, and they remain the most readable hand-tuned mixtures in the open; newer corpora ship their own weights, but few are as legible as these, which is why they are still the ones to read.

## Quality filtering and versioning are the real workflow

Two more things that never appear in a paper's method section but eat practitioner weeks.

**Quality filtering.** Teleop data is full of drift, pauses, fumbles, and failed episodes. The from-scratch version you own is ch3.7's `if ok:` (a MimicGen trajectory joins the pile *only* if the expert still succeeds) and its production twin is `success_rate = num_success / num_attempts`. At OXE scale the gate becomes a pipeline: success labels, idle-frame and "no-op" trimming (the `*_no_noops` dataset variants in the mixtures are literally this), episode-level scoring. The filter is not edge cleanup; it decides what your policy learns.

**Versioning.** The moment a dataset is a shared, mixed, re-weighted asset, "which data trained this checkpoint?" becomes a question you must be able to answer months later. That is why the metadata is first-class: `CODEBASE_VERSION`, the info/stats/episodes triple, dataset repo IDs and Hub revisions. A reproducible robot result is a *pinned mixture at a pinned dataset revision*, not "we trained on OXE."

## Where it's all heading: the data pyramid

The frontier (Physical Intelligence's π0.5, NVIDIA's GR00T N1.5) frames all of this as a **data pyramid**, and it is worth holding as the mental model. Wide base: internet-scale video and vision-language data, cheap and plentiful but not embodiment-specific. Middle: *synthetic* trajectories, including ones generated by video/world models. Peak: expensive, precious, in-house **real-robot teleoperation**, smallest layer, highest embodiment-specificity. You train by sampling batches *across all three layers at once*. Every theme in this module is one face of that pyramid: the format holds the peak, OXE-style mixtures weight across sources, quality filtering guards the peak's value, and co-training is the pyramid working.

## Read the real thing

Un-pinned on purpose. Go to `main` and read:

1. **LeRobotDataset v3.0**: `huggingface/lerobot`, `docs/source/lerobot-dataset-v3.mdx`, then the loader/writer under `src/lerobot/datasets/`. Map `meta/info.json`, `meta/episodes/`, `data/chunk-*` onto the tiny dataset your ch0.4 `record.py` produced. It is byte-shaped the same way.
2. **Open X-Embodiment**: `google-deepmind/open_x_embodiment`. Read a `colabs/` visualization notebook and note the RLDS episode structure; that is the corpus format the mixtures below sample from.
3. **The mixture tables**: `openvla/openvla` → `prismatic/vla/datasets/rlds/oxe/mixtures.py`, and `octo-models/octo` → `octo/data/oxe/oxe_dataset_mixes.py`. Read them side by side as the empirical answer to "what data, in what proportion."

Then reread **ch3.7** (`phase3_advanced/ch3.7_scale_data`). Its `augment` region, its `if ok:` filter, and its per-embodiment min/max are the from-scratch miniature of everything above, the built counterpart you can run on a laptop in minutes. The papers describe code you have already written; this reading track is where you go see it at a million trajectories.

**Honesty caveats.** Every path here is on a moving `main` branch: verify it exists before you cite it; never trust a SHA in this file, there isn't one. The OXE counts (22 embodiments, ~60 datasets, 1M+ trajectories) are a growing snapshot. The RT-1-X (+50%) and RT-2-X (~3×) figures are the OXE paper's headline numbers under its eval protocol, not a promise about your robot. And the mixture weights are somebody's tuned judgment on their corpus. Copy them as a starting point, not as truth.

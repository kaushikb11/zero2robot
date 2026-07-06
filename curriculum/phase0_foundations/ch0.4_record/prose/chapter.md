# 0.4: Teleoperation & Your First Dataset

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work
<!-- The hook. Embedded live browser teleop demo (demo/embed.yaml): drag the
     pusher with your mouse, push the T home, hit record. 1-2 paragraphs: what
     the learner is looking at, what to try with their mouse, and the one
     observable behavior — a real episode being captured as (obs, action) pairs. -->

## The problem
<!-- Chapters 0.1-0.3 built and understood the sim; nothing has LEARNED anything.
     A policy needs demonstrations, and demonstrations are episodes in a specific
     on-disk format. Show the gap: you can drive the robot, but training needs a
     dataset. Ends with: we record episodes and write them as LeRobot v3. -->

## Build
<!-- Region-by-region walkthrough of record.py in dependency order. Each region:
     one orienting sentence -> include-by-region code block -> prose that earns it.
     Subheadings mirror region names: ### Setup, ### Features, ### Teleop,
     ### Ingest, ### Write, ### Run. The spine: two inputs, one canonical writer. -->

## Run it
<!-- The exact commands (local record + interchange ingest). The honest
     wall-clock table (auto-rendered). What your recorded episode looks like in
     rerun on the sim_time timeline, and the first thing to check if it doesn't. -->

## Break it
<!-- One deliberate failure: declare the wrong feature shape and watch parity
     with the training data break (the ch1.1 BC contract). The diagnosis walk. -->

## Exercises
<!-- Rendered from exercises/. One line of framing only. -->

## What's next
<!-- One paragraph. You have a dataset but haven't LOOKED at it — is it any good?
     Sets up 0.5 (inspect the dataset) and the continuity to 1.1 (train on it). -->

# Meeting Minutes – Client Meeting (3rd)

**Project:** VOC Sensor Data Pipeline for Disease Monitoring  
**Date:** 16 March 2026  
**Time:** 11:30 - 12:00  
**Stakeholder:** Eyuel Ayele  
**Participants:**

- Adityanand Singh
- Faezeh Kianimoravej
- Sepideh Qorbani

---

## 1. Purpose of the Meeting
The purpose of this third client meeting was to confirm the expected minimum end-to-end pipeline for the project, focusing on how to extract VOC “events” (spikes/responses) from the sensor signal and then build feature engineering + ML/MLOps around those events.

---

## 2. Project Context
Eyuel explained that when a chemical is introduced, the sensor responses appear as spike-like patterns that can be treated as discrete “events”. The pipeline should start from the stage where the signal is already cleaned (after preprocessing such as low-pass filtering), so the event detection and subsequent feature engineering/modeling are based on meaningful signal structure.

---

## 3. Key Discussion Points
### Meeting logistics / coordination
- Participants confirmed they were present and discussed where they were located (e.g., “basement”).
- Eyuel joined later in the conversation and confirmed screen sharing availability.
- Communication/setup issues (e.g., meeting timing and being on board before the session) were briefly discussed.

### VOC event-based pipeline approach
- VOC spikes can be treated as events (responses detected after preprocessing).
- In raw signals, such spike structure may resemble noise; preprocessing (e.g., low-pass filtering) helps reveal the “clean signal” where events are detectable.
- The team should not over-focus on what happens before the clean-signal stage; start from the cleaned signal for the pipeline.

### Minimum baseline and implementation guidance
- After the baseline processing, the next step is to build the ML/MLOps flow on top of the extracted events.
- Create feature engineering around the detected VOC event(s) so models have learnable inputs.
- Introduce one or more models to demonstrate the idea.
- Perform end-to-end testing and provide a demo (including a deploy/test step if applicable).
- The pipeline does not need to match the demonstrated workflow exactly, but it should meet a minimum baseline.
- Recommended workflow is to start with “version zero” and iterate until the work matches the assignment requirements.

---

## 4. Decisions Made
- No explicit final decisions were captured in the provided transcript segment.

---

## 5. Action Items
| Action | Responsible |
|------|------|
| Start from the cleaned signal (after preprocessing such as low-pass filtering) and then detect VOC spike/event patterns | Student Team |
| Build feature engineering based on the detected VOC events for model training | Student Team |
| Implement an ML/MLOps-style flow: train a couple of models and prepare an end-to-end demo/test | Student Team |
| Start from a “version zero” baseline implementation, then iterate toward the full assignment requirements | Student Team |
| If unsure/blocked, send quick questions to Eyuel via email rather than waiting until the next scheduled client discussion | Student Team |

---

## 6. Open Questions
- None explicitly stated in the provided transcript segment.

---

## 7. Next Steps
1. Validate preprocessing produces a clean signal suitable for VOC event/spike detection.
2. Implement event extraction and feature engineering around those events.
3. Train one or more models and run an end-to-end test/demo.
4. Iterate from the baseline (“version zero”) to align with the assignment requirements.

---

## 8. Notes
- Eyuel emphasized that the goal is a workable minimum baseline and iterative improvement, rather than reproducing the exact example workflow.

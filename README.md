# FlowState

**No patient waits unseen.** FlowState watches an ER waiting room (existing cameras or seat sensors), runs each patient's CTAS re-check timer, and buzzes the nurse's smartwatch when someone is due for a re-check, slumps, stops moving, or leaves.

Built at natHacks 2026 (Healthcare & Accessibility).

## How it works
Sensors / camera -> Arduino Uno -> USB -> laptop script (rules + timers) -> Supabase -> nurse smartwatch (ntfy) + dashboard + buzzer.
The patient page (QR code) talks to Supabase directly.

## Repo layout
| Folder | What's in it | Owner |
| --- | --- | --- |
| `arduino/` | Uno sketch that reads every sensor | Hardware lead |
| `backend/` | Laptop script, alert rules, camera mode, insights | Backend + Data lead |
| `frontend/` | Nurse dashboard, patient page, insights tab | Frontend |
| `docs/` | Research, hardware pin map, calibration data, pitch | Business + everyone |
| `data/exports/` | Anonymized CSV exports (not committed) | Data lead |

## How to run it
TODO (Backend + Lead): fill in on Saturday night.
1. Copy `.env.example` to `.env` and fill in the values.
2. Upload `arduino/flowstate_sensors/flowstate_sensors.ino` to the Uno.
3. `cd backend && pip install -r requirements.txt && python main.py`
4. `cd frontend && npm install && npm run dev`

## Team and credits
TODO

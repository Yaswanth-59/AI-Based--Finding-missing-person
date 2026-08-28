# Finding Missing Person AI — Web Face Matching

Flask web application for registering missing-person cases and comparing an uploaded found-person photo against registered photos.

## Main AI flow
1. Register a missing person and upload a clear face photo.
2. Open **AI Match**.
3. Upload a found-person photo.
4. The system detects the face.
5. Each registered missing-person photo is compared using OpenCV LBPH when available.
6. Results are ranked by similarity and show **Similarity / 100**, **LBPH distance**, and **View Case**.

## Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

Demo login:
- admin / admin123

## Notes
- CPU-only OpenCV Haar face detection is used; MediaPipe is not required.
- LBPH is a demo/decision-support face matching method, not proof of identity.
- Clear, front-facing photos produce better results.

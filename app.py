import os, json, uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from sqlmodel import create_engine, Session, select
from werkzeug.utils import secure_filename
from model.data_models import RegisteredCases, PublicSubmissions
from ai_utils import detect_all_faces
from match_engine import find_matches, compare_uploaded_photo

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, 'sqlite_database.db')
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'missing-person-ai-dev-secret')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit
UPLOAD = os.path.join(BASE, 'static', 'uploads')
os.makedirs(UPLOAD, exist_ok=True)

# Create tables
RegisteredCases.__table__.create(engine, checkfirst=True)
PublicSubmissions.__table__.create(engine, checkfirst=True)


def logged_in():
    return bool(session.get('user'))


def require_login():
    if not logged_in():
        return redirect(url_for('login'))
    return None


def save_upload(file):
    if not file or not file.filename:
        return None
    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    if ext not in {'.jpg','.jpeg','.png','.webp'}:
        return None
    name = f"{uuid.uuid4()}{ext}"
    file.save(os.path.join(UPLOAD, name))
    return name

@app.route('/')
def index():
    if not logged_in(): return redirect(url_for('login'))
    with Session(engine) as db:
        cases = db.exec(select(RegisteredCases)).all()
        found = sum(1 for c in cases if c.status == 'F')
        missing = sum(1 for c in cases if c.status == 'NF')
        public = len(db.exec(select(PublicSubmissions)).all())
        cities = {}
        for c in cases:
            city = c.city or 'Unknown'
            cities[city] = cities.get(city, 0) + 1
    return render_template('index.html', found=found, missing=missing, public=public, total=len(cases), cities=cities)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username','').strip()
        password = request.form.get('password','')
        # Demo login. Change credentials before deployment.
        valid = {'admin':'admin123', 'gagan':'abc'}
        if user in valid and password == valid[user]:
            session['user'] = user
            session['role'] = 'Admin'
            flash('Login successful.', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))

@app.route('/register', methods=['GET','POST'])
def register():
    guard = require_login()
    if guard: return guard
    if request.method == 'POST':
        f = request.files.get('photo')
        filename = save_upload(f)
        if not filename:
            flash('Please upload a JPG, JPEG, PNG or WEBP photo.', 'danger')
            return render_template('register.html')
        try:
            from PIL import Image
            import numpy as np
            image = np.array(Image.open(os.path.join(UPLOAD, filename)).convert('RGB'))
            faces = detect_all_faces(image, max_faces=5)
        except Exception as e:
            os.remove(os.path.join(UPLOAD, filename)); flash(f'Face detection failed: {e}', 'danger'); return render_template('register.html')
        if not faces:
            os.remove(os.path.join(UPLOAD, filename)); flash('No face detected. Use a clear front-facing photo.', 'danger'); return render_template('register.html')
        idx = int(request.form.get('face_index','0'))
        idx = max(0, min(idx, len(faces)-1))
        case_id = str(uuid.uuid4())
        # rename photo to stable case id
        ext = os.path.splitext(filename)[1]
        final_name = case_id + ext
        os.rename(os.path.join(UPLOAD, filename), os.path.join(UPLOAD, final_name))
        case = RegisteredCases(id=case_id, submitted_by=session['user'], name=request.form.get('name','').strip(), father_name=request.form.get('father_name','').strip(), age=request.form.get('age','').strip(), complainant_name=request.form.get('complainant_name','').strip(), complainant_mobile=request.form.get('complainant_mobile','').strip(), complainant_email=request.form.get('complainant_email','').strip() or None, adhaar_card=request.form.get('adhaar_card','').strip(), last_seen=request.form.get('last_seen','').strip(), address=request.form.get('address','').strip(), city=request.form.get('city','').strip() or None, description=request.form.get('description','').strip() or None, face_mesh=json.dumps(faces[idx]['landmarks']), status='NF', birth_marks=request.form.get('birth_marks','').strip(), matched_with='')
        with Session(engine) as db: db.add(case); db.commit()
        flash(f'Case registered successfully. ID: {case_id}', 'success')
        return redirect(url_for('cases'))
    return render_template('register.html')

@app.route('/cases')
def cases():
    guard=require_login()
    if guard: return guard
    q=request.args.get('q','').strip().lower(); status=request.args.get('status','All')
    with Session(engine) as db:
        data=db.exec(select(RegisteredCases).order_by(RegisteredCases.submitted_on.desc())).all()
    if status in ('F','NF'): data=[c for c in data if c.status==status]
    if q: data=[c for c in data if q in (c.name or '').lower() or q in (c.city or '').lower()]
    return render_template('cases.html', cases=data, q=q, status=status)

@app.route('/public', methods=['GET','POST'])
def public_case():
    guard=require_login()
    if guard: return guard
    if request.method=='POST':
        f=request.files.get('photo'); filename=save_upload(f)
        if not filename: flash('Upload a valid image.', 'danger'); return redirect(url_for('public_case'))
        try:
            from PIL import Image
            import numpy as np
            image=np.array(Image.open(os.path.join(UPLOAD,filename)).convert('RGB'))
            faces=detect_all_faces(image, max_faces=1)
        except Exception as e:
            os.remove(os.path.join(UPLOAD,filename)); flash(f'Face detection failed: {e}','danger'); return redirect(url_for('public_case'))
        if not faces:
            os.remove(os.path.join(UPLOAD,filename)); flash('No face detected.','danger'); return redirect(url_for('public_case'))
        pid=str(uuid.uuid4()); ext=os.path.splitext(filename)[1]; os.rename(os.path.join(UPLOAD,filename),os.path.join(UPLOAD,pid+ext))
        row=PublicSubmissions(id=pid, submitted_by=session['user'], face_mesh=json.dumps(faces[0]['landmarks']), location=request.form.get('location','').strip(), mobile=request.form.get('mobile','').strip(), email=request.form.get('email','').strip() or None, status='NF', birth_marks=request.form.get('birth_marks','').strip())
        with Session(engine) as db: db.add(row); db.commit()
        flash(f'Public/found-person case submitted. ID: {pid}', 'success')
        return redirect(url_for('match'))
    return render_template('public.html')

@app.route('/match', methods=['GET','POST'])
def match():
    guard=require_login()
    if guard: return guard
    results=[]
    uploaded_name=None
    if request.method=='POST':
        f=request.files.get('photo')
        if not f or not f.filename:
            flash('Please upload a found-person photo.', 'danger')
            return render_template('match.html', results=results)
        filename=save_upload(f)
        if not filename:
            flash('Please upload a JPG, JPEG, PNG or WEBP photo.', 'danger')
            return render_template('match.html', results=results)
        try:
            from PIL import Image
            import numpy as np
            image=np.array(Image.open(os.path.join(UPLOAD,filename)).convert('RGB'))
            results, error=compare_uploaded_photo(image, engine)
            if error:
                os.remove(os.path.join(UPLOAD,filename))
                flash(error, 'danger')
                return render_template('match.html', results=[])
            uploaded_name=filename
            if not results:
                flash('No registered case could be compared. Make sure registered cases have photos.', 'warning')
        except Exception as e:
            if os.path.exists(os.path.join(UPLOAD,filename)):
                os.remove(os.path.join(UPLOAD,filename))
            flash(f'AI matching failed: {e}', 'danger')
    return render_template('match.html', results=results, uploaded_name=uploaded_name)

@app.route('/case/<case_id>/photo')
def case_photo(case_id):
    guard=require_login()
    if guard: return guard
    for ext in ('.jpg','.jpeg','.png','.webp'):
        path=os.path.join(UPLOAD, case_id + ext)
        if os.path.exists(path):
            return send_from_directory(UPLOAD, case_id + ext)
    return ('',404)

@app.route('/case/<case_id>')
def case_detail(case_id):
    guard=require_login()
    if guard: return guard
    with Session(engine) as db: case=db.get(RegisteredCases, case_id)
    if not case: flash('Case not found.','danger'); return redirect(url_for('cases'))
    return render_template('case_detail.html', case=case)

@app.route('/case/<case_id>/found/<public_id>', methods=['POST'])
def confirm_match(case_id, public_id):
    guard=require_login()
    if guard: return guard
    with Session(engine) as db:
        case=db.get(RegisteredCases,case_id); public=db.get(PublicSubmissions,public_id)
        if case and public:
            case.status='F'; case.matched_with=public_id; public.status='F'; db.add(case); db.add(public); db.commit(); flash('Match confirmed. Case marked Found.','success')
    return redirect(url_for('case_detail', case_id=case_id))

@app.route('/uploads/<name>')
def uploads(name): return send_from_directory(UPLOAD,name)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

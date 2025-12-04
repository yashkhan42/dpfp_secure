# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import hashlib
import numpy as np
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = '9f92d2931fcc409def681feab2f7b38c'  # CHANGE IN PRODUCTION
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'secure_data'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


SECRET_KEY_DPF = 'mysecretkey123'  


DATA_PATH = os.path.join(app.config['DATA_FOLDER'], 'final_update_2.parquet')
df = pd.read_parquet(DATA_PATH)


HASH_COLUMN = '__fp_hash__'
L = 64
xi = 3

def _row_to_hash(row: pd.Series) -> int:
    
    key = ''.join(str(v).replace(' ', '') for v in row if pd.notna(v))
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)

def generate_fingerprinted_dataset(original_df, userid, secret_key,
                                   epsilon=1.0, xi=xi, L=L):
    df = original_df.copy()

    
    df[HASH_COLUMN] = df.apply(_row_to_hash, axis=1).astype('uint64')

    p = 1/(np.exp(epsilon/xi)+1)
    select_prob = 2*p
    m = max(1, int(1/select_prob))

    internal_id = hashlib.sha256((str(secret_key) + str(userid)).encode()).hexdigest()
    f_hash = hashlib.sha256((str(secret_key) + internal_id).encode()).digest()
    f_bits = ''.join(format(byte, '08b') for byte in f_hash)[:L]
    f = [int(bit) for bit in f_bits]

    for i in df.index:
        val = int(df.at[i, HASH_COLUMN])
        new_val = val
        for k in range(xi):
            seed = f"{secret_key}{i}{k}"
            h1 = hashlib.sha256((seed + 'U1').encode()).hexdigest()
            u1 = int(h1, 16)
            if u1 % m != 0:
                continue
            h2 = hashlib.sha256((seed + 'U2').encode()).hexdigest()
            x = int(h2, 16) % 2
            h3 = hashlib.sha256((seed + 'U3').encode()).hexdigest()
            l = int(h3, 16) % L
            B = x ^ f[l]
            new_val ^= (B << k)
        df.at[i, HASH_COLUMN] = new_val

    return df

def identify_user_from_fingerprint(fingerprinted_df, secret_key, suspect_userids,
                                   epsilon=1.0, xi=xi, L=L,
                                   match_threshold=0.60, debug=False):
    df = fingerprinted_df.copy()

    if HASH_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {HASH_COLUMN}")

    df[HASH_COLUMN] = df[HASH_COLUMN].astype('uint64')

    p = 1 / (np.exp(epsilon / xi) + 1)
    select_prob = 2 * p
    m = max(1, int(1 / select_prob))
    count = [[0, 0] for _ in range(L)]

    for i in df.index:
        val = int(df.at[i, HASH_COLUMN])
        for k in range(xi):
            seed = f"{secret_key}{i}{k}"
            h1 = hashlib.sha256((seed + 'U1').encode()).hexdigest()
            u1 = int(h1, 16)
            if u1 % m != 0:
                continue
            h2 = hashlib.sha256((seed + 'U2').encode()).hexdigest()
            x = int(h2, 16) % 2
            h3 = hashlib.sha256((seed + 'U3').encode()).hexdigest()
            l = int(h3, 16) % L
            b = (val >> k) & 1
            est_bit = b ^ x
            count[l][est_bit] += 1

    extracted_f = [None] * L
    extracted_conf = [0.0] * L
    for l in range(L):
        total = count[l][0] + count[l][1]
        if total == 0:
            continue
        r0, r1 = count[l][0]/total, count[l][1]/total
        extracted_f[l] = 0 if r0 > r1 else 1
        extracted_conf[l] = max(r0, r1)

    best_userid, best_score = None, 0.0
    for userid in suspect_userids:
        internal_id = hashlib.sha256((str(secret_key) + str(userid)).encode()).hexdigest()
        f_hash = hashlib.sha256((str(secret_key) + internal_id).encode()).digest()
        f_bits = ''.join(format(byte, '08b') for byte in f_hash)[:L]
        suspect_f = [int(b) for b in f_bits]

        matches = sum(1 for a, b in zip(extracted_f, suspect_f) if a is not None and a == b)
        total = sum(1 for a in extracted_f if a is not None)
        score = matches / total if total else 0.0

        if debug:
            print(f"[identify] userid={userid} score={score:.3f}")

        if score > best_score:
            best_score, best_userid = score, userid

    if best_score >= match_threshold:
        return best_userid, best_score
    return None, best_score

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(email=email, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            new_password = request.form['new_password']
            user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
            db.session.commit()
            flash('Password reset successful. Please log in.')
            return redirect(url_for('login'))
        flash('Email not found')
    return render_template('reset_password.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/download_dataset')
@login_required
def download_dataset():
    userid = current_user.id
    
    fingerprinted_df = generate_fingerprinted_dataset(df, userid, SECRET_KEY_DPF)

    export_df = fingerprinted_df.copy()

    # Export as .parquet
    output = BytesIO()
    export_df.to_parquet(output, index=False, compression='gzip')
    output.seek(0)

    filename = f"dataset_user_{userid}.parquet"
    
    return send_file(
        output,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=filename
    )

@app.route('/identify', methods=['GET', 'POST'])
def identify():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        file = request.files['file']
        if not file.filename:
            flash('No file selected')
            return redirect(request.url)
        
        if file.filename.endswith(('.csv', '.parquet')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df_upload = pd.read_parquet(filepath) if filename.endswith('.parquet') else pd.read_csv(filepath)
                
                suspect_userids = [u.id for u in User.query.all()]
                identified, score = identify_user_from_fingerprint(
                    df_upload, SECRET_KEY_DPF, suspect_userids,
                    match_threshold=0.60, debug=False)
                
                if identified:
                    user = User.query.get(identified)
                    flash(f'Leaker identified: {user.email} (confidence: {score:.3f})')
                else:
                    flash(f'No match found (best score: {score:.3f})')
            except Exception as e:
                flash(f'Error: {str(e)}')
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            return redirect(url_for('identify'))
    
    return render_template('identify.html')

@app.route('/admin/upload', methods=['GET', 'POST'])
@login_required
def admin_upload():
    if not current_user.is_admin:
        flash('Admin access required')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        file = request.files['file']
        if not file.filename:
            flash('No file selected')
            return redirect(request.url)
        
        if file.filename.endswith(('.csv', '.parquet')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df_upload = pd.read_parquet(filepath) if filename.endswith('.parquet') else pd.read_csv(filepath)
                
                suspect_userids = [u.id for u in User.query.all()]
                identified, score = identify_user_from_fingerprint(
                    df_upload, SECRET_KEY_DPF, suspect_userids,
                    match_threshold=0.55, debug=True)
                
                if identified:
                    user = User.query.get(identified)
                    flash(f'[ADMIN] Leaker: {user.email} (score: {score:.3f})')
                else:
                    flash(f'[ADMIN] No match (best: {score:.3f})')
            except Exception as e:
                flash(f'Error: {str(e)}')
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            return redirect(url_for('admin_upload'))
    
    return render_template('admin_upload.html')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        admin = User.query.filter_by(email='admin@example.com').first()
        if not admin:
            hashed = generate_password_hash('adminpass', method='pbkdf2:sha256')
            admin = User(email='admin@example.com', password=hashed, is_admin=True)
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True)
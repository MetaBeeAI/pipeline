# MetaBeeAI webapp instructions

This document contains both the development instructions for local setup and the deployment instructions for running the application on a webserver.

m.mieskolainen@imperial.ac.uk, 2025

---

## 1. Development Instructions

### Backend (FastAPI)

Install the required Python packages:
```bash
pip install -r backend-requirements.txt
```

Start the FastAPI server in the root folder:
```bash
uvicorn main:app --reload --port 8000
```

### Frontend (React)

Navigate to the frontend directory:

```bash
cd metabeeai-frontend
```

Install dependencies and start the React development server
```bash
npm install
npm run dev
```

### Troubleshooting on Ubuntu

#### Fixing Node.js Version Issues:
1. Remove the old version of Node.js:
```bash
sudo apt remove nodejs
```
2. Install Node.js 18 LTS using NodeSource:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs (--force)
```
3. Verify the installation:
```bash
node -v
```

---

## 2. Deployment Instructions on a Webserver

### 1. Environment Setup
- Ensure Python is installed on your server.
- Transfer your project files to the server.
- Install the required packages on the server:
```bash
cd MetaBeeAI_LLM
source venv/bin/activate
pip install -r backend-requirements.txt
```

### 2. Running FastAPI backend and React Frontend in Production
 
Make sure you have `nginx` and `uvicorn` installed.

#### A. Build the React frontend application (re-build always when modifying)
```
cd metabeeai-frontend
npm install
npm run build
```

#### B. Modify nginx setup (see Step 3 below) and make a symbolic link
```
sudo nano /etc/nginx/sites-available/fastapi
sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled/
```
And remove `default` from nginx folders above, if needed.

#### C. Run uvicorn backend server on background
```
nohup uvicorn main:app --host 127.0.0.1 --port 8000 &
```

#### D. Start nginx frontend server
```
sudo systemctl start nginx
sudo systemctl stop nginx
```

#### E. Access and error logs
```
cat /var/log/nginx/access.log
cat /var/log/nginx/error.log
cat nohop.out
```

#### F. URL base setup

Make sure in the file`src/pages/index.tsx`, the constant `API_BASE_URL` is set as
```
// const API_BASE_URL = "http://localhost:8000"
const API_BASE_URL = "/api";
```

## 3. **Reverse Proxy Setup:**
   
Create a file `sudo nano /etc/nginx/sites-available/fastapi`, like the one below

```
server {
   listen 80;
   server_name _;

   # Serve static files from your React build directory
   root /home/ubuntu/MetaBeeAI_LLM/metabeeai-frontend/out;
   index index.html;

   # Proxy API requests to FastAPI
   location /api/ {
      proxy_pass http://127.0.0.1:8000/;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
   }

   # For all other requests, try to serve the static file
   # If not found, fall back to index.html (React Router)
   location / {
      try_files $uri /index.html;
   }
}
```

### Additional Considerations

- **Security:**  
  Ensure that proper security measures are in place (e.g., SSL/TLS certificates, firewall rules).

- **Monitoring & Logging:**  
  Set up monitoring and logging to keep track of the application's performance and errors.

- **Environment Variables:**  
  Configure environment-specific settings (such as database URLs, API keys) using environment variables or configuration files.

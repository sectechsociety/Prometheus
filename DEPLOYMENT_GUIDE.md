# Vercel + Railway Deployment Guide

## Overview
- **Frontend (Vercel)**: Free static hosting, no credit card required initially
- **Backend (Railway)**: Free $5/month credit, no card needed for trial
- **Total Setup Time**: 10-15 minutes

---

## Part 1: Deploy Backend to Railway (Do This First!)

### Step 1: Create Railway Account
1. Go to https://railway.app
2. Click **"Start a New Project"** or **"Login with GitHub"**
3. Authorize Railway to access your GitHub repositories
4. **No credit card required for signup!**

### Step 2: Create New Project from GitHub
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your repository: **`Tech-Society-SEC/Prometheus`**
4. Railway will detect it's a Python project

### Step 3: Configure Backend Service
1. Railway will show deployment settings
2. Click **"Add variables"** and add these environment variables:

   ```
   PORT=8000
   CHROMA_DATA_PATH=/app/services/ingest/chroma_db
   PYTHONPATH=/app
   ```

3. Under **"Settings"** → **"Root Directory"**, leave blank (Railway will use root)
4. Under **"Settings"** → **"Start Command"**, it should auto-detect or set:
   ```
   cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

### Step 4: Add Persistent Storage for ChromaDB
1. Go to your Railway project dashboard
2. Click on your service
3. Click **"Variables"** tab
4. Click **"New Variable"** → **"Add Volume"**
5. Mount path: `/app/services/ingest/chroma_db`
6. Size: 1 GB (ChromaDB needs ~200MB, this gives room to grow)

### Step 5: Deploy and Get Backend URL
1. Click **"Deploy"** (Railway auto-deploys on push to main)
2. Wait 2-3 minutes for build to complete
3. Go to **"Settings"** → **"Networking"** → **"Generate Domain"**
4. Copy your backend URL (something like: `https://prometheus-production-XXXX.up.railway.app`)
5. Test it: Open `https://your-railway-url.railway.app/health` in browser
   - Should see: `{"status": "ok"}`

**✅ Backend is live! Save this URL - you'll need it for frontend.**

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Create Vercel Account
1. Go to https://vercel.com
2. Click **"Sign Up"** → **"Continue with GitHub"**
3. Authorize Vercel to access your repositories
4. **No credit card required!**

### Step 2: Import Project
1. From Vercel Dashboard, click **"Add New..."** → **"Project"**
2. Find and select **`Tech-Society-SEC/Prometheus`**
3. Click **"Import"**

### Step 3: Configure Build Settings
Vercel should auto-detect Vite, but verify these settings:

1. **Framework Preset**: Vite
2. **Root Directory**: `frontend` (IMPORTANT!)
3. **Build Command**: `npm run build`
4. **Output Directory**: `dist`
5. **Install Command**: `npm install`

### Step 4: Add Environment Variable
This is CRITICAL - connects frontend to your Railway backend:

1. Click **"Environment Variables"**
2. Add this variable:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://your-railway-url.railway.app` (paste your Railway backend URL from Part 1)
   - **Environment**: Production, Preview, Development (select all)
3. Click **"Add"**

### Step 5: Deploy
1. Click **"Deploy"**
2. Wait 1-2 minutes for build
3. Vercel will show deployment progress
4. Once done, you'll see **"Visit"** button
5. Copy your Vercel URL (like: `https://prometheus-XXXX.vercel.app`)

**✅ Frontend is live!**

---

## Part 3: Final Testing

### Test Your Deployed App
1. Open your Vercel URL: `https://prometheus-XXXX.vercel.app`
2. Type a test prompt: "Write a blog post about AI"
3. Click **"Enhance"**
4. You should see 5 enhanced prompts appear!
5. Test **Copy** and **Export** buttons

### Troubleshooting

**If frontend shows "Failed to fetch" error:**
1. Check Railway backend is running: `https://your-railway-url.railway.app/health`
2. Verify `VITE_API_URL` environment variable in Vercel matches Railway URL exactly
3. Redeploy Vercel frontend after fixing env variable

**If backend shows errors:**
1. Check Railway logs: Dashboard → Your Service → "Deployments" → Latest → "View Logs"
2. Common issues:
   - ChromaDB not initialized: Check volume mount is correct
   - Port binding: Ensure `PORT` environment variable is set
   - Module import errors: Check `PYTHONPATH=/app` is set

**If ChromaDB errors appear:**
1. Verify Railway volume is mounted at `/app/services/ingest/chroma_db`
2. Check backend logs for "CHROMA_DATA_PATH" value
3. May need to manually populate ChromaDB (see below)

### Populate ChromaDB (If Needed)
If the vector database is empty, you'll need to run the ingestion script once:

1. In Railway Dashboard, click your service → **"Settings"** → **"Deploy Trigger"**
2. Add a one-time startup command:
   ```bash
   python services/ingest/ingest.py && cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
3. Or run it manually via Railway's CLI (requires installing Railway CLI)

---

## Part 4: Update CORS (Important!)

After both deployments, update backend CORS to allow your Vercel domain:

1. The code is already configured in `backend/app/main.py` to read from environment
2. In Railway Dashboard → Your Service → **"Variables"**
3. Add new variable:
   - **Key**: `FRONTEND_URL`
   - **Value**: `https://prometheus-XXXX.vercel.app` (your actual Vercel URL)
4. Railway will auto-redeploy

---

## Quick Reference

### Your Deployment URLs
- **Frontend (Vercel)**: https://prometheus-XXXX.vercel.app
- **Backend (Railway)**: https://prometheus-production-XXXX.up.railway.app
- **Health Check**: https://prometheus-production-XXXX.up.railway.app/health
- **API Endpoint**: https://prometheus-production-XXXX.up.railway.app/augment

### Environment Variables Summary

**Railway (Backend)**:
```
PORT=8000
CHROMA_DATA_PATH=/app/services/ingest/chroma_db
PYTHONPATH=/app
FRONTEND_URL=https://prometheus-XXXX.vercel.app
```

**Vercel (Frontend)**:
```
VITE_API_URL=https://prometheus-production-XXXX.up.railway.app
```

### Free Tier Limits
- **Vercel**: Unlimited deployments, 100GB bandwidth/month
- **Railway**: $5 free credit/month (~500 hours runtime)

---

## Next Steps After Deployment

1. **Test thoroughly** - Try different prompts, verify all 5 results appear
2. **Add to PPT** - Include live demo URL in presentation slides
3. **Monitor usage** - Check Railway dashboard for resource usage
4. **Share with Dean** - Send the Vercel URL for live demo!

## Support
If you encounter any issues during deployment, check:
1. Railway logs (for backend errors)
2. Vercel build logs (for frontend errors)  
3. Browser console (F12) for network errors
4. GitHub repository Actions tab for any failed workflows

**Good luck with your presentation! 🚀**

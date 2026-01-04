# 📤 GitHub Upload Guide

## ⚠️ IMPORTANT: First Revoke Your Exposed Token!

Your GitHub token was exposed and must be revoked immediately:

1. Go to: https://github.com/settings/tokens
2. Find the token and click "Delete"
3. Create a new token if needed (but keep it private!)

## 🚀 Upload to GitHub - Step by Step

### Method 1: Command Line (Recommended)

Open PowerShell or Command Prompt in your project folder and run:

```powershell
# Step 1: Initialize Git repository
git init

# Step 2: Add all files
git add .

# Step 3: Commit files
git commit -m "Initial commit: Alzheimer Detection System with 99.2% accuracy"

# Step 4: Add remote repository
git remote add origin https://github.com/hamzanawazsangha/Alzeihmer-Detection.git

# Step 5: Push to GitHub
git branch -M main
git push -u origin main
```

**When prompted for credentials:**
- Username: `hamzanawazsangha`
- Password: Use a **NEW** personal access token (not the old one!)

### Method 2: GitHub Desktop (Easiest)

1. **Download GitHub Desktop**
   - Visit: https://desktop.github.com/
   - Install and sign in

2. **Add Your Repository**
   - File → Add Local Repository
   - Choose: `C:\Users\ahsan\Documents\Desktop\Alzehiemer_Detectection_System`
   - Click "Create Repository" if prompted

3. **Publish to GitHub**
   - Click "Publish repository"
   - Name: `Alzeihmer-Detection`
   - Description: "Alzheimer's Detection System using Deep Learning"
   - Uncheck "Keep this code private" (if you want it public)
   - Click "Publish Repository"

### Method 3: VS Code

1. **Open Project in VS Code**
   - Open the project folder

2. **Initialize Repository**
   - Click Source Control icon (left sidebar)
   - Click "Initialize Repository"

3. **Stage and Commit**
   - Click "+" to stage all files
   - Enter commit message: "Initial commit"
   - Click checkmark to commit

4. **Publish to GitHub**
   - Click "Publish to GitHub"
   - Select "Publish to GitHub public repository"
   - Choose repository name: `Alzeihmer-Detection`

## 📋 Pre-Upload Checklist

Before uploading, ensure:

- [x] `.gitignore` file created (already done!)
- [ ] Remove any sensitive data (API keys, passwords)
- [ ] Model file size < 100MB (GitHub limit)
- [ ] README.md is complete and professional
- [ ] All code is tested and working
- [ ] Remove debug logs and temporary files

## 🔐 Creating a New GitHub Token (If Needed)

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Name: "Alzheimer Detection Upload"
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy and save securely** - you won't see it again!
7. **Never share it publicly!**

## 📦 If Model File is Too Large

If `Alzheimer_Detection_model.h5` is > 100MB:

### Option A: Use Git LFS
```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git add .gitattributes
git add model/Alzheimer_Detection_model.h5
git commit -m "Add model file with Git LFS"
git push
```

### Option B: Upload to Cloud Storage
1. Upload model to Google Drive / Dropbox
2. Add download link to README.md
3. Add model file to `.gitignore`

## 🎯 Recommended Repository Settings

After uploading:

1. **Add Description**
   - "AI-powered Alzheimer's Detection System using EfficientNetB0 with 99.2% accuracy"

2. **Add Topics**
   - `deep-learning`
   - `tensorflow`
   - `alzheimers-detection`
   - `medical-imaging`
   - `computer-vision`
   - `flask`
   - `machine-learning`

3. **Add README Sections**
   - Already included in README.md!

4. **Enable GitHub Pages** (Optional)
   - Settings → Pages
   - Source: Deploy from branch
   - Branch: main, folder: /docs (if you create docs)

## 🔄 Future Updates

To push updates later:

```powershell
git add .
git commit -m "Description of changes"
git push
```

## ❌ Common Issues

### Issue: "Repository already exists"
```powershell
# Remove existing remote and re-add
git remote remove origin
git remote add origin https://github.com/hamzanawazsangha/Alzeihmer-Detection.git
```

### Issue: "Authentication failed"
- Use a valid personal access token as password
- Ensure token has `repo` permissions

### Issue: "Large files detected"
- Use Git LFS (see above)
- Or upload model separately

### Issue: "Permission denied"
- Check you're logged into correct GitHub account
- Verify repository exists and you have access

## 📞 Need Help?

- GitHub Docs: https://docs.github.com/
- Git Docs: https://git-scm.com/doc
- Contact: Open an issue on GitHub

---

**Remember: Keep your tokens private and secure!** 🔐

Good luck with your upload! 🚀


# 🌱 Plant Disease Recognition - MLOps Project

> Complete end-to-end MLOps pipeline with Azure ML, FastAPI, Docker, Kubernetes, and GitHub Actions

**Deadline**: 15 December 2025, 23:59

---

## 📋 Overview

Automated ML pipeline for plant disease classification that:
- Trains a ResNet50 CNN model on Azure ML
- Registers and versions models automatically
- Deploys as scalable REST API on Kubernetes
- Automates everything through GitHub Actions CI/CD

**Use Case**: Agricultural app where farmers photograph crops → API identifies disease → Displays treatment recommendations

---

## 📁 Clean Project Structure

```
Project/
├── .github/workflows/
│   └── ci-cd.yaml                        # ⚡ Main automation workflow
│
├── components/                            # 🔧 Azure ML components
│   ├── data_prep/data_prep.py            # Image preprocessing
│   ├── data_split/data_split.py          # Train/test split
│   ├── training/train.py                 # Model training
│   ├── data_prep.yaml                    # Component definitions
│   ├── data_split.yaml
│   └── training.yaml
│
├── environment/                           # 🐍 Python environment
│   ├── environment.yaml                  # Azure ML environment
│   └── conda.yaml                        # Dependencies
│
├── pipelines/
│   └── plant-disease-classification.yaml  # 🔄 Main ML pipeline
│
├── inference/                             # 🚀 FastAPI deployment
│   ├── main.py                           # REST API
│   ├── Dockerfile                        # Container
│   ├── requirements.txt                  # API dependencies
│   └── model/                            # Downloaded model (auto)
│
├── k8s/                                   # ☸️ Kubernetes configs
│   ├── deployment.yaml                   # Deployment
│   └── service.yaml                      # Service
│
│
├── config.yaml                            # ⚙️ Azure configuration
└── requirements.txt                       # Local development
```

---

## � Configuration Setup (IMPORTANT!)

**Before pushing to GitHub, you need to configure your Azure credentials:**

### Option 1: Using config.yaml (Local Development)

```bash
# Copy the template
cp config.template.yaml config.yaml

# Edit config.yaml with your values:
# - subscription_id
# - resource_group  
# - workspace_name
# - compute_name
```

⚠️ **config.yaml is gitignored and won't be committed to GitHub** (keeps your credentials safe!)

### Option 2: Using .env file

```bash
# Copy the template
cp .env.example .env

# Edit .env with your values
```

⚠️ **.env is also gitignored** for security!

---

## �🚀 Quick Start (4 Steps!)

### Step 1: Create GitHub Repository

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/plant-disease-mlops.git
git push -u origin main
```

### Step 2: Add GitHub Secrets

Go to: **GitHub Repo → Settings → Secrets → Actions**

Add two secrets:

1. **AZURE_CREDENTIALS** (from your instructor)
   - If you need to create it yourself:
   ```bash
   az ad sp create-for-rbac --name "plant-disease-sp" --role contributor \
     --scopes /subscriptions/{sub-id}/resourceGroups/rg-nathan-mpops --sdk-auth
   ```

2. **GHCR_TOKEN** (GitHub Personal Access Token)
   - Go to: GitHub Settings → Developer settings → Personal access tokens
   - Generate token with `write:packages` + `read:packages` scopes
   - Copy and save as secret

### Step 3: Update Your Username

Edit `k8s/deployment.yaml` line 20:

```yaml
# Change:
image: ghcr.io/danyukezz/plant-disease-api:latest
# To:
image: ghcr.io/YOUR_GITHUB_USERNAME/plant-disease-api:latest
```

### Step 4: Run the Workflow!

1. Go to your GitHub repo → **Actions** tab
2. Select "Azure ML Workflow - Plant Disease"
3. Click "**Run workflow**"
4. Keep all settings as "false"
5. Click "**Run workflow**" button

**That's it!** The workflow will:
- ✅ Create Azure ML workspace & compute
- ✅ Register components & environment
- ✅ Run training pipeline (ResNet50)
- ✅ Download trained model
- ✅ Build Docker image
- ✅ Deploy to Kubernetes

**Time**: ~20-30 minutes

---

## 📸 Screenshots for Report (Take These!)

1. **Azure ML Workspace** - Portal dashboard
2. **Pipeline Running** - Jobs page
3. **Pipeline Graph** - Visual workflow
4. **Training Metrics** - Loss/accuracy charts
5. **Model Registry** - Registered model
6. **GitHub Actions** - Successful workflow
7. **Swagger UI** - API docs at `http://<IP>/docs`
8. **Prediction Result** - API response example
9. **Kubernetes** - `kubectl get pods` output

---

## 🧪 Testing the API

After deployment completes:

```bash
# 1. Get the external IP
kubectl get service plant-disease-api-service

# 2. Test health check
curl http://<EXTERNAL-IP>/health

# 3. View API docs in browser
open http://<EXTERNAL-IP>/docs

# 4. Test prediction
curl -X POST "http://<EXTERNAL-IP>/predict" \
  -F "file=@plant_image.jpg"
```

---

## 📝 Report Writing Guide

### Structure (12-15 pages):

1. **Introduction** (1 page)
   - Project overview & dataset
   - Problem statement

2. **Architecture** (1-2 pages)
   - System architecture diagram
   - Workflow explanation

3. **Azure ML Implementation** (3-4 pages)
   - Pipeline components explained
   - Training process & screenshots
   - Model metrics & results

4. **FastAPI & Integration** (2 pages)
   - API endpoints
   - Integration possibilities
   - Screenshots

5. **Docker & Kubernetes** (2 pages)
   - Containerization approach
   - K8s deployment strategy
   - Scaling & monitoring

6. **CI/CD Automation** (1-2 pages)
   - GitHub Actions workflow
   - Automation benefits

7. **Business Use Case** (1 page)
   - Fictional company scenario
   - Integration example
   - Benefits

8. **Conclusions** (1 page)
   - Lessons learned
   - Future improvements

---

## 🔧 Azure Configuration (Already Set!)

```yaml
Resource Group: rg-nathan-mpops
Workspace: danylo-bordunov-mlops-project
Location: Germany West Central
Compute: compute-danylo (Standard_DS2_v2)
```

---

## 🛠️ Troubleshooting

### "Compute not found"
```bash
az ml compute create --name compute-danylo --type ComputeInstance --size Standard_DS2_v2
```

### "Component not found"
```bash
az ml component create --file components/data_prep.yaml
az ml component create --file components/data_split.yaml
az ml component create --file components/training.yaml
```

### "Model download failed"
Check that model name matches: `plant-disease-classifier-model`

### "Kubernetes pod not starting"
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

---

## 💰 Cost Management

**IMPORTANT**: After your project is graded:

```bash
# Stop compute to save money
az ml compute stop --name compute-danylo

# Delete everything (AFTER grading!)
az group delete --name rg-nathan-mpops --yes
```

---

## ✅ Pre-Submission Checklist

- [ ] GitHub repository created and code pushed
- [ ] Both GitHub secrets added (AZURE_CREDENTIALS, GHCR_TOKEN)
- [ ] Updated GitHub username in k8s/deployment.yaml
- [ ] Workflow ran successfully (all 3 jobs green)
- [ ] Model trained and registered
- [ ] API deployed and accessible
- [ ] All 9 screenshots taken
- [ ] Report written (12-15 pages)
- [ ] Code is clean and documented
- [ ] Created .zip with report + code + screenshots
- [ ] Submitted before 15 Dec 23:59

---

## 🎯 Key MLOps Concepts Demonstrated

✅ **Version Control** - Git for code versioning  
✅ **Experiment Tracking** - MLflow in Azure ML  
✅ **Model Registry** - Azure ML model versioning  
✅ **CI/CD** - GitHub Actions automation  
✅ **Containerization** - Docker  
✅ **Orchestration** - Kubernetes  
✅ **Pipeline Automation** - Azure ML pipelines  
✅ **API Development** - FastAPI  
✅ **Cloud Computing** - Azure ML compute

---

## 👥 Project Info

- **Course**: MLOps
- **Deadline**: 15 December 2025, 23:59
- **Dataset**: Plant Disease Recognition (Kaggle)
- **Model**: ResNet50 (Transfer Learning)
- **Framework**: PyTorch

---

## 📞 Need Help?

Review the GitHub Actions logs and Azure ML portal for pipeline status.

**Good luck! 🍀 You've got this!**

---

*Last Updated: 13 December 2025*

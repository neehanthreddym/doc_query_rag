# Docker & AWS Deployment Guide

A simple deployment guide for the DocQuery RAG application using Docker and AWS.

---

## 🎯 AWS CI/CD Pipeline Demo

This project demonstrates an end-to-end CI/CD pipeline on AWS:

1. **Docker Build** - Containerize the RAG application
2. **GitHub Workflow** - Automated CI/CD pipeline
3. **IAM User in AWS** - AWS credentials for deployment
4. **ECR** - Container registry
5. **EC2** - Self-hosted runner for deployment

---

## 📋 Prerequisites

- AWS Account
- GitHub Account
- Docker installed locally (for testing)

---

## 🐳 Docker Setup in EC2

SSH into your EC2 instance and run these commands:

### Optional (Update System)
```bash
sudo apt-get update -y
sudo apt-get upgrade
```

### Required (Install Docker)
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Setup Vector Database (First-Time Only)

After Docker is installed, prepare the vector database:

```bash
# Create persistent data directories
mkdir -p ~/docquery_data/vector_store
mkdir -p ~/docquery_data/pdf_files

# Copy PDF files to EC2 (choose one method)

# Method 1: From your local machine
scp -i your-key.pem -r data/pdf_files/* ubuntu@YOUR_EC2_IP:~/docquery_data/pdf_files/

# Method 2: Clone repo on EC2
git clone https://github.com/YOUR_USERNAME/doc_query_rag.git
cp -r doc_query_rag/data/pdf_files/* ~/docquery_data/pdf_files/

# Build the vector database (one-time setup)
# First, pull your Docker image
docker pull YOUR_ECR_URI/docquery-rag:latest

# Run the build command
docker run --rm \
  -v ~/docquery_data/vector_store:/app/data/vector_store \
  -v ~/docquery_data/pdf_files:/app/data/pdf_files:ro \
  YOUR_ECR_URI/docquery-rag:latest \
  python app.py --build

# This creates the ChromaDB database in ~/docquery_data/vector_store
# You only need to do this once!
```

---

## ⚙️ Configure EC2 as Self-Hosted Runner

1. Go to your GitHub repository
2. Navigate to **Settings** → **Actions** → **Runners**
3. Click **New self-hosted runner**
4. Select **Linux** as the operating system
5. Follow the commands provided by GitHub on your EC2 instance

Example:
```bash
# Download
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure
./config.sh --url https://github.com/YOUR_USERNAME/doc_query_rag --token YOUR_TOKEN

# Install and start
sudo ./svc.sh install
sudo ./svc.sh start
```

---

## 🔐 Setup GitHub Secrets

Go to your GitHub repository and add these secrets:

**Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key | From AWS IAM |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key | From AWS IAM |
| `AWS_REGION` | AWS region | `us-east-1` |
| `AWS_ECR_LOGIN_URI` | ECR repository URI | `123456789012.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | ECR repository name | `docquery-rag` |
| `GROQ_API_KEY` | Groq API key | Get from https://console.groq.com/keys |

---

## 🚀 AWS Setup Steps

### 1. Create IAM User
1. Go to AWS Console → **IAM**
2. Create new user with programmatic access
3. Attach policies: `AmazonEC2ContainerRegistryFullAccess`
4. Save the Access Key ID and Secret Access Key

### 2. Create ECR Repository
```bash
aws ecr create-repository --repository-name docquery-rag --region us-east-1
```

Note the repository URI (e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com/docquery-rag`)

### 3. Launch EC2 Instance
1. Go to AWS Console → **EC2**
2. Launch instance with:
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Instance type**: t3.medium (or larger)
   - **Storage**: 30GB
   - **Security Group**: Allow ports 22 (SSH) and 8000 (HTTP)
3. Create or select an existing key pair
4. Launch instance

---

## 🔄 Deployment Workflow

Once everything is set up:

1. **Push code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy to AWS"
   git push origin main
   ```

2. **GitHub Actions will automatically**:
   - Run linting and tests
   - Build Docker image
   - Push to AWS ECR
   - Deploy to EC2 self-hosted runner
   - **Build vector database** (first time only, then reuses existing)
   - Start the application

3. **Access your application**:
   ```
   http://YOUR_EC2_PUBLIC_IP:8000
   ```

> **Note**: The first deployment will take longer (~5-10 minutes) because it builds the vector database from PDF files. Subsequent deployments will be faster as the database is persisted.

---

## 🧪 Local Testing

Test the Docker image locally before deploying:

```bash
# Build image
docker build -t docquery-rag .

# Run container
docker run -p 8000:8000 -e GROQ_API_KEY=your_key docquery-rag

# Access at http://localhost:8000
```

---

## 📊 Architecture

```
Developer → GitHub → GitHub Actions → AWS ECR → EC2 → Users
```

---

## ❗ Troubleshooting

**Docker permission denied:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Container fails to start:**
```bash
docker logs docquery-rag
```

**EC2 connection refused:**
- Check security group allows port 8000
- Verify container is running: `docker ps`

---

## 📚 References

- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [AWS ECR Guide](https://docs.aws.amazon.com/ecr/)
- [AWS EC2 Guide](https://docs.aws.amazon.com/ec2/)

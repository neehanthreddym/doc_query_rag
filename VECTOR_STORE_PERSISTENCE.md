# Vector Store Persistence in AWS Deployment

When deploying DocQuery RAG to AWS, the ChromaDB vector store needs special handling since it stores data locally.

---

## 🎯 Problem

- **Local Development**: Vector store is in `data/vector_store/` directory
- **Docker Container**: Data inside container is lost when container restarts
- **AWS EC2**: Need persistent storage across deployments

---

## ✅ Solution: Docker Volumes (Recommended for Students)

### Approach

Use Docker volumes to persist the vector store on the EC2 host machine.

---

## 📋 Step-by-Step Setup

### 1. Prepare EC2 Instance

SSH into your EC2 instance and create directories:

```bash
# Create persistent data directories on EC2
mkdir -p ~/docquery_data/vector_store
mkdir -p ~/docquery_data/pdf_files

# Copy your PDF files to EC2 (one-time setup)
# Option A: Use SCP from local machine
scp -i your-key.pem -r data/pdf_files/* ubuntu@your-ec2-ip:~/docquery_data/pdf_files/

# Option B: Clone repo and copy
cd ~
git clone https://github.com/YOUR_USERNAME/doc_query_rag.git
cp -r doc_query_rag/data/pdf_files/* ~/docquery_data/pdf_files/
```

### 2. Build Vector Database on EC2

**First-time setup only** - Build the vector database on your EC2 instance:

```bash
# Pull the Docker image
docker pull YOUR_ECR_URI/docquery-rag:latest

# Run the build command (one-time)
docker run --rm \
  -v ~/docquery_data/vector_store:/app/data/vector_store \
  -v ~/docquery_data/pdf_files:/app/data/pdf_files:ro \
  YOUR_ECR_URI/docquery-rag:latest \
  python app.py --build

# This creates the ChromaDB database in ~/docquery_data/vector_store
```

### 3. Update GitHub Actions Workflow

Update the deployment step in `.github/workflows/main.yaml` to use volumes:

```yaml
- name: Run docker image to serve users
  run: |
    docker run -d -p 8000:8000 --name=docquery-rag \
      -e GROQ_API_KEY=${{ secrets.GROQ_API_KEY }} \
      -v /home/ubuntu/docquery_data/vector_store:/app/data/vector_store \
      -v /home/ubuntu/docquery_data/pdf_files:/app/data/pdf_files:ro \
      ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
```

**Note the `-v` flags**: 
- `-v /home/ubuntu/docquery_data/vector_store:/app/data/vector_store` - Mounts EC2 directory to container
- `-v /home/ubuntu/docquery_data/pdf_files:/app/data/pdf_files:ro` - Mounts PDFs as read-only

---

## 🔄 Deployment Workflow

```
First Time Setup:
1. Create directories on EC2
2. Copy PDF files to EC2
3. Build vector database once
4. Run container with volumes

Subsequent Deployments:
1. Push code to GitHub
2. GitHub Actions builds new image
3. Pushes to ECR
4. EC2 pulls new image
5. Stops old container
6. Starts new container with SAME volumes
7. Vector database persists! ✅
```

---

## 📝 Complete Updated GitHub Actions Workflow

Here's the updated `continuous-deployment` job:

```yaml
continuous-deployment:
  needs: build-and-push-ecr-image
  runs-on: self-hosted
  steps:
    - name: Checkout Code
      uses: actions/checkout@v4
    
    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ secrets.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
    
    - name: Pull latest image
      run: |
        docker pull ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
    
    - name: Stop and remove old container
      run: |
        docker stop docquery-rag || true
        docker rm docquery-rag || true
    
    - name: Run docker image with persistent volumes
      run: |
        docker run -d -p 8000:8000 --name=docquery-rag \
          --restart=unless-stopped \
          -e GROQ_API_KEY=${{ secrets.GROQ_API_KEY }} \
          -v /home/ubuntu/docquery_data/vector_store:/app/data/vector_store \
          -v /home/ubuntu/docquery_data/pdf_files:/app/data/pdf_files:ro \
          ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
    
    - name: Clean previous images and containers
      run: |
        docker system prune -f
```

---

## 🎓 Alternative Approaches

### Option 2: AWS EBS Volume (Advanced)

For production, you could use an EBS volume:

```bash
# Attach EBS volume to EC2
# Mount it to /mnt/docquery_data
# Use same volume mounting approach
```

### Option 3: Build Database in Docker Image (Not Recommended)

You could build the vector database INTO the Docker image, but this makes the image very large and rebuilds the database every deployment.

---

## ✅ Advantages of Volume Approach

1. **Persistent** - Data survives container restarts
2. **Fast** - No need to rebuild database on every deployment
3. **Simple** - Just mount directories
4. **Updatable** - Can add PDFs without rebuilding image

---

## 🔍 Verify It's Working

After deployment:

```bash
# SSH to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Check if vector store exists
ls -lh ~/docquery_data/vector_store/

# Should see ChromaDB files
# Example output:
# chroma.sqlite3
# [uuid folders]

# Check container is using the volume
docker inspect docquery-rag | grep Mounts -A 20
```

---

## 📊 Summary

**Problem**: ChromaDB vector store is local, gets lost on container restart

**Solution**: Use Docker volumes to persist data on EC2 host

**Setup**: 
1. Create directories on EC2
2. Build database once
3. Mount volumes when running container

**Result**: Vector database persists across deployments! 🎉

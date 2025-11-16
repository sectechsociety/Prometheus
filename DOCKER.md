# Docker Deployment Guide 🐳

Complete guide for deploying Project Prometheus using Docker and Docker Compose.

## 📋 Overview

Project Prometheus uses Docker Compose to orchestrate two main services:
- **Backend**: FastAPI application with RAG system (port 8000)
- **Frontend**: React + Vite application (port 5173)

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM recommended
- 2GB+ disk space

### Basic Commands

```bash
# Start all services (development mode with hot reload)
docker-compose up --build

# Start in detached mode (background)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### First Time Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd Prometheus

# 2. Verify required files exist
ls -la services/ingest/chroma_db/  # ChromaDB should be populated
ls -la backend/app/model/prometheus_lora_adapter/  # LoRA adapter

# 3. Build and start
docker-compose up --build

# 4. Verify services
curl http://localhost:8000/health  # Backend health
curl http://localhost:5173  # Frontend (should return HTML)

# 5. Access application
# Open browser: http://localhost:5173
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Backend
LOG_LEVEL=INFO
CHROMA_DB_PATH=services/ingest/chroma_db
PROMETHEUS_MODEL_PATH=app/model/prometheus_lora_adapter

# Frontend
VITE_API_BASE_URL=http://localhost:8000

# Docker
COMPOSE_PROJECT_NAME=prometheus
```

### Port Configuration

To change default ports, edit `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8080:8000"  # Host:Container
  
  frontend:
    ports:
      - "3000:5173"  # Host:Container
```

## 📦 Service Details

### Backend Service

**Image**: Python 3.11-slim  
**Port**: 8000  
**Health Check**: http://localhost:8000/health  
**Restart Policy**: unless-stopped

**Volumes**:
- `./backend:/app` - Source code (hot reload)
- `./services/ingest/chroma_db:/app/services/ingest/chroma_db` - Vector DB
- `./backend/app/model/prometheus_lora_adapter:/app/app/model/prometheus_lora_adapter` - Model

**Command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

### Frontend Service

**Image**: Node 20-alpine  
**Port**: 5173  
**Depends On**: backend (healthy)  
**Restart Policy**: unless-stopped

**Volumes**:
- `./frontend:/app` - Source code (hot reload)
- `/app/node_modules` - Prevent overwrite

**Environment**:
- `CHOKIDAR_USEPOLLING=true` - Enable file watching in Docker

## 🛠️ Development Workflow

### Hot Reload Development

```bash
# Start with live reload
docker-compose up

# Make changes to code
# Backend: Changes auto-reload via uvicorn --reload
# Frontend: Changes auto-reload via Vite HMR

# View logs in real-time
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Debugging

```bash
# Exec into running container
docker-compose exec backend bash
docker-compose exec frontend sh

# View container logs
docker-compose logs backend --tail=100
docker-compose logs frontend --tail=100

# Restart specific service
docker-compose restart backend
docker-compose restart frontend

# Rebuild specific service
docker-compose build --no-cache backend
docker-compose up -d backend
```

### Database Management

```bash
# Backup ChromaDB
docker-compose exec backend tar -czf /tmp/chroma_backup.tar.gz services/ingest/chroma_db
docker cp prometheus-backend:/tmp/chroma_backup.tar.gz ./chroma_backup.tar.gz

# Restore ChromaDB
docker cp ./chroma_backup.tar.gz prometheus-backend:/tmp/
docker-compose exec backend tar -xzf /tmp/chroma_backup.tar.gz -C /

# Repopulate database
docker-compose exec backend python -m app.rag.populate_db
```

## 🚀 Production Deployment

### Production Build

For production, remove volume mounts and disable hot reload:

**docker-compose.prod.yml**:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/app/services/ingest/chroma_db
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=WARNING
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
    restart: always

  frontend:
    build:
      context: ./frontend
      target: production  # Use nginx stage
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: always

volumes:
  chroma_data:

networks:
  default:
    driver: bridge
```

**Deploy**:
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start production services
docker-compose -f docker-compose.prod.yml up -d

# View status
docker-compose -f docker-compose.prod.yml ps
```

### Docker Swarm (Scalable)

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml prometheus

# Scale services
docker service scale prometheus_backend=3

# View services
docker stack services prometheus

# Remove stack
docker stack rm prometheus
```

## 📊 Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "guidelines_count": 811
}

# Container health status
docker-compose ps
# Should show "healthy" status for backend
```

### Resource Usage

```bash
# View resource usage
docker stats

# Specific container
docker stats prometheus-backend
docker stats prometheus-frontend

# Memory limit (add to docker-compose.yml)
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 512M
```

### Logs

```bash
# All logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f backend

# Export logs
docker-compose logs > prometheus_logs.txt
```

## 🐛 Troubleshooting

### Common Issues

**1. Port already in use**
```bash
# Find process using port
lsof -i :8000
lsof -i :5173

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

**2. Container fails to start**
```bash
# View detailed logs
docker-compose logs backend

# Check if image built correctly
docker images | grep prometheus

# Rebuild without cache
docker-compose build --no-cache backend
```

**3. ChromaDB permission errors**
```bash
# Fix permissions
sudo chown -R $USER:$USER services/ingest/chroma_db/

# Or run with proper user in Dockerfile
USER prometheus
```

**4. Frontend can't connect to backend**
```bash
# Verify network
docker network ls
docker network inspect prometheus-network

# Check backend is accessible
docker-compose exec frontend wget -O- http://backend:8000/health
```

**5. Out of memory**
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory (4GB+)

# Or add memory limits to docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

### Clean Slate Reset

```bash
# Stop and remove everything
docker-compose down -v

# Remove all images
docker rmi $(docker images -q prometheus-*)

# Remove orphaned volumes
docker volume prune

# Rebuild from scratch
docker-compose up --build
```

## 📚 Best Practices

### Development
- ✅ Use volume mounts for hot reload
- ✅ Keep logs visible with `docker-compose logs -f`
- ✅ Use `.dockerignore` to speed up builds
- ✅ Tag images with version numbers

### Production
- ✅ Remove volume mounts (embed code in image)
- ✅ Use multi-stage builds for smaller images
- ✅ Set restart policy to `always`
- ✅ Use environment variables for config
- ✅ Implement health checks
- ✅ Use specific version tags (not `latest`)
- ✅ Run as non-root user
- ✅ Enable HTTPS/SSL
- ✅ Set up log aggregation
- ✅ Monitor resource usage

### Security
- ✅ Don't expose unnecessary ports
- ✅ Use secrets for sensitive data
- ✅ Keep base images updated
- ✅ Scan images for vulnerabilities
- ✅ Use minimal base images (alpine, slim)
- ✅ Run containers as non-root

## 🔗 Useful Commands Reference

```bash
# Build
docker-compose build
docker-compose build --no-cache
docker-compose build backend

# Start
docker-compose up
docker-compose up -d
docker-compose up --build

# Stop
docker-compose stop
docker-compose down
docker-compose down -v

# Logs
docker-compose logs
docker-compose logs -f
docker-compose logs backend --tail=50

# Exec
docker-compose exec backend bash
docker-compose exec frontend sh

# Restart
docker-compose restart
docker-compose restart backend

# Status
docker-compose ps
docker-compose top

# Clean
docker-compose down -v
docker system prune -a
docker volume prune
```

## 📄 Related Files

- `docker-compose.yml` - Main compose configuration
- `backend/Dockerfile` - Backend image definition
- `frontend/Dockerfile` - Frontend image definition
- `backend/.dockerignore` - Backend build exclusions
- `frontend/.dockerignore` - Frontend build exclusions

## 🆘 Support

For issues or questions:
1. Check this guide
2. View logs: `docker-compose logs`
3. Check GitHub issues
4. Rebuild from scratch

---

**Last Updated**: November 16, 2025  
**Docker Version**: 24.0+  
**Docker Compose Version**: 2.0+

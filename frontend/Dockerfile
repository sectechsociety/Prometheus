# Prometheus Frontend - Production Dockerfile
# Multi-stage build for development and production

FROM node:20-alpine as base

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy application source
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Development command (with hot reload)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# Production build stage (optional, use for static hosting)
FROM base as builder
RUN npm run build

# Production serve stage (optional, use nginx for static hosting)
FROM nginx:alpine as production
COPY --from=builder /app/dist /usr/share/nginx/html
COPY --from=builder /app/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]


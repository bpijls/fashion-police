# Fashion Police — kiosk frontend. Build the SPA, serve it (and proxy /api) with Caddy.
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# VITE_API_BASE stays as the default "/api/v1" (same origin, proxied below).
# Pass --build-arg VITE_MODEL_BASE=/models to use bundled TF.js models.
ARG VITE_MODEL_BASE=""
RUN VITE_MODEL_BASE="${VITE_MODEL_BASE}" npm run build

FROM caddy:2-alpine
COPY --from=build /app/dist /srv/www
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 80

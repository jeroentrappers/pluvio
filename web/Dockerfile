# Build the PWA once; the API origin + tile key are injected at *runtime*
# (config.js, rendered from env on startup — see docker-entrypoint.d).
FROM node:24-alpine AS build
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
# Runtime config: an entrypoint script renders /config.js from this template
# using the VITE_* env vars before nginx starts.
COPY config.template.js /etc/pluvio/config.template.js
COPY --chmod=0755 docker-entrypoint.d/40-pluvio-config.sh /docker-entrypoint.d/40-pluvio-config.sh
EXPOSE 80

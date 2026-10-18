# Build the static site, then serve it from nginx. The MediaPipe runtime and
# model are fetched during the build (npm run vendor) and served from the same
# origin, so the demo does not depend on a CDN (CLAUDE.md 14).
FROM node:20-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run vendor && npm run build
# Compress once at build time; nginx serves the .gz with gzip_static, so the
# VM spends no CPU on it. Measured: 17.7 MB of runtime + model + bundle becomes
# 8.8 MB (the wasm shrinks to 30%, the .task to 74%).
RUN find dist -type f \( -name '*.wasm' -o -name '*.js' -o -name '*.css' \
        -o -name '*.html' -o -name '*.task' \) \
        -exec sh -c 'gzip -9 -c "$1" > "$1.gz"' _ {} \;

FROM nginx:1.27-alpine
COPY --from=build /web/dist /usr/share/nginx/html

# Build the static site, then serve it from nginx. The MediaPipe runtime and
# model are fetched during the build (npm run vendor) and served from the same
# origin, so the demo does not depend on a CDN (CLAUDE.md 14).
FROM node:20-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run vendor && npm run build

FROM nginx:1.27-alpine
COPY --from=build /web/dist /usr/share/nginx/html

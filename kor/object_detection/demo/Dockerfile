FROM nginx
WORKDIR /usr/share/nginx/html/

COPY static/ static/
COPY home.html index.html

COPY nginx/conf.d/default.conf /etc/nginx/conf.d/default.conf

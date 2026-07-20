#!/bin/sh
# Sourced automatically by Tomcat's catalina.sh at startup.
# Heap sized generously for the demo (OpenOLAT is a large webapp); override
# via docker-compose.override.yml if the host is memory-constrained.
export CATALINA_OPTS="${CATALINA_OPTS} \
  -Xms512m -Xmx2048m -XX:MaxMetaspaceSize=512m \
  -Djava.awt.headless=true \
  -Djava.net.preferIPv4Stack=true \
  -Duser.timezone=UTC \
  -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/opt/openolat/logs"

# OpenOLAT 19.1.6 on a NATIVE arm64 Tomcat — no emulation, no recompilation.
#
# WHY THIS SHAPE (see FIDELITY.md § "arm64 deploy"):
#   * OpenOLAT publishes no prebuilt WAR, and the only public image
#     (frentix/openolat:19.1.6) is amd64-only. But an OpenOLAT deployment is a
#     pure-Java *exploded webapp* — architecture-independent JVM bytecode.
#   * Stage 1 references that amd64 image ONLY as a file source. `COPY --from`
#     reads its filesystem; it NEVER executes amd64 code, so this builds and
#     runs natively on aarch64 with zero qemu/binfmt.
#   * Stage 2 is the official multi-arch `tomcat:10.1.35-jdk17-temurin`
#     (arm64 native) — the exact Tomcat/JDK major that frentix bundles
#     (Tomcat 10.1.35 / Temurin 17), so the Jakarta-EE webapp runs unchanged.
#
# The design pinned OpenOLAT 20.3.x; 19.1.6 is the newest version for which a
# ready-to-run webapp is publicly fetchable (no 20.x WAR or image exists).
# 19.1.6 carries every surface the demo needs: webcontent/CP page hosting,
# the LTI + External-Page course elements for the embed, and the REST API for
# scripted provisioning. See FIDELITY.md § "Version deviation".

# ---- Stage 1: file source (amd64 image, never executed) --------------------
FROM --platform=linux/amd64 frentix/openolat:19.1.6 AS oo-src

# ---- Stage 2: native runtime ----------------------------------------------
FROM tomcat:10.1.35-jdk17-temurin

# Remove the stock Tomcat sample webapps (ROOT/docs/examples/manager) so only
# OpenOLAT is served and the attack surface is minimal.
RUN rm -rf "$CATALINA_HOME"/webapps/* "$CATALINA_HOME"/webapps.dist 2>/dev/null || true

# The exploded OpenOLAT webapp lives OUTSIDE appBase (webapps/) so Tomcat does
# not double-deploy it; the context descriptor below points docBase here.
COPY --from=oo-src /home/openolat/webapp /opt/openolat/webapp

# Container-level JNDI DataSource (defined in the context descriptor) is
# instantiated by Tomcat's OWN classloader, so the Postgres driver must sit in
# ${catalina.home}/lib — not just the webapp's WEB-INF/lib. Copy the exact
# driver the webapp already ships (guaranteed-compatible 42.7.5).
COPY --from=oo-src /home/openolat/webapp/WEB-INF/lib/postgresql-42.7.5.jar "$CATALINA_HOME"/lib/postgresql.jar

# olat.local.properties must be on the common classpath. In the official image
# CATALINA_BASE == CATALINA_HOME == /usr/local/tomcat, and catalina.properties'
# default common.loader includes "${catalina.base}/lib" — so dropping the file
# in tomcat/lib places it on the classpath exactly as frentix's split
# CATALINA_BASE=/home/openolat does. (See FIDELITY.md § "classpath".)
COPY config/olat.local.properties "$CATALINA_HOME"/lib/olat.local.properties

# Per-context descriptor: binds /openolat -> the exploded webapp and declares
# the Postgres JNDI DataSource the app resolves via db.jndi.
COPY config/openolat.xml "$CATALINA_HOME"/conf/Catalina/localhost/openolat.xml

# Heap / headless / IPv4 JVM opts (Tomcat sources bin/setenv.sh automatically).
COPY config/setenv.sh "$CATALINA_HOME"/bin/setenv.sh
RUN chmod +x "$CATALINA_HOME"/bin/setenv.sh \
 && mkdir -p /opt/openolat/olatdata /opt/openolat/logs

EXPOSE 8080
# catalina.sh run is the stock CMD from the base image.

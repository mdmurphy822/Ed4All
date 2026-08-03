# IMS Common Cartridge schema dependency

Ed4All validates Common Cartridge packages offline, but it does not publish the
third-party IMS Global/1EdTech and W3C schema payloads in this repository.
Operators must install the upstream files in this directory before running QTI
or cartridge-conformance gates. Missing or unreadable schemas are blocking
validation errors; Ed4All does not silently fall back to partial validation.

## Required files

Install these nine files directly beside this README:

- `cc_extresource_assignmentv1p0.xsd`
- `ccv1p3_imsccauth_v1p3.xsd`
- `ccv1p3_imscp_v1p2_v1p0.xsd`
- `ccv1p3_imscsmd_v1p0.xsd`
- `ccv1p3_imsdt_v1p3.xsd`
- `ccv1p3_lommanifest_v1p0.xsd`
- `ccv1p3_lomresource_v1p0.xsd`
- `ccv1p3_qtiasiv1p2p1.xsd`
- `xml.xsd`

The exact installation path is `Courseforge/schemas/imscc/<filename>`,
relative to the repository root.

## Provenance and version pin

The dependency set is pinned by specification identity: IMS Common Cartridge
1.3 (including its CP 1.2 manifest profile, QTI 1.2.1 profile, and LOM 1.3
profiles), the Assignment extension 1.0, and the W3C XML namespace schema.
Acquire each file from its official HTTPS URL and install it under the local
name shown here:

| Local filename | Official upstream file |
|---|---|
| `cc_extresource_assignmentv1p0.xsd` | `https://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd` |
| `ccv1p3_imsccauth_v1p3.xsd` | `https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsccauth_v1p3.xsd` |
| `ccv1p3_imscp_v1p2_v1p0.xsd` | `https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd` |
| `ccv1p3_imscsmd_v1p0.xsd` | `https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscsmd_v1p0.xsd` |
| `ccv1p3_imsdt_v1p3.xsd` | `https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsdt_v1p3.xsd` |
| `ccv1p3_lommanifest_v1p0.xsd` | `https://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lommanifest_v1p0.xsd` |
| `ccv1p3_lomresource_v1p0.xsd` | `https://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lomresource_v1p0.xsd` |
| `ccv1p3_qtiasiv1p2p1.xsd` | `https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd` |
| `xml.xsd` | `https://www.w3.org/2001/xml.xsd` |

The two shortened local names are deliberate compatibility names used by the
validators. The uppercase `LOM` URL segment is significant.

The standards bodies do not publish a stable checksum manifest alongside this
set. URLs alone are therefore not a reproducibility pin: an operator must
record the SHA-256 of both the upstream response and the installed bytes in
ignored local state. Review and accept a newly generated lock before using it
as the pin for another seat or build. Regenerate it only for an intentional
dependency update, and keep the prior lock for comparison.

Preserve every upstream copyright, IPR, license, and distribution notice in
full. Do not copy these payloads into a commit. The repository's `.gitignore`
intentionally excludes every `*.xsd` in this directory.

The downloaded manifest schema uses network imports. Offline validation
requires replacing only its five `schemaLocation` values with these sibling
filenames while leaving the namespace declarations unchanged:

- `xml.xsd`
- `ccv1p3_imsccauth_v1p3.xsd`
- `ccv1p3_lommanifest_v1p0.xsd`
- `ccv1p3_lomresource_v1p0.xsd`
- `ccv1p3_imscsmd_v1p0.xsd`

## Deterministic acquisition and verification

Use an operator-controlled acquisition script that performs these steps in
order:

1. Download every URL above over HTTPS into a temporary directory.
2. Refuse redirects to a non-HTTPS origin and refuse missing or empty files.
3. Hash the unmodified responses and compare them with the accepted local lock,
   when one exists.
4. Copy them to the mapped local filenames and apply the five exact manifest
   import substitutions described above.
5. Hash the installed files, parse all nine as XML, and compile the manifest as
   an XML Schema.
6. Write the source URL, upstream SHA-256, local filename, and installed
   SHA-256 to
   `runtime/state/dependency-locks/imscc-v1p3.json`.

Both `runtime/` and the installed `*.xsd` files are gitignored. The lock is
operator evidence, not a repository artifact; do not commit it or the schema
payloads. On every later installation, download into a fresh temporary
directory and require both recorded hashes to match before replacing the local
files. A mismatch is a dependency update requiring review, not a reason to
silently rewrite the lock.

The following command implements that two-pass process. On its first run it
downloads and validates into a temporary directory, writes only
`imscc-v1p3.candidate.json`, and does not install the payloads. Review the
candidate, rename it to `imscc-v1p3.json` to accept it, then rerun the same
command. The second run requires every hash to match before it installs the
files:

```bash
python - <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from lxml import etree

destination = Path("Courseforge/schemas/imscc")
lock_dir = Path("runtime/state/dependency-locks")
accepted_lock = lock_dir / "imscc-v1p3.json"
candidate_lock = lock_dir / "imscc-v1p3.candidate.json"
sources = {
    "cc_extresource_assignmentv1p0.xsd": "https://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd",
    "ccv1p3_imsccauth_v1p3.xsd": "https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsccauth_v1p3.xsd",
    "ccv1p3_imscp_v1p2_v1p0.xsd": "https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd",
    "ccv1p3_imscsmd_v1p0.xsd": "https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscsmd_v1p0.xsd",
    "ccv1p3_imsdt_v1p3.xsd": "https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsdt_v1p3.xsd",
    "ccv1p3_lommanifest_v1p0.xsd": "https://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lommanifest_v1p0.xsd",
    "ccv1p3_lomresource_v1p0.xsd": "https://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lomresource_v1p0.xsd",
    "ccv1p3_qtiasiv1p2p1.xsd": "https://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd",
    "xml.xsd": "https://www.w3.org/2001/xml.xsd",
}

imports = {
    "http://www.imsglobal.org/xsd/w3/2001/xml.xsd": "xml.xsd",
    "http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsccauth_v1p3.xsd": "ccv1p3_imsccauth_v1p3.xsd",
    "http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lommanifest_v1p0.xsd": "ccv1p3_lommanifest_v1p0.xsd",
    "http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lomresource_v1p0.xsd": "ccv1p3_lomresource_v1p0.xsd",
    "http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscsmd_v1p0.xsd": "ccv1p3_imscsmd_v1p0.xsd",
}

with TemporaryDirectory() as temporary:
    staging = Path(temporary)
    upstream_hashes = {}
    for local_name, url in sources.items():
        request = Request(url, headers={"User-Agent": "Ed4All schema installer"})
        with urlopen(request, timeout=30) as response:
            if not response.geturl().startswith("https://"):
                raise SystemExit(f"Refusing non-HTTPS redirect: {response.geturl()}")
            payload = response.read()
        if not payload:
            raise SystemExit(f"Empty response from {url}")
        upstream_hashes[local_name] = sha256(payload).hexdigest()
        (staging / local_name).write_bytes(payload)

    manifest = staging / "ccv1p3_imscp_v1p2_v1p0.xsd"
    text = manifest.read_text(encoding="utf-8")
    for remote, local in imports.items():
        if text.count(remote) != 1:
            raise SystemExit(f"Expected exactly one manifest import for {remote}")
        text = text.replace(remote, local)
    manifest.write_text(text, encoding="utf-8")

    for path in sorted(staging.glob("*.xsd")):
        etree.parse(str(path))
    etree.XMLSchema(etree.parse(str(manifest)))

    records = []
    for local_name, url in sources.items():
        records.append({
            "local_filename": local_name,
            "source_url": url,
            "upstream_sha256": upstream_hashes[local_name],
            "installed_sha256": sha256(
                (staging / local_name).read_bytes()
            ).hexdigest(),
        })
    lock = {"dependency_set": "imscc-v1p3", "files": records}

    lock_dir.mkdir(parents=True, exist_ok=True)
    if not accepted_lock.exists():
        candidate_lock.write_text(json.dumps(lock, indent=2) + "\n")
        raise SystemExit(
            f"Review {candidate_lock}, then rename it to {accepted_lock} and rerun"
        )
    if json.loads(accepted_lock.read_text()) != lock:
        raise SystemExit("Downloaded schemas do not match the accepted IMSCC lock")

    destination.mkdir(parents=True, exist_ok=True)
    for local_name in sources:
        shutil.copy2(staging / local_name, destination / local_name)
    print("Installed nine schemas matching the accepted IMSCC dependency lock.")
PY
```

For the eight unmodified files, the upstream and installed hashes should be
identical. Only the manifest should differ, solely because its five imports are
made local.

## Verify the installation

From the repository root, run:

```bash
python - <<'PY'
from pathlib import Path
from lxml import etree

root = Path("Courseforge/schemas/imscc")
required = {
    "cc_extresource_assignmentv1p0.xsd",
    "ccv1p3_imsccauth_v1p3.xsd",
    "ccv1p3_imscp_v1p2_v1p0.xsd",
    "ccv1p3_imscsmd_v1p0.xsd",
    "ccv1p3_imsdt_v1p3.xsd",
    "ccv1p3_lommanifest_v1p0.xsd",
    "ccv1p3_lomresource_v1p0.xsd",
    "ccv1p3_qtiasiv1p2p1.xsd",
    "xml.xsd",
}
missing = sorted(name for name in required if not (root / name).is_file())
if missing:
    raise SystemExit("Missing IMSCC schemas: " + ", ".join(missing))
for name in sorted(required):
    etree.parse(str(root / name))
etree.XMLSchema(etree.parse(str(root / "ccv1p3_imscp_v1p2_v1p0.xsd")))
print("IMSCC schema dependency is complete and loadable.")
PY
```

Then run the schema-dependent validator tests:

```bash
pytest -q lib/validators/tests/test_qti_well_formed.py \
  lib/validators/tests/test_cartridge_conformance.py
```

See [the installation guide](../../../docs/operations/installation.md) for the
full environment and dependency setup.

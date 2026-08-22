# three.js — vendored

Revision **r185**, npm package `three@0.185.1`. `THREE.REVISION` in the
bundle is the authoritative check; the minifier folds it into a variable, so
read it from a loaded module rather than grepping the file.

Browser asset, not a Python dependency: it is served from `/static/` and never
imported by anything under `backend/`, so it does not belong in
`backend/requirements.txt` and is out of scope for RULE 6's pip audit.

Vendored rather than loaded from a CDN because the rest of this frontend has no
build step and no third-party runtime origin. A CDN would add a DNS lookup, a
TLS handshake and an availability dependency to first paint, and would put a
`<script>` origin outside our control on a page that renders medical content.

## Both files are required

`three.module.min.js` ends with an import of `./three.core.min.js` — a relative
sibling. Copying only the first gives a 404 on the second, and the failure is
silent: the module graph never resolves, no exception surfaces on the page, and
the canvas simply stays blank. Verified with:

    grep -o "from *['\"][^'\"]*['\"]" three.module.min.js
    # => from"./three.core.min.js"

## Upgrading

    npm pack three@<version>
    tar -xzf three-<version>.tgz
    cp package/build/three.{module,core}.min.js  backend/api/static/vendor/three/
    cp package/LICENSE                           backend/api/static/vendor/three/

Then re-run the import-specifier grep above: three has moved this split before
and may move it again.

## Cache busting

The importmap target deliberately carries no `?v=` query. The version is pinned
by content — this directory only changes when someone runs the upgrade steps —
and a query on the mapped URL would be inherited by `three.module.min.js` but
not by the relative `./three.core.min.js` it pulls in, leaving the two halves on
different cache lifetimes.

Licence: MIT, see `LICENSE`.

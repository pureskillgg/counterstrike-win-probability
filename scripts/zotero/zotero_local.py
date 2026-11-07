"""Read and write a Zotero library through Zotero 10's LOCAL API (the running desktop app, 127.0.0.1:23119). Standard library only.

Why this exists: the Meridian `zotero_search` tool can only read, and the zotero.org web API needs the user's key. Zotero 10 exposes the web API's
shape on localhost; reads are open, writes need a key that the USER grants once in a Zotero dialog (POST /api/local/authorize with an appName).
That key is kept OUTSIDE the repository (path in $ZOTERO_LOCAL_KEY_FILE, a JSON file {"key": "..."}); this module never prints it.

  from zotero_local import Zotero
  z = Zotero()                                   # reads $ZOTERO_LOCAL_KEY_FILE
  z.collections()                                # {key: {"name", "parent", "version"}}
  z.ensure_collection("2 Calibration", parent=z.find_collection("CS2 CSDS Round Win Probability"))
  z.items(collection=key)                        # top-level items (no attachments/notes)
  z.create_items([{...}], collection_keys=[key]) # new items, written in batches of 50, with a write token each
  z.set_collections(item, [keys])                # replace an item's collection list (version-checked)

Command line:  python zotero_local.py tree        (print the collection tree with item counts)
               python zotero_local.py authorize   (ask the user for a write key and store it in $ZOTERO_LOCAL_KEY_FILE)
Every write goes through _write(), which sends If-Unmodified-Since-Version (item/collection version or the library version) so a concurrent
edit made in the Zotero window is never overwritten silently.
"""
import json
import os
import secrets
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:23119"
LIB = BASE + "/api/users/0"
SKIP_TYPES = {"attachment", "note", "annotation"}


class ZoteroError(RuntimeError):
    pass


class Zotero:
    def __init__(self, key_file=None, timeout=60):
        self.timeout = timeout
        path = key_file or os.environ.get("ZOTERO_LOCAL_KEY_FILE")
        self._key = None
        if path and os.path.exists(path):
            self._key = json.load(open(path, encoding="utf-8"))["key"]
        self._server_id = None
        self.library_version = None

    # --- transport
    def _call(self, method, url, body=None, headers=None, timeout=None):
        h = {"Zotero-API-Version": "3"}
        if self._server_id:
            h["Zotero-Server-ID"] = self._server_id
        if self._key:
            h["Zotero-API-Key"] = self._key
        h.update(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout if timeout is None else timeout) as r:
                raw = r.read()
                return r.status, r.headers, (json.loads(raw) if raw and "json" in (r.headers.get("Content-Type") or "") else raw.decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ZoteroError(f"Zotero is not reachable on {BASE} ({e}); is the desktop app running?") from e

    def _ensure(self):
        if self._server_id is None:
            status, hdr, _ = self._call("GET", LIB + "/collections?limit=1")
            if status != 200:
                raise ZoteroError(f"Zotero is not answering on {BASE} (status {status}); is the desktop app running?")
            self._server_id = hdr.get("Zotero-Server-ID")
            self.library_version = int(hdr.get("Last-Modified-Version") or 0)

    def _get_all(self, path):
        self._ensure()
        out, start = [], 0
        while True:
            sep = "&" if "?" in path else "?"
            status, hdr, body = self._call("GET", f"{LIB}{path}{sep}limit=100&start={start}")
            if status != 200:
                raise ZoteroError(f"GET {path} -> {status}: {str(body)[:200]}")
            out.extend(body)
            self.library_version = int(hdr.get("Last-Modified-Version") or self.library_version or 0)
            start += len(body)
            if len(body) < 100 or start >= int(hdr.get("Total-Results") or 0):
                return out

    def _write(self, method, path, body, version, library_level=False):
        self._ensure()
        if not self._key:
            raise ZoteroError("no write key: run `authorize` once and set ZOTERO_LOCAL_KEY_FILE")
        h = {"Zotero-Write-Token": secrets.token_hex(16)}
        if version is not None:
            h["If-Unmodified-Since-Version"] = str(version)
        status, hdr, resp = self._call(method, LIB + path, body, h)
        if status == 412 and library_level:
            # a library-level write (new collection or items): other writes (PATCH answers carry no new version) moved the library version on;
            # re-read it and retry once. Item-level writes (library_level=False, version = that item's version) are never retried: a 412 there
            # means somebody edited the item in Zotero meanwhile, and that edit must not be overwritten.
            _, hdr2, _ = self._call("GET", LIB + "/collections?limit=1")
            self.library_version = int(hdr2.get("Last-Modified-Version") or 0)
            h["If-Unmodified-Since-Version"] = str(self.library_version)
            h["Zotero-Write-Token"] = secrets.token_hex(16)
            status, hdr, resp = self._call(method, LIB + path, body, h)
        if status in (200, 201, 204):
            if hdr.get("Last-Modified-Version"):
                self.library_version = int(hdr["Last-Modified-Version"])
            return resp
        raise ZoteroError(f"{method} {path} -> {status}: {str(resp)[:300]}")

    # --- reads
    def collections(self):
        return {c["key"]: {"name": c["data"]["name"], "parent": c["data"].get("parentCollection") or None, "version": c["version"],
                           "num_items": c["meta"].get("numItems", 0)} for c in self._get_all("/collections")}

    def find_collection(self, name, parent=None):
        hits = [k for k, c in self.collections().items() if c["name"] == name and (c["parent"] or None) == parent]
        return hits[0] if hits else None

    def items(self, collection=None, top=True):
        path = f"/collections/{collection}/items" if collection else "/items"
        path += "/top" if top else ""
        return [i for i in self._get_all(path) if i["data"]["itemType"] not in SKIP_TYPES]

    def all_items(self, include_trash=False):
        """Top-level items of the whole library; with include_trash also the items in the Zotero trash (so a source the user deleted is not re-added)."""
        out = [i for i in self._get_all("/items/top") if i["data"]["itemType"] not in SKIP_TYPES]
        if include_trash:
            out += [i for i in self._get_all("/items/trash") if i["data"]["itemType"] not in SKIP_TYPES]
        return out

    # --- writes
    def ensure_collection(self, name, parent=None):
        """The key of the collection called `name` under `parent`, created when missing."""
        existing = self.find_collection(name, parent)
        if existing:
            return existing
        body = [{"name": name, **({"parentCollection": parent} if parent else {})}]
        resp = self._write("POST", "/collections", body, self.library_version, library_level=True)
        return resp["successful"]["0"]["key"]

    def create_items(self, items, collection_keys=()):
        """Create items (Zotero item JSON without keys) in batches of 50; returns the new item keys in order."""
        keys = []
        for i in range(0, len(items), 50):
            batch = []
            for it in items[i:i + 50]:
                d = dict(it)
                if collection_keys:
                    d["collections"] = list(collection_keys)
                batch.append(d)
            resp = self._write("POST", "/items", batch, self.library_version, library_level=True)
            if resp.get("failed"):
                raise ZoteroError(f"item batch {i}: {resp['failed']}")
            keys.extend(resp["successful"][str(n)]["key"] for n in range(len(batch)))
        return keys

    def set_collections(self, item, collection_keys):
        """Replace the collection list of one item (a dict as returned by items()); version-checked."""
        return self._write("PATCH", f"/items/{item['key']}", {"collections": list(collection_keys)}, item["version"])

    def add_tags(self, item, tags):
        have = {t["tag"] for t in item["data"].get("tags", [])}
        new = [{"tag": t} for t in tags if t not in have]
        if new:
            self._write("PATCH", f"/items/{item['key']}", {"tags": item["data"].get("tags", []) + new}, item["version"])


def authorize(app_name, key_file):
    """Ask the user (a dialog in the Zotero window) for a write key and store it in key_file. Blocks until the user answers."""
    z = Zotero(key_file="")
    z._ensure()
    # the user may take minutes to answer the dialog in Zotero: wait up to 25 minutes (the default socket timeout would give up after a minute)
    status, _, body = z._call("POST", BASE + "/api/local/authorize", {"appName": app_name}, timeout=1500)
    if status != 200 or not isinstance(body, dict) or "key" not in body:
        raise ZoteroError(f"authorization refused or failed: {status} {str(body)[:120]}")
    os.makedirs(os.path.dirname(os.path.abspath(key_file)), exist_ok=True)
    json.dump({"key": body["key"]}, open(key_file, "w", encoding="utf-8"))
    return key_file


def print_tree(z):
    cols = z.collections()
    kids = {}
    for k, c in cols.items():
        kids.setdefault(c["parent"], []).append(k)

    def walk(parent, depth):
        for k in sorted(kids.get(parent, []), key=lambda k: cols[k]["name"]):
            print(f"{'  ' * depth}{cols[k]['name']}  [{k}]  {cols[k]['num_items']} items")
            walk(k, depth + 1)

    walk(None, 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tree"
    if cmd == "tree":
        print_tree(Zotero())
    elif cmd == "authorize":
        print("approve the dialog in the Zotero window; waiting...")
        print("stored:", authorize("Claude Code (CS2 win-probability project)", os.environ["ZOTERO_LOCAL_KEY_FILE"]))
    else:
        sys.exit(__doc__)

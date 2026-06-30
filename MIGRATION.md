# Migrating vendored model libraries to git submodules

The model libraries under `nets/` are currently **copied into this repo** (vendored).
That is why a fresh clone is large (~236 MB `.git`) and why upstream updates are hard
to pull in. The goal of this migration is to reference upstream repositories as
**git submodules** so this repo holds only *your* code: the unified `main.py` CLI,
the configs, the conversion scripts, and the thin adapters in `adapters/`.

> ⚠️ This is a history-rewriting / structural change. It requires commits and a
> force-push, and collaborators must re-clone. Do it on a dedicated branch and
> review carefully before pushing.

---

## Important: not all libraries are cleanly convertible

There are two categories of vendored library, and they need different handling.

### 1. Clean wrapper (✅ safe to convert): `nets/ultralytics`

Only three files here are yours — they have already been extracted to
`adapters/ultralytics/` (`train.py`, `test.py`, `inference.py`). Everything else
under `nets/ultralytics/ultralytics/` is the upstream package, unmodified. `main.py`
now imports the adapters from `adapters/ultralytics`, so the vendored tree can be
replaced by a submodule (or by the PyPI package) without losing anything.

### 2. Modified forks (⚠️ converting will DROP your changes): `nets/efficientdet`, `nets/yolov7`, `nets/yolov9`

These were edited **in place**:

- `nets/efficientdet/train.py` was rewritten to expose `def train(**cfg)` — upstream
  ships a `if __name__ == "__main__"` script instead.
- `nets/yolov7` / `nets/yolov9` entry points (`train.py`, `val.py`, `detect.py`) are
  called by `main.py` with `**task_config` keyword arguments, but the upstream
  signatures are positional (e.g. `train(hyp, opt, device, ...)`). So these trees are
  modified relative to upstream too.

Replacing them with a plain submodule pointing at upstream would silently delete your
patches. For these you must first **separate your changes from upstream** (see below)
before converting.

---

## Step 0 — Prerequisites

```bash
# Work on a branch; never do this directly on main.
git checkout -b chore/submodules

# Install git-filter-repo (used to shrink history after removing vendored code).
pip install git-filter-repo
```

Record the upstream commit each vendored copy was taken from, if known. If unknown,
pin to a tag/release close to when the code was added (around early 2024 based on the
wandb run timestamps in history).

| Library       | Upstream                                            | Suggested pin |
| ------------- | --------------------------------------------------- | ------------- |
| ultralytics   | https://github.com/ultralytics/ultralytics          | a `v8.x` tag  |
| yolov7        | https://github.com/WongKinYiu/yolov7                | `main`        |
| yolov9        | https://github.com/WongKinYiu/yolov9                | `main`        |
| efficientdet  | https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch | `master` |

---

## Step 1 — Convert the clean library (ultralytics)

```bash
# Remove the vendored copy from the working tree and index.
git rm -r nets/ultralytics

# Add upstream as a submodule in its place.
git submodule add https://github.com/ultralytics/ultralytics nets/ultralytics
cd nets/ultralytics && git checkout <pinned-tag> && cd -

git add .gitmodules nets/ultralytics
git commit -m "Replace vendored ultralytics with submodule; adapters in adapters/ultralytics"
```

`main.py` already adds `nets/ultralytics` to `sys.path` and imports the adapters from
`adapters/ultralytics`, so no code change is needed for this one. Smoke-test:

```bash
pytest tests/test_smoke.py -q
python main.py --model yolov8 --mode inference   # needs deps + weights
```

---

## Step 2 — Handle the modified forks (efficientdet, yolov7, yolov9)

For each, separate your edits from upstream so they survive the submodule swap.
Recommended pattern, mirroring what was done for ultralytics:

1. Identify your changes:
   ```bash
   # Clone pristine upstream next to the repo and diff against the vendored copy.
   git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch /tmp/effdet-upstream
   diff -ru /tmp/effdet-upstream nets/efficientdet | less
   ```
2. Move your adapter logic into `adapters/<lib>/` (e.g. the `def train(**cfg)` wrapper),
   importing from the upstream package the same way `adapters/ultralytics` does.
3. Replace the vendored tree with a submodule (as in Step 1).
4. Update the corresponding branch in `import_model_functions` in `main.py` to import
   from `adapters/<lib>` instead of `nets/<lib>`.

If a fork's changes are too entangled to extract cleanly, the alternative is to keep it
vendored but maintain it as **your own fork** on GitHub and reference *that* as the
submodule — you still get a small main repo and a clear upstream lineage.

---

## Step 3 — Shrink history

Removing files from the working tree does **not** shrink `.git`; the blobs stay in
history (the large notebooks deleted earlier are still there). After all submodules are
in place:

```bash
# Inspect the biggest blobs still in history.
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob"' | sort -k3 -n -r | head -20

# Purge the vendored paths and large notebooks from ALL history.
git filter-repo --path nets/ultralytics --invert-paths \
                --path-glob 'nets/yolov7/tools/*.ipynb' --invert-paths

git gc --prune=now --aggressive
```

> `git filter-repo` rewrites every commit hash. After this, force-push and have all
> collaborators re-clone:
> ```bash
> git push --force-with-lease origin <branch>
> ```

---

## Step 4 — Update docs and clone instructions

The README clone step must use `--recurse-submodules`:

```bash
git clone --recurse-submodules https://github.com/myatthukyaw/detnets.git
# or, for an existing clone:
git submodule update --init --recursive
```

---

## Rollback

Until you force-push, everything is recoverable:

```bash
git checkout main          # abandon the chore/submodules branch
git branch -D chore/submodules
```

Keep a backup clone before running `git filter-repo`:

```bash
git clone --mirror . ../detnets-backup.git
```

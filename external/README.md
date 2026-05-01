# external/ — vendored research code

This directory holds copies of two upstream research repos. They are **vendored**: cloned, `.git` stripped, and committed into this repo as plain source. We expect to read them, port pieces, and edit them in place — there is no upstream sync expected.

| Subdirectory | Upstream | Pinned commit | License |
|---|---|---|---|
| `le-wm/` | https://github.com/lucas-maes/le-wm | `bf04d3e8c3752ac24f3692fbc5f4cf50209fa765` | MIT (© 2026 Lucas Maes) |
| `dreamerv3-torch/` | https://github.com/NM512/dreamerv3-torch | `6ef8646d807cd10ce0c88e10a7e943211e7fc44c` | MIT (© 2023 NM512) |

Both upstream `LICENSE` files are preserved verbatim alongside the source.

## Re-vendoring procedure

If you ever need a fresh snapshot:

```bash
cd external
rm -rf le-wm
git clone --depth 1 https://github.com/lucas-maes/le-wm.git le-wm
rm -rf le-wm/.git le-wm/.github

rm -rf dreamerv3-torch
git clone --depth 1 https://github.com/NM512/dreamerv3-torch.git dreamerv3-torch
rm -rf dreamerv3-torch/.git dreamerv3-torch/.github
```

Then update the SHA table above with the new commits before committing.

## Why vendor?

- **`le-wm`** — small, research-quality code; we will modify it heavily (encoder choice, SIGReg variants, probe heads). Vendoring beats submodules for this kind of editing.
- **`dreamerv3-torch`** — community port of DreamerV3; we will integrate it as a submodule of training code rather than as a pip dependency, so we can patch RSSM internals if needed.

## Why not pip / PyPI?

Neither repo is published to PyPI. Vendoring keeps the project self-contained without forcing a `git+https://` install or a Git submodule (which would require an extra `git submodule update --init` step on every clone).

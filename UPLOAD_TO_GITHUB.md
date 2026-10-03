# Uploading to GitHub and submitting to JOSS

Everything JOSS checks for is now in this folder. Three placeholders need real values first,
then it is four commands to get it online.

## 1. Replace three placeholders

| File | Find | Replace with |
|------|------|--------------|
| `paper/paper.md` | `0000-0000-0000-0000` | your ORCID (free at <https://orcid.org>) |
| `CODE_OF_CONDUCT.md` | `[ADD CONTACT EMAIL]` | an email you will read |
| `README.md` | `your-org` in the clone URL | your GitHub username |

Also check `paper/paper.md`: the affiliation reads "Independent Researcher, United Arab
Emirates". Change it if you prefer, but list only an institution that would confirm the
association. JOSS papers are permanent and public.

## 2. Create the repository on GitHub

Go to <https://github.com/new>. Name it `aurora-financial-intelligence`. Set it **public**.
Do not let GitHub add a README, licence or .gitignore, since this folder already has them.

## 3. Push

Open Terminal on your Mac:

```bash
cd "/Users/yuvrajvikrantbharadia/Desktop/Aurora - Financial Intelligence"
git init
git add .
git commit -m "Aurora v0.1.0"
git branch -M main
git remote add origin https://github.com/YOURNAME/aurora-financial-intelligence.git
git push -u origin main
```

Before committing, confirm the ignore rules are doing their job:

```bash
git status --short | wc -l        # expect about 111 files
du -ch $(git ls-files) | tail -1  # expect about 2.9 MB, not 75 MB
git ls-files | grep -i disclosure # expect no output
```

If that last command prints anything, stop and tell me.

## 4. Check the Actions tab

Two workflows run on push.

**Aurora CI** runs lint, type checks and tests. It may fail the first time on `ruff` or
`black` formatting, which is cosmetic and fixable with:

```bash
pip install ruff black isort
black aurora/ tests/ && isort aurora/ tests/ && ruff check --fix aurora/ tests/
git commit -am "Apply formatting" && git push
```

**Draft PDF** compiles `paper/paper.md` exactly as JOSS will. Open the workflow run, scroll to
Artifacts, download `paper.zip` and read the PDF. If it looks right here it will look right at
JOSS.

## 5. Release and archive

```bash
git tag -a v0.1.0 -m "Aurora v0.1.0, JOSS submission"
git push origin v0.1.0
```

On GitHub: **Releases** > **Draft a new release** > tag `v0.1.0` > publish.

The two `.tar.gz` study archives and the `RSMoE_study` folders are excluded from the repo
because they total about 72 MB. Attach them as files to this Release if you want them
available, which is where large artefacts belong.

Then for the DOI:

1. Sign in to <https://zenodo.org> with GitHub.
2. Under **GitHub** in your Zenodo settings, switch on this repository.
3. Publish a **new** release on GitHub. Zenodo only captures releases made after the switch is
   on, so if you tagged first, make it `v0.1.1`.
4. Copy the DOI Zenodo mints.
5. In Zenodo's edit view, make the title and author list match `paper/paper.md` exactly.
   Reviewers check this, and a mismatch costs a round trip.

## 6. Submit

<https://joss.theoj.org/papers/new>, sign in with GitHub, and provide the repository URL, the
version `v0.1.0`, and the Zenodo DOI.

A bot opens a public issue, runs pre-review checks and compiles your paper. An editor is
assigned, then two reviewers, usually within one to three weeks. Review is a running
conversation in that issue. Reply to every comment even where you disagree, make changes as
commits, and link them.

Typical time from submission to acceptance is one to three months, most of it waiting on
reviewers.

## What I changed in this folder

- **Fixed `pyproject.toml`.** The build backend read `setuptools.backends.legacy:build`, which
  is not a real backend. `pip install -e .` would have failed for every reviewer on the first
  command they ran. It now reads `setuptools.build_meta`.
- **Added `tests/test_non_anticipation.py`.** The README and paper both lead on the causal
  guarantee, but no test enforced it. Four tests now do, all passing, including a control that
  confirms the smoothed posterior fails the same check.
- **Added** `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `.gitignore`,
  `paper/paper.md`, `paper/paper.bib`, and `.github/workflows/draft-pdf.yml`.
- **Extended `README.md`** with a causal-integrity section, citation pointer, contributing
  pointer, and a scope-and-limitations note.

## One thing to decide

`.gitignore` excludes `Aurora_Invention_Disclosure_FINAL.pdf` from the repository, and I would
leave it that way. It is marked confidential pre-filing material, it contains the claim
skeleton and the candid assessment of which claims are weak, and publishing it to a public
repository is a public disclosure. If the provisional has not been filed, that forfeits
novelty in Europe, China and most jurisdictions outside the United States.

The three research paper PDFs are included, which is normal and fine.

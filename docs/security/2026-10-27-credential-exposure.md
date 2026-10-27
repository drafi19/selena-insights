# Credential hygiene: a leak I inherited, and what this repo does about it

While rebuilding this project I re-read the original capstone repository and found a committed
service-account credential file (`application_default_credentials.json`, type `authorized_user`,
carrying `client_id`, `client_secret` and `refresh_token`). It had been public since December 2024.

## What I did and did not do

- I did **not** try to revoke it. The credential belongs to a teammate's Google account; I am not
  its owner, I have no access to that console, and acting on someone else's account would be worse
  than the original mistake. The owner has to revoke it.
- I did **not** copy the file, its values, or the affected repository's data into this project.
- I recorded the finding and moved on, because the lesson is transferable even though the fix is
  not mine to make.

## Why removing the file from HEAD would not have helped

The token is in the git history of a public repository. Deleting the file in a later commit leaves
the blob reachable; the only real fix is revocation by the owner.

## What this repository does instead

- `.gitignore` carries `*credentials*.json`, `*.pem`, `.env*` from the first commit.
- CI runs a secret scan on every push (`.github/workflows/ci.yml`) and fails the build on a hit.
- No cloud credentials exist in this project at all: the serving image is self-contained and the
  only secret it needs is a read-only local API token, read from outside the repo.
- Wallet data and the local token never enter git: `data/raw/`, `data/processed/` and
  `reports/manual_review.csv` are ignored, and a CI check fails if any of them is ever tracked.

## Lesson

Secrets do not leak through clever attacks; they leak through a `git add -A` at 2am. The cheapest
control is a scan in CI that runs whether or not anyone remembers.

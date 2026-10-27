# Incident: GCP credentials committed to a public repo (found 2026-10-27)

- **What:** `CC/application_default_credentials.json` (type `authorized_user`) is committed in the
  public repo `drafi19/Selena-App`, a Bangkit team project. It carries `client_id`,
  `client_secret` and `refresh_token`.
- **Impact:** anyone who cloned that repo since Dec 2024 holds a refresh token for that Google
  account.
- **What I did:** recorded it here, and confirmed I am **not** the owner of that account - the
  credential belongs to a Cloud Computing teammate. I did not attempt to revoke it, did not open
  the GCP console, and did not touch any GCP resource.
- **Why it is not fixed here:** the team is out of contact, so the owner cannot be asked to revoke.
  Removing the file from HEAD would not help anyway - the token is in the git history and the
  repo is public. Revocation by the owner is the only real fix.
- **What this repo does about it:** this project does not use the capstone dataset or any GCP
  service. `selena-insights` has no GCP dependency at all.
- **Lesson:** `.gitignore` must carry `*credentials*.json`, `*.pem`, `.env*`; CI runs a secret scan
  on every push (see `.github/workflows/ci.yml`).

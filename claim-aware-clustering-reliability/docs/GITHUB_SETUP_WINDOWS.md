# Publish this repository on GitHub (Windows)

Use the repository folder extracted from the ZIP, not the ZIP file itself. Review `docs/RELEASE_CHECKLIST.md` before publishing anything publicly.

## Option A: PowerShell and Git

1. Install Git for Windows and sign in to your GitHub account. On github.com, choose **New repository**. Name it `claim-aware-clustering-reliability`. Start with **Private**. Do **not** check Add a README, `.gitignore`, or license; they already live in this folder.
2. Extract the ZIP into `C:\Projects`. Open PowerShell:

```powershell
cd C:\Projects\claim-aware-clustering-reliability

git init
git branch -M main
git add .
git status
git commit -m "Organize clustering audit experiments and evidence"

git remote add origin https://github.com/YOUR_USERNAME/claim-aware-clustering-reliability.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your own GitHub username. Git will use your configured GitHub credential manager or authentication flow; do not put a password or personal access token in the commands, repo, or README.

If Git asks you to identify the author of your commits, set it with the name and email you want displayed on GitHub:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-verified-github-email@example.com"
```

3. Confirm on GitHub that the `scripts`, `notebooks`, `evidence`, and `docs` folders are visible, and that `data`, `.env`, patient-level files, and `outputs` are not present.
4. After you add verified missing experiment sources, commit and push again:

```powershell
git add .
git status
git commit -m "Add checked experiment provenance"
git push
```

If a remote origin already exists, check `git remote -v` before adding another. If you initialized the GitHub repository with a README by mistake, do not use force-push to work around mismatched histories.

## Option B: GitHub Desktop

Install and sign in to GitHub Desktop. Select **File > Add local repository** and choose the extracted folder. If it is not yet a Git repository, GitHub Desktop can help you create one. Commit the changes, then choose **Publish repository**. Keep the repository private until the release check is complete. GitHub Desktop can handle routine commits and pushes without PowerShell.

## When the package is ready to cite

Confirm the included MIT license and author list, make the reviewed repository public, create a numbered GitHub Release, and connect the repository to Zenodo for an archival DOI. See `docs/ZENODO_RELEASE_GUIDE.md`. Insert that DOI into the manuscript only after the archive exists and you have checked the contents of the released version. Keep the GitHub project URL and the versioned archival DOI distinct.

Official instructions:
- https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop

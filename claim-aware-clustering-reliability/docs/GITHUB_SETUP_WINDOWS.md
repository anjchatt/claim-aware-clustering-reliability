# Update script names in an existing GitHub repository (Windows)

If you already pushed the previous version, you do **not** need to delete
or recreate the GitHub repository. Use the supplied migration script.

1. Extract the updated ZIP into Downloads. Find the inner `claim-aware-clustering-reliability` folder.
2. Make sure your existing Git repository has no uncommitted changes (`git status`).
3. From PowerShell, run this using the actual paths on your computer:

```powershell
cd "C:\Users\anjch\Downloads\claim-aware-clustering-reliability"
.\update_existing_repo.ps1 -RepoPath "C:\Users\anjch\Downloads\JBHI_Clustering_GitHub_Zenodo_Release_Review_20261021"
```

If execution policy blocks the script, open PowerShell and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\update_existing_repo.ps1" -RepoPath "C:\Users\anjch\Downloads\JBHI_Clustering_GitHub_Zenodo_Release_Review_20261021"
```

The script detects whether your GitHub repository contains a nested
`claim-aware-clustering-reliability` folder. It only stages changes; it
does **not** push automatically. Inspect the list before committing.

```powershell
git status
git diff --cached --stat
git commit -m "Rename analysis scripts by purpose"
git push origin main
```

This updates the current GitHub files. The old names remain visible in past
commits, which is normal and does not affect the current Zenodo release.
Do not use `git push --force` or delete the repository.

For the full old/new name list, see `docs/SCRIPT_NAMES.md`.

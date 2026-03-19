# FAQ

---

## How to Update Your Fork with the Latest Changes from the Original Repository

You can use GitHub's web interface to sync your fork:

1. Navigate to your forked repository on GitHub.
2. Look for a message that says:
   > "This branch is X commits behind [original-owner]:[branch]"
3. Click the **Sync fork** button.
4. Click **Update branch**.

Your fork will now be up to date with the latest changes from the original repository.

---

## How to Avoid Sync Conflicts?

Sync conflicts usually happen when you make changes to files or folders that are also updated in the upstream repository. Here are some tips to avoid conflicts:

- **Work only in the `student` folder:**
  - Keep all your work, assignments, and custom files inside the `student` directory. Avoid editing files or folders outside of it.
- **Pull upstream changes regularly:**
  - Before starting new work, sync your fork with the upstream repository to get the latest updates.
- **Avoid renaming or deleting shared files:**
  - Do not rename, move, or delete files that exist in the upstream repository, especially outside the `student` folder.
- **Resolve conflicts promptly:**
  - If a conflict does occur, follow GitHub’s instructions to resolve it, or ask your instructor for help.

> **Tip:** If you are unsure about a change, make a backup of your work before syncing.

---

## Best Practices to Keep Your Repo Updated with the Instructor's

To ensure your repository stays up to date with the instructor’s (upstream) repository, follow these best practices:

1. **Sync your fork on GitHub regularly:**  
   Use the **Sync fork** button on GitHub to pull in the latest changes from the instructor’s repository.

2. **Update your local repository:**  
   After syncing your fork on GitHub, pull the changes to your local machine or GCP instance:
   ```sh
   git pull origin main
   ```
 

3. **Update dependencies:**  
   If there are changes to dependencies, run:
   ```sh
   poetry install
   ```
   This will install any new or updated packages.

4. **Work only in the `student` folder:**  
   Keep your work inside the `student` directory to minimize merge conflicts.

5. **Sync before starting new work:**  
   Always update your repo before you start a new homework.

---

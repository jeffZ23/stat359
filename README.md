# STAT359 Course Repository

Welcome to the course repository for STAT359! This repo is designed to support your learning and project work throughout the course. Please read this document carefully to get started and make the most of the provided resources.

---

## 📁 Repository Structure

- `instructor/` — Official course materials, homework assignments, and resources (provided by instructors).
- `student/` — Your own work: homework solutions, project code, and notes. **All your submissions should go here.**
- `.md` files — Setup and environment guides to help you get started on your laptop or on Google Cloud Platform (GCP).

---

## 🚀 Getting Started

### 1. Fork and Set Up Your Repo
- **Fork** this repository to your own GitHub account.
- **Add the instructor and TA as collaborators** to your forked repo.
- **Clone** your fork to your local machine.
- **Sync regularly** with the upstream (instructor's) repo to get updates:
  ```sh
  git remote add upstream <instructor-repo-url>
  git fetch upstream
  git merge upstream/main
  ```

### 2. Set Up Your Development Environment
- **Follow the setup guides:**
  - [`venv_setup.md`](./venv_setup.md): Set up a Python virtual environment using Poetry.
  

- **Recommended IDE:** [Visual Studio Code](https://code.visualstudio.com/)

---

## 💻 Local vs. Cloud Development

- **Local Development:**
  - Do most of your coding, debugging, and preliminary testing on your laptop.
  - If your laptop has a suitable GPU (NVIDIA or Apple M-series), you can complete most assignments locally.

- **Cloud (GCP) Development:**
  - For large models or heavy computation, use a GCP GPU instance.
  - Two ways to use GCP:
    1. **Code Locally, Run Remotely:**
       - Push your code to GitHub, then SSH into your GCP instance and run scripts from the command line.
    2. **VS Code Remote SSH:**
       - [`install_gcp.md`](./install_gcp.md): Set up a GCP GPU instance (if needed).
       - [`dev_gcp.md`](./dev_gcp.md): Develop on GCP using VS Code Remote SSH.
       - Use the Remote - SSH extension to develop directly on the GCP instance with the full power of VS Code.

---

## 📝 Homework & Collaboration

- **Instructor posts materials and assignments** in the `instructor/` folder.
- **Students work in the `student/` folder** and submit assignments there.
- **Keep your repo up to date** by syncing with the upstream instructor repo.

---

## 📚 Setup Guides

- [Python Virtual Environment Setup (`venv_setup.md`)](./venv_setup.md)
- [GCP GPU Instance Setup (`install_gcp.md`)](./install_gcp.md)
- [GCP Development with VS Code (`dev_gcp.md`)](./dev_gcp.md)
- [Frequently Asked Questions (`faq.md`)](./faq.md)

---

## ❓ Need Help?
- If you have questions, please reach out to the instructor or TA via GitHub Issues or email.

Happy coding and learning!

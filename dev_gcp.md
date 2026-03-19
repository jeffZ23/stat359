# GCP Development Environment Setup

Before you begin, make sure you have completed all steps in [`install_gcp.md`](./install_gcp.md).

## 1. Install Remote SSH for VS Code
- Install the [Remote - SSH extension](https://code.visualstudio.com/docs/remote/ssh) in VS Code.
- **Note:** You only need to install the extension. Do **not** configure SSH through the doc yet.

## 2. Configure SSH for GCP Instance
- Open your terminal and run:

  ```sh
  gcloud compute config-ssh
  ```
- This command will generate the necessary SSH configuration for your GCP instance.

## 3. Connect to GCP Instance via VS Code
- Use the Remote-SSH extension to connect.
- You should see your GPU instance listed in the 'Connect to...' options.

## 4. Clone Your GitHub Repository
- Once connected, clone your GitHub repository to the GCP instance.
- Open the project folder in VS Code.

## 5. Set Up the Virtual Environment
- Follow the instructions in [`venv_setup.md`](./venv_setup.md) to set up your Python virtual environment.

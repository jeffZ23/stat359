# Python Virtual Environment Setup with Poetry

This guide will help you set up a Python virtual environment for your project using [Poetry](https://python-poetry.org/). Follow these steps to ensure a smooth development experience in VS Code.

---

## 1. Install Poetry (One-Time Setup)

### Before You Start: Turn Off COnda's `(base)` Environment

Before getting started, please note: when you open a new terminal, your command prompt may start with **`(base)`**.  

This indicates that the **Conda base environment** is being automatically activated. If you previously installed Conda on your laptop, this auto-activation can cause conflicts with system or course-specific packages.  

Since we will **not** be using Conda to manage the environment for this course, and to ensure proper isolation, please disable Conda’s automatic base activation permanently by running:  

```bash
conda config --set auto_activate_base false
```

### Install Python

Poetry is a Python package manager, but it does not install Python itself. You must have Python installed on your system before using Poetry.

For this course, you will need Python 3.12 to ensure compatibility with the required packages. If you don't have either installed, please follw the instructions below to install it.

#### For Windows:

1. **Download Python from the official website:**
   - Visit [python.org/downloads](https://www.python.org/downloads/)
   - Download Python 3.12.x (latest stable version recommended)
   - Choose the "Windows installer (64-bit)" for most modern computers

2. **Run the installer:**
   - **IMPORTANT:** Check the box "Add Python to PATH" during installation
   - Choose "Install Now" for default installation
   - If prompted, allow the installer to disable the path length limit

3. **Verify the installation:**
   ```powershell
   python --version
   ```
   This should display Python 3.12.x 

#### For macOS:

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python using Homebrew:**
   ```bash
   # Install Python 3.12
   brew install python@3.12

   ```

3. **Verify the installation:**
   ```bash
   python3 --version
   ```
   This should display Python 3.12.x

4. **Important for macOS users - Check Python architecture:**
   
   **Note:** PyTorch has discontinued support for x86_64 (amd64/Intel) architecture on macOS since version 2.3. If you're using an Apple Silicon Mac (M1, M2, M3, or newer), you need to ensure your Python installation is ARM64 (Apple Silicon) compatible.

   **Check your Python architecture:**
   ```bash
   python3 -c "import platform; print(f'Architecture: {platform.machine()}, Platform: {platform.platform()}')"
   ```

   **Expected output for Apple Silicon Macs:**
   - Architecture should show `arm64`
   - Platform should include `arm64`

   **If you see `x86_64` or `amd64`:**
   You're running Intel Python on Apple Silicon, which will cause PyTorch compatibility issues. You need to reinstall Python for ARM64:

   ```bash
   # Uninstall current Python
   brew uninstall --ignore-dependencies python@3.12


   # Reinstall Python for Apple Silicon
   arch -arm64 brew install python@3.12


   # Verify the fix
   python3 -c "import platform; print(f'Architecture: {platform.machine()}')"
   ```

   **For Intel Macs (older models):**
   If you're using an Intel Mac, you can continue with x86_64 Python, but note that you may need to use older PyTorch versions or CPU-only builds.



### Install Poetry Globally

**Important:** While you *can* use `pip install poetry` to install Poetry, this is **not recommended** because it will install Poetry into your current Python environment, which can cause dependency conflicts and version management issues.

Instead, please follow the official installation method below to install Poetry globally on your system, which keeps it isolated from your project dependencies:

#### For macOS/Linux:

1. Run the automatic installation script:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. Ensure the Poetry executable is in your `PATH`. Add the following line to your `~/.zshrc` or `~/.bash_profile` (above any `pyenv` lines):

    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

    > **Note:** You may need to restart your terminal or run `source ~/.zshrc` for changes to take effect.

3. Configure Poetry to create virtual environments inside your project root:

    ```bash
    poetry config virtualenvs.in-project true
    ```

#### For Windows (PowerShell):

1. Run the following command:

    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

2. Make sure the Poetry executable is in your `PATH` (see the [official docs](https://python-poetry.org/docs/#installing-with-the-official-installer)).

  > **Note:** You may need to restart your terminal or VS code for changes to take effect

#### Verify Installation:

1. Run the following command:
   

    ```powershell
    poetry --version
    ```
    
    If it prints the installed version, you’re ready to proceed!
---

## 2. Set Up Your Project Environment

1. **Clone this GitHub repository** (if you haven't already):

    ```bash
    git clone <this-repo-url>
    cd <this-project-folder>
    ```
    > *If `git` is not installed, you may need to run `sudo apt install git` (Linux) or use Homebrew on macOS.*

2. **Configure poetry to use `.venv` in project directionary** 

    ```bash
    poetry config virtualenvs.in-project true
    ```

3. **Configure Poetry to use the correct Python version**

   If you have multiple Python versions on your system, Poetry might have trouble selecting the correct one. To ensure your project uses Python 3.12, you need to explicitly specify which Python interpreter Poetry should use.

   First, locate your Python installation path, then configure Poetry to use it:

   ```bash
   poetry env use /path/to/your/python/executable
   ```

   **Examples:**

   - **Windows:** If you have Python 3.12 installed at:
     - `C:\Users\lsi8012\AppData\Local\Programs\Python\Python312\python.exe`
     
     To use Python 3.12, run:
     ```powershell
     poetry env use C:\Users\lsi8012\AppData\Local\Programs\Python\Python312\python.exe
     ```

   - **macOS/Linux:** To use Python 3.12, you might run:
     ```bash
     poetry env use python3.12
     # or with full path
     poetry env use /usr/bin/python3.12
     ```

   **To find your Python installation paths:**
   - **Windows:** Check `C:\Users\[username]\AppData\Local\Programs\Python\` or use `where python` in Command Prompt
   - **macOS:** Use `which python3.12` or `brew --prefix python@3.12`
   - **Linux:** Use `which python3.12` or `whereis python3.12`
4. **Install project dependencies** (from the folder containing `pyproject.toml`):

    ```bash
    poetry install --no-root
    ```

5. **Verify the Python interpreter**:

    ```bash
    poetry run which python
    ```
    This should point to the `.venv` folder under your project root.

6. **Restart VS Code** and select the Python interpreter from `.venv` for the best experience.

7. **Test your torch environment**:

    ```bash
    poetry run python test_torch_env.py
    ```
    If this runs without errors, your environment is set up correctly.

---

You are now ready to use VS Code for development!


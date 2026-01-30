# Comprehensive Setup Guide: FDA Agent Application

This guide provides step-by-step instructions to run the **Clause FDA Agent** application on both **Windows 11** and **MacBook Pro**.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Windows 11 Setup](#windows-11-setup)
4. [MacBook Pro Setup](#macbook-pro-setup)
5. [Configuration & API Keys](#configuration--api-keys)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The **Clause FDA Agent** is a multi-model, agentic Streamlit application designed to assist FDA/TFDA reviewers and consultants in medical device document processing and review workflows. It supports:

- **Multiple AI Models:** OpenAI (GPT-4o, GPT-4.1), Google Gemini (2.5), Anthropic Claude (3.5), and Grok
- **Multi-language Interface:** English and Traditional Chinese
- **Key Features:**
  - 510(k) Intelligence Analysis
  - PDF-to-Markdown Conversion
  - Document Review Pipelines
  - Agent Configuration Management
  - Dashboard with Usage Analytics
  - 20 Artistic Background Themes

### Tech Stack

- **Framework:** Streamlit
- **Language:** Python 3.10+
- **Key Dependencies:**
  - OpenAI, Google Generative AI, Anthropic, HTTPx
  - pandas, numpy, altair (for charts)
  - pypdf, python-docx, reportlab (for document handling)
  - PyYAML (for agent configuration)

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 11 / macOS 12+ | Windows 11 / macOS 13+ |
| **RAM** | 4 GB | 8+ GB |
| **Disk Space** | 2 GB | 5+ GB |
| **Internet** | Required | Required (for API calls) |

### Software Requirements

- **Python 3.10 or higher** (3.11+ recommended)
- **pip** (Python package manager)
- **Git** (optional, for cloning the repo)

### API Keys Required

You'll need at least one of the following API keys:

1. **OpenAI API Key** - [Get it here](https://platform.openai.com/api-keys)
2. **Google Generative AI API Key** - [Get it here](https://aistudio.google.com/app/apikey)
3. **Anthropic API Key** - [Get it here](https://console.anthropic.com/api/keys)
4. **Grok (xAI) API Key** - [Get it here](https://console.x.ai)

---

## Windows 11 Setup

### Step 1: Install Python 3.10+

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT:** Check the box **"Add Python to PATH"** during installation
4. Click "Install Now"

**Verify Installation:**
```powershell
python --version
pip --version
```

### Step 2: Navigate to Project Directory

```powershell
# Example path - adjust to your actual project location
cd C:\path\to\project\1-Hannah012926
```

### Step 3: Create a Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you encounter execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

After activation, your command prompt should show `(venv)` prefix.

### Step 4: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 5: Install Project Dependencies

```powershell
pip install -r requirements.txt
```

**Expected Output:** All packages should install successfully (no error messages at the end).

### Step 6: Verify Installation

```powershell
python -c "import streamlit; import openai; import google.generativeai; import anthropic; print('All imports successful!')"
```

### Step 7: Set Environment Variables (Optional but Recommended)

For security, store API keys as environment variables instead of entering them in the UI:

```powershell
# Set environment variables in PowerShell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your-key-here', 'User')
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'your-key-here', 'User')
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'your-key-here', 'User')
[System.Environment]::SetEnvironmentVariable('GROK_API_KEY', 'your-key-here', 'User')

# Restart PowerShell or your IDE for changes to take effect
```

---

## MacBook Pro Setup

### Step 1: Install Python 3.10+

**Option A: Using Homebrew (Recommended)**

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

**Option B: Direct Download**

1. Download from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. Follow the prompts

**Verify Installation:**
```bash
python3 --version
pip3 --version
```

### Step 2: Navigate to Project Directory

```bash
cd /path/to/project/1-Hannah012926
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

After activation, your terminal prompt should show `(venv)` prefix.

### Step 4: Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

### Step 5: Install Project Dependencies

```bash
pip install -r requirements.txt
```

**Expected Output:** All packages should install successfully (no error messages at the end).

### Step 6: Verify Installation

```bash
python3 -c "import streamlit; import openai; import google.generativeai; import anthropic; print('All imports successful!')"
```

### Step 7: Set Environment Variables (Recommended)

For security, store API keys as environment variables:

```bash
# Add to ~/.zprofile or ~/.bash_profile (depending on your shell)
export OPENAI_API_KEY='your-key-here'
export GOOGLE_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'
export GROK_API_KEY='your-key-here'

# Reload shell configuration
source ~/.zprofile  # or ~/.bash_profile
```

---

## Configuration & API Keys

### How API Keys are Managed

The application checks for API keys in this order:

1. **Environment Variables** (Most Secure)
   - If set, the Streamlit sidebar shows "from environment"
   - Keys are **not displayed** in the UI for security

2. **Sidebar Input** (Session Memory Only)
   - If environment variables are not set, you can paste API keys in the Streamlit sidebar
   - Keys are stored **only in current session memory** (not saved to disk)
   - Keys are cleared when you close the application

### Best Practices for API Key Security

✅ **DO:**
- Store API keys in environment variables
- Use separate keys for development and production
- Rotate keys regularly
- Keep `.env` files in `.gitignore` if using local files

❌ **DON'T:**
- Commit API keys to Git repositories
- Share API keys in emails or chat
- Hardcode keys in the application source code
- Save keys in browser history or autocomplete

### Configuration Files

The application uses two configuration files:

1. **agents.yaml**
   - Defines all AI agents and their configurations
   - Includes system prompts, temperature, and max_tokens settings
   - Modify this file to customize agent behavior

2. **app.py**
   - Main Streamlit application file
   - Contains UI logic, multi-language support, and theme settings

---

## Running the Application

### Windows 11

1. **Activate Virtual Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Run Streamlit:**
   ```powershell
   streamlit run app.py
   ```

3. **Access the Application:**
   - Open your browser and go to: `http://localhost:8501`
   - A new browser tab should automatically open

### MacBook Pro

1. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application:**
   - Open your browser and go to: `http://localhost:8501`
   - A new browser tab should automatically open

### Expected Startup Output

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://XXX.XXX.X.XXX:8501
```

---

## Application Features & Usage

### Main Tabs

1. **Dashboard**
   - View usage statistics and analytics
   - Monitor API calls and token consumption
   - Track model usage distribution

2. **TW Premarket**
   - TFDA medical device registration assistance
   - Document analysis for devices requiring pre-approval

3. **510(k) Intelligence**
   - FDA 510(k) document analysis
   - Predicate device comparison
   - Risk analysis and technical summaries

4. **PDF → Markdown**
   - Convert PDF documents to structured Markdown
   - Preserve formatting and tables
   - Extract key sections automatically

5. **510(k) Review Pipeline**
   - Manage complete review workflows
   - Generate checklists and reports
   - Track review progress

6. **Note Keeper & Magics**
   - Create and manage notes
   - Access special features and utilities

7. **Agents Config**
   - Configure individual agents
   - Adjust AI model parameters
   - Customize system prompts

### UI Features

- **Language Toggle:** Switch between English and Traditional Chinese (中文)
- **Theme Selection:** Choose from 20 artistic painter-inspired backgrounds
- **Jackpot:** Randomly select a theme
- **Dark/Light Mode:** Toggle between themes

---

## Troubleshooting

### Common Issues

#### 1. "Python is not recognized as an internal or external command"

**Windows:**
- Python wasn't added to PATH during installation
- **Solution:** Reinstall Python and check "Add Python to PATH"
- Or manually add Python to PATH:
  1. Open Environment Variables (search "Environment" in Windows)
  2. Click "Edit the system environment variables"
  3. Click "Environment Variables"
  4. Under "System variables," click "Path" → "Edit"
  5. Add: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311`
  6. Click OK and restart your terminal

#### 2. "No module named 'streamlit'" or other import errors

**Windows & Mac:**
```powershell
# Windows
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Mac
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. "Permission denied" when running streamlit (Mac)

```bash
chmod +x app.py
streamlit run app.py
```

#### 4. API Key Not Working / "Invalid API Key" Error

- Verify the API key is correct and hasn't expired
- Check that you have API credits/quota available
- Ensure the environment variable name matches:
  - `OPENAI_API_KEY` (OpenAI)
  - `GOOGLE_API_KEY` (Gemini)
  - `ANTHROPIC_API_KEY` (Claude)
  - `GROK_API_KEY` (Grok)
- Restart the Streamlit app after setting environment variables

#### 5. Port 8501 Already in Use

**Windows:**
```powershell
netstat -ano | findstr :8501
taskkill /PID <PID> /F
streamlit run app.py --logger.level=debug
```

**Mac:**
```bash
lsof -i :8501
kill -9 <PID>
streamlit run app.py --logger.level=debug
```

Or run on a different port:
```bash
streamlit run app.py --server.port 8502
```

#### 6. Memory/Performance Issues

If the app runs slowly or crashes:

1. **Reduce max_tokens:** Modify in the UI or agents.yaml
2. **Restart Streamlit:** Close the app and rerun
3. **Check System Resources:**
   - Windows: Task Manager (Ctrl+Shift+Esc)
   - Mac: Activity Monitor (Cmd+Space → "Activity Monitor")
4. **Upgrade Dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

#### 7. PDF Processing Issues

- Ensure pypdf is installed: `pip install pypdf`
- Some encrypted PDFs may not process correctly
- Try converting PDF to text first and paste content

#### 8. "ModuleNotFoundError: No module named 'docx'"

```bash
pip install python-docx
```

---

## Advanced Configuration

### Customizing Agents

Edit `agents.yaml` to modify:

- **temperature:** Lower (0-0.5) for deterministic outputs, higher (0.7-1.0) for creative
- **max_tokens:** Adjust token limits per agent (default 12000, max 120000)
- **system_prompt:** Customize the AI instructions
- **model:** Change the AI model for specific agents

### Environment-Specific Setup

For **development:**
```bash
pip install -r requirements.txt
pip install -e .  # For editable installations
```

For **production:**
```bash
pip install -r requirements.txt
streamlit run app.py --logger.level=error
```

---

## Version Information

- **Python:** 3.10+
- **Streamlit:** 1.51.0+
- **Key Dependencies:**
  - openai
  - google-generativeai
  - anthropic
  - pyyaml
  - pandas
  - numpy
  - pypdf
  - altair
  - httpx
  - python-docx
  - reportlab

---

## Getting Help

### Documentation & Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Streamlit Community:** https://discuss.streamlit.io
- **API Documentation:**
  - OpenAI: https://platform.openai.com/docs
  - Gemini: https://ai.google.dev/docs
  - Anthropic: https://docs.anthropic.com
  - Grok: https://docs.x.ai

### Debugging

Enable debug mode:

```bash
streamlit run app.py --logger.level=debug
```

Check Streamlit version:
```bash
streamlit --version
```

---

## Quick Reference

### Windows 11 Cheat Sheet

```powershell
# Initial setup (one-time)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Running the app (every time)
.\venv\Scripts\Activate.ps1
streamlit run app.py

# Deactivate environment
deactivate
```

### MacBook Pro Cheat Sheet

```bash
# Initial setup (one-time)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Running the app (every time)
source venv/bin/activate
streamlit run app.py

# Deactivate environment
deactivate
```

---

## Support & Feedback

For issues, suggestions, or contributions, please refer to the project's issue tracker or documentation.

**Last Updated:** January 29, 2026

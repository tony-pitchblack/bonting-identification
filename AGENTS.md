# General instructions regarding environment
If micromamba is not present, install with
```bash
export BIN_FOLDER="${HOME}/.local/bin" \
       INIT_YES=yes \
       CONDA_FORGE_YES=yes \
       PREFIX_LOCATION="${HOME}/micromamba"
"${SHELL}" <(curl -L micro.mamba.pm/install.sh) < /dev/null
```

Before launching any of the .sh or .py scripts activate env with
```bash
micromamba activate bonting-id
```

If git submodules are empty, update them and add their dependencies to micromamba env with
```bash
./install_submodules.sh
```

# General agentic pipeline
1. Make sure to execute script manage_data/download_data_from_hf.sh first to download data.
2. Then execute any prompted scripts.
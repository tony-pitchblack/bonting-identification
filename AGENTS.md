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

Make sure to download data before processing any requests or running any scripts requested by user:
```bash
./manage_data/download_data_from_hf.sh
```
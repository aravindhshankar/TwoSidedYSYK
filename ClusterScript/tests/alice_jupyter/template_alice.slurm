#!/bin/bash

#SBATCH --job-name=jupyter_notebook
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu-short
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

unset XDG_RUNTIME_DIR
module load ALICE/default
module load JupyterLab

echo "Running the notebook on $(hostname)"
IPADDR=$(hostname -i)

port=8989

SALT="$(head /dev/urandom | tr -dc 'A-Z-a-z-0-9{}[]=+.:-_' | head -c 16;echo;)"
#password="$(head /dev/urandom | tr -dc 'A-Z-a-z-0-9{}[]=+.:-_' | head -c 16;echo;)"
password=""
PASSWORD_SHA="$(echo -n "${password}${SALT}" | openssl dgst -sha256 | awk '{print $NF}')"
echo "-------------------------"
echo "Log in using the password: $password"
echo "-------------------------"

# the jupyter server config file
export CONFIG_FILE="${PWD}/config.py"

(
umask 077
cat > "${CONFIG_FILE}" << EOL
c.ServerApp.ip = '${IPADDR}'
c.ServerApp.port = ${port}
c.ServerApp.port_retries = 1
c.ServerApp.password = u'sha256:${SALT}:${PASSWORD_SHA}'
c.ServerApp.base_url = '/node/$(hostname)/${port}/'
c.ServerApp.open_browser = False
c.ServerApp.allow_origin = '*'
c.ServerApp.root_dir = '${HOME}'
c.ServerApp.disable_check_xsrf = True
EOL
)

echo "#### Starting the JupyterLab server" 
set -x
jupyter lab --config="${CONFIG_FILE}"
echo "#### Terminated JupyterLab server. Done"

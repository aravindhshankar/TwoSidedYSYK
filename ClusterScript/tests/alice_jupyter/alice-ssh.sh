#!/bin/bash

# Default SSH Host
SSH_HOST="alice-notebook"

# Parse arguments for -N (node_name) and -P (port_name)
while getopts "N:P:" opt; do
  case $opt in
    N) NODE_NAME="$OPTARG" ;;
    P) PORT_NAME="$OPTARG" ;;
    *) echo "Usage: $0 -N <node_name> -P <port_name>" >&2; exit 1 ;;
  esac
done

# Ensure both node_name and port_name are provided
if [ -z "$NODE_NAME" ] || [ -z "$PORT_NAME" ]; then
  echo "Error: You must specify a node name using -N <node_name> and a port name using -P <port_name>"
  exit 1
fi

# Echo the dynamic URL
echo "http://localhost:${PORT_NAME}/node/${NODE_NAME}/${PORT_NAME}/lab"

# Build the ssh command dynamically
ssh -o "LocalForward ${PORT_NAME} ${NODE_NAME}:${PORT_NAME}" "$SSH_HOST"

#NOTE example nodename : node020
#example port name : 8989
#
#ADD THE FOLLOWING TO the ~/.ssh/config file : 
#Host alice-notebook
#   HostName login1.alice.universiteitleiden.nl
#   ProxyJump shankaras1@ssh-gw.alice.universiteitleiden.nl:22
#   User shankaras1
#   ServerAliveInterval 60
#
#

#!/bin/bash

OS=$(uname)

# Check the OS
case "$OS" in
    "Linux")
        # Report that the OS is Linux
        DEFAULT_SHELL=$(basename $(getent passwd $LOGNAME | cut -d: -f7))
        echo "$OS"
        ;;
        
    "Darwin")
        # Report that the OS is macOS
        DEFAULT_SHELL=$(basename $(dscl . -read /Users/$USER UserShell | awk '{print $2}'))
        echo "$OS"
        ;;
        
    *)
        echo "Unsupported OS."
        ;;
esac

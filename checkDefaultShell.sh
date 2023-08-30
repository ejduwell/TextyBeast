#!/bin/bash

OS=$(uname)

# Check the OS
case "$OS" in
    "Linux")
        # Get the default shell in Linux
        DEFAULT_SHELL=$(basename $(getent passwd $LOGNAME | cut -d: -f7))
        echo "$DEFAULT_SHELL"
        ;;
        
    "Darwin")
        # Get the default shell in macOS
        DEFAULT_SHELL=$(basename $(dscl . -read /Users/$USER UserShell | awk '{print $2}'))
        echo "$DEFAULT_SHELL"
        ;;
        
    *)
        echo "Unsupported OS."
        ;;
esac





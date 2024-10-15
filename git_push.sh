#!/bin/bash

# git init
# git remote add origin https://github.com/dasnil500/end-to-end-v2.0.git
# git config user.name "Nil"
# git config user.email "dasnil500@gmail.com"
# password: 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



# git add .
# git commit -m "models are excluded and incomplete results are there"
# git push -u origin master

# Check if a commit message is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <commit message>"
    exit 1
fi

# Add all changes
git add .

# Commit changes with the provided message
git commit -m "$1"

# Push changes to the remote repository
git push -u origin main
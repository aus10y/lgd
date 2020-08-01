#!/usr/bin/env bash

# What is the algorithm for completion?
# For now we'll only try to provide completions when the last command arg is -t
#


_lgd_completions() {
    args=()

    # Filter for lgd command line args
    for word in ${COMP_WORDS[@]}
    do
        if [[ $word == -* ]]; then
            args+=($word)
        fi
    done

    # Check if there are no command line args
    if [ ${#args[@]} -eq 0 ]; then
        return
    fi

    # Check if the "current" arg is -t/--tags
    if [[ ${args[-1]} == "-t" ]] || [[ ${args[-1]} == "--tags" ]]; then
        # Parse out the tags column, skipping the header row
        tags_raw=$(lgd -T | awk -e '{ if (NR > 1) { print $1 } }')
        
        # Strip empty lines and terminal color codes
        tags=$(echo $tags_raw | sed -e '/^$/ d' -e 's/\x1b\[[0-9;]*m//g')

        # Provide the tags
        COMPREPLY=($(compgen -W "$tags" -- "${COMP_WORDS[COMP_CWORD]}"))
    fi
}

# Register the completion function
complete -F _lgd_completions lgd

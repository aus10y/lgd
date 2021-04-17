#!/usr/bin/env bash

lgd_deactivate () {
    PS1="$_OLD_VIRTUAL_PS1"
    export PS1
    unset _OLD_VIRTUAL_PS1

    unalias lgd
    unset LGD_DB_PATH
}

_OLD_VIRTUAL_PS1="${PS1-}"
PS1="(lgd) $PS1"
export PS1

# Alias lgd to ./src/lgd.py
alias lgd="./src/lgd.py"

# Point to test database
export LGD_DB_PATH="~/.lgd/test.db"


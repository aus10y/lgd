LGD_SRC_DIR = ./src/app/
LGD_PY = ./bin/lgd
LGD = ~/.local/bin/lgd
LGD_TESTS = ./tests/test_lgd.py
LGD_DB = ~/.lgd/logs.db
BACKUP_NOTES = ~/.lgd/backup_notes.csv
BACKUP_TAGS = ~/.lgd/backup_tags.csv
COMPLETIONS = ./scripts/lgd_completion.sh
COMPLETIONS_ETC = /etc/bash_completion.d/lgd_completion.sh

.PHONY: help install uninstall update test backup build

help:
	@echo "Options:"
	@echo "- make install"
	@echo "- make uninstall"
	@echo "- make install-completions (requires sudo)"
	@echo "- make uninstall-completions (requires sudo)"
	@echo "- make update"
	@echo "- make test"
	@echo "- make backup"
	@echo "- make build"

install:
	@cp ${LGD_PY} ${LGD}
	@chmod +x ${LGD}
	@echo "Copied '${LGD_PY}' to '${LGD}', and set as executable."

uninstall:
	@rm -f ${LGD}
	@echo "Removed ${LGD}"

install-completions:
	cp ${COMPLETIONS} ${COMPLETIONS_ETC}

uninstall-completions:
	rm ${COMPLETIONS_ETC}

${LGD}: ${LGD_PY}
	@cp ${LGD_PY} ${LGD}
	@# @chmod +x ${LGD}
	@rm ~/.lgd/logs.db
	@echo "Restoring..."
	@lgd --note-import ${BACKUP_NOTES}
	@lgd --tag-import ${BACKUP_TAGS}
	@touch ${BACKUP_NOTES} ${BACKUP_TAGS}

update: backup ${LGD}
	@echo "Up-to-date"

test:
	@${LGD_TESTS}

${BACKUP_NOTES}: ${LGD_DB}
	@lgd --note-export ${BACKUP_NOTES}

${BACKUP_TAGS}: ${LGD_DB}
	@lgd --tag-export ${BACKUP_TAGS}

backup: ${BACKUP_NOTES} ${BACKUP_TAGS}

# ${LGD_PY}: ${LGD_SRC_DIR}
# 	@python3 -m zipapp src/app -o bin/lgd -p "/usr/bin/env python3"
# 	@echo "Packaged lgd app"

# build: ${LGD_PY}

build:
	@zip bin/lgd.zip lgd/* __main__.py
	@echo '#!/usr/bin/env python3' | cat - bin/lgd.zip > bin/lgd
	@chmod 755 bin/lgd
	@rm bin/lgd.zip
	@echo "Packaged lgd app"

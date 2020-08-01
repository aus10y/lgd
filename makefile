LGD_PY = ./src/lgd.py
LGD = ~/.local/bin/lgd
LGD_TESTS = ./src/test_lgd.py
LGD_DB = ~/.lgd/logs.db
BACKUP_NOTES = ~/.lgd/backup_notes.csv
BACKUP_TAGS = ~/.lgd/backup_tags.csv
COMPLETIONS = ./scripts/lgd_completion.sh
COMPLETIONS_ETC = /etc/bash_completion.d/lgd_completion.sh

.PHONY: help install uninstall update test backup

help:
	@echo "Options:"
	@echo "- make install"
	@echo "- make uninstall"
	@echo "- make update"
	@echo "- make test"
	@echo "- make backup"

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
	@chmod +x ${LGD}
	@rm ~/.lgd/logs.db
	@echo "Restoring..."
	@lgd -NI ${BACKUP_NOTES}
	@lgd -TI ${BACKUP_TAGS}
	@touch ${BACKUP_NOTES} ${BACKUP_TAGS}

update: backup ${LGD}
	@echo "Up-to-date"

test:
	@${LGD_TESTS}

${BACKUP_NOTES}: ${LGD_DB}
	@lgd -NE ${BACKUP_NOTES}

${BACKUP_TAGS}: ${LGD_DB}
	@lgd -TE ${BACKUP_TAGS}

backup: ${BACKUP_NOTES} ${BACKUP_TAGS}

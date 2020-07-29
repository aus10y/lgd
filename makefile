LGD_PY = lgd.py
LGD = lgd
LGD_TESTS = test_lgd.py
LGD_DB = ~/.lgd/logs.db
BACKUP_NOTES = ~/.lgd/backup_notes.csv
BACKUP_TAGS = ~/.lgd/backup_tags.csv

.PHONY: help install uninstall test backup

help:
	@echo "Options:"
	@echo "- make install"
	@echo "- make uninstall"
	@echo "- make test"
	@echo "- make backup"

install:
	@cp ./src/${LGD_PY} ~/.local/bin/${LGD}
	@chmod +x ~/.local/bin/${LGD}
	@echo "Copied '${LGD_PY}' to ~/.local/bin as '${LGD}', and set as executable."

uninstall:
	@rm -f ~/.local/bin/${LGD}
	@echo "Removed ${lgd} from ~/.local/bin/"

test:
	@./src/${LGD_TESTS}

${BACKUP_NOTES}: ${LGD_DB}
	@lgd -NE ${BACKUP_NOTES}

${BACKUP_TAGS}: ${LGD_DB}
	@lgd -TE ${BACKUP_TAGS}

backup: ${BACKUP_NOTES} ${BACKUP_TAGS}
	@echo "Done"

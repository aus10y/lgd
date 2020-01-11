#!/usr/bin/env python3
import argparse
import cmd
import difflib
import sys
import sqlite3
import tempfile
import os

from pathlib import Path
from subprocess import call


EDITOR = os.environ.get('EDITOR','vim')


# Argparse stuff
parser = argparse.ArgumentParser(
    description="A flexible knowledge store."
)
parser.add_argument(
    '-s', '--show', action='append', nargs='*', dest='tags',
    help=(
        "Show messages.\n"
        " Filter messages by adding one or more tags separated by spaces.\n"
        " Matching messages must contain all given tags.\n"
        " Ex. `-s foo`, `-s foo bar`.\n"
        " Additional flag usage will OR the tag groups together.\n"
        " Ex. `-s foo bar -s baz`.\n"
    )
)
parser.add_argument(
    '-o', '--output', action="store", type=str,
    help="Specify an output file, or leave blank to output to stdio."
)
parser.add_argument(
    '-D', '--delete', action='store', type=int,
    help="Delete the message with the given ID."
)
# TODO: Implement the output file redirection.


#-----------------------------------------------------------------------------
# Path

LGD_PATH = Path.home() / Path('.lgd')
def dir_setup():
    # If our dir doesn't exist, create it.
    LGD_PATH.mkdir(mode=0o770, exist_ok=True)

#-----------------------------------------------------------------------------
# Database

DB_NAME = 'logs.db'
DB_PATH = LGD_PATH / Path(DB_NAME)
DB_USER_VERSION = 1

# Column Names
ID = 'id'
LOG = 'log'
MSG = 'msg'
TAG = 'tag'
TAGS = 'tags'
CREATED_AT = 'created_at'

CREATE_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at int NOT NULL,
    msg TEXT NOT NULL
);
"""
CREATE_TAGS_TABLE = """
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag TEXT NOT NULL UNIQUE
);
"""
CREATE_TAG_INDEX = """
CREATE INDEX IF NOT EXISTS tag_index ON tags (tag);
"""
CREATE_ASSOC_TABLE = """
CREATE TABLE IF NOT EXISTS logs_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log INTEGER NOT NULL,
    tag INTEGER NOT NULL,
    FOREIGN KEY (log) REFERENCES logs(id) ON DELETE CASCADE,
    FOREIGN KEY (tag) REFERENCES tags(id) ON DELETE CASCADE
);
"""
CREATE_ASSC_LOGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_log_index ON logs_tags (log);
"""
CREATE_ASSC_TAGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_tag_index ON logs_tags (tag);
"""

def get_connection():
    # This creates the sqlite db if it doesn't exist.
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_user_version(conn):
    c = conn.execute("PRAGMA user_version;")
    return c.fetchone()[0]


def set_user_version(conn, version, commit=True):
    version = int(version)
    conn.execute(f"PRAGMA user_version = {version};")
    conn.commit() if commit else None
    return version


def db_init(conn):
    # Ensure logs table
    conn.execute(CREATE_LOGS_TABLE)

    # Ensure tags table
    conn.execute(CREATE_TAGS_TABLE)
    conn.execute(CREATE_TAG_INDEX)

    # Ensure association table
    conn.execute(CREATE_ASSOC_TABLE)
    conn.execute(CREATE_ASSC_LOGS_INDEX)
    conn.execute(CREATE_ASSC_TAGS_INDEX)

    conn.commit()
    print("performed initial db setup")


def db_updated(conn):
    print(f"DB updated")


def db_setup(conn):
    """Set up the database and perform necessary migrations."""
    version = get_user_version(conn)
    if version == DB_USER_VERSION:
        return # the DB is up to date.

    # TODO: transactions?
    # TODO: Backup the database before migrating.

    migrations = [
        (1, db_init),
    ]

    for migration_version, migration in migrations:
        if version < migration_version:
            migration(conn)
            version = set_user_version(conn, version + 1)


#-----------------------------------------------------------------------------

class TagPrompt(cmd.Cmd):

    intro = 'Enter comma separated tags:'
    prompt = '(tags) '

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._personal_tags = None
        self._final_tags = None

    @staticmethod
    def _tag_split(line):
        # Use a set in order to de-duplicate tags, then convert back to list.
        tags = (tag.strip() for tag in line.split(','))
        tags = {t for t in tags if t}
        return list(tags)

    def default(self, line):
        self._final_tags = self._tag_split(line)

    def postcmd(self, stop, line):
        return True

    def completedefault(self, text, line, begidx, endidx):
        tag = self._tag_split(text)[-1]
        if tag:
            return [t for t in self._personal_tags if t.startswith(tag)]
        else:
            return self._personal_tags

    def completenames(self, text, *ignored):
        # Complete the last tag on the line
        tag = self._tag_split(text)[-1]
        if tag:
            return [t for t in self._personal_tags if t.startswith(tag)]
        else:
            return self._personal_tags

    def populate_tags(self, conn):
        c = conn.execute("SELECT tags.tag FROM tags;")
        self._personal_tags = [r[0] for r in c.fetchall()]

    @property
    def user_tags(self):
        return self._final_tags


INSERT_LOG = """
INSERT into logs (created_at, msg) VALUES (CURRENT_TIMESTAMP, ?);
"""
def insert_msg(conn, msg):
    c = conn.execute(INSERT_LOG, (msg,))
    conn.commit()
    return c.lastrowid


SELECT_LOGS_HAVING_TAGS_TEMPL = """
SELECT logs.id
FROM logs
WHERE logs.id in (
    SELECT log
    FROM logs_tags
    WHERE tag in (
        SELECT id
        FROM tags
        WHERE tag in ({tags})
    )
    GROUP BY log
    HAVING COUNT(tag) >= ?
);
"""
SELECT_LOGS_WITH_TAGS_ALL = """
SELECT logs.*, group_concat(tags.tag) as tags
FROM logs
INNER JOIN logs_tags lt ON lt.log = logs.id
INNER JOIN tags ON tags.id = lt.tag
GROUP BY logs.id, logs.created_at, logs.msg
ORDER BY logs.created_at;
"""
SELECT_LOGS_AND_TAGS_TEMPL = """
SELECT logs.*, group_concat(tags.tag) as tags
FROM logs
INNER JOIN logs_tags lt ON lt.log = logs.id
INNER JOIN tags ON tags.id = lt.tag
WHERE logs.id in ({tags})
GROUP BY logs.id, logs.created_at, logs.msg
ORDER BY logs.created_at;
"""

def _msg_ids_having_tags(conn, tag_groups):
    msg_ids = set()  # using a set in order to de-duplicate.

    for tags in tag_groups:
        select = SELECT_LOGS_HAVING_TAGS_TEMPL.format(
            tags=', '.join('?' for _ in tags)
        )

        # TODO: Achieve the following in the DB via a UNION?
        for row in conn.execute(select, (*tags, len(tags))):
            msg_ids.add(row[ID])

    return msg_ids


def messages_with_tags(conn, tag_groups):
    if not tag_groups or ((len(tag_groups) == 1) and not tag_groups[0]):
        return list(conn.execute(SELECT_LOGS_WITH_TAGS_ALL))

    msg_ids = _msg_ids_having_tags(conn, tag_groups)

    select = SELECT_LOGS_AND_TAGS_TEMPL.format(
        tags=', '.join('?' for _ in msg_ids)
    )

    return list(conn.execute(select, tuple(msg_ids)).fetchall())


def msg_exists(conn, msg_id):
    sql = 'SELECT id from logs where id = ?;'
    return conn.execute(sql, (msg_id,)).fetchone() is not None


def delete_msg(conn, msg_id, commit=True):
    """Delete the message with the given ID.

    propagate: If `True` (default), delete the associates to tags,
        but not the tags themselves.
    commit: If `True`, persist the changes to the DB.
    """
    msg_id = int(msg_id)

    # Delete the log message.
    msg_delete = "DELETE FROM logs WHERE id = ?;"
    c = conn.execute(msg_delete, (msg_id,))
    if c.rowcount != 1:
        return False

    if commit:
        conn.commit()

    return True


def delete_tag(conn, tag, commit=True):
    """Delete the tag with the given value.

    propagate: If `True` (default), delete the associates to logs,
        but not the logs themselves.
    commit: If `True`, persist the changes to the DB.
    """
    # Find the id of the tag.
    tag_select = "SELECT id FROM tags WHERE tag = ?;"
    c = conn.execute(tag_select, (tag,))
    result = c.fetchone()
    if not result:
        return False
    tag_id = result[0]

    # Delete the tag.
    tag_delete = "DELETE FROM tags WHERE id = ?;"
    c = conn.execute(tag_delete, (tag_id,))
    if c.rowcount != 1:
        return False

    if commit:
        conn.commit()

    return True


class RenderedLog:

    def __init__(self, logs, tags):
        """
        logs: A list/tuple, of 2-tuples (id, message)
        tags: The tags used to find the given logs. A list of lists of tags.
        """
        self.logs = list(logs)
        self.tags = list(tags) if tags else tuple()
        self._lines = []
        self._line_map = []

        self._render()

    def _render(self):
        # Header
        if self.tags:
            tag_groups = (', '.join(group) for group in self.tags)
            tags_together = (' || '.join(f"<{tg}>" for tg in tag_groups))
            header = f"# TAGS: {tags_together}\n"
            self._lines.append(header)

        # Body
        linenum_init = None
        linenum_last = None
        for row in self.logs:
            # Set the header for each message.
            self._lines.extend((
                f'\n',
                f'{79*"-"}\n',
                f'# ID: {row[ID]}\n',
                f'# Created: {row[CREATED_AT]}\n',
                f'# Tags: {row[TAGS].split(",")}\n',
                f'\n',
            ))

            linenum_init = len(self._lines) + 1

            msg_lines = row[MSG].splitlines(keepends=True)
            #assert len(msg_lines) > 0, "Message must have >= 1 line!"

            self._lines.extend(msg_lines)
            linenum_last = len(self._lines)
            self._line_map.append((row[ID], linenum_init, linenum_last))

        # Footer
        self._lines.append('\n')

    @property
    def rendered(self):
        return self._lines

    def diff(self, other, debug=False):
        """
        return an iterable of LogDiffs
        """
        linenum = 0
        msg_diff_lines = []
        log_diffs = []
        msg_map_idx = 0
        msg_id, line_init, line_last = self._line_map[msg_map_idx]
        new_msg = False

        for line in difflib.ndiff(self._lines, list(other)):
            if line.startswith('? '):
                # These intraline differences are not needed.
                continue

            if not (line.startswith('+ ') or line.startswith('? ')):
                linenum += 1

            if linenum > line_last and not new_msg:
                # Store the accumulated msg diff
                log_diffs.append(
                    LogDiff(
                        msg_id,
                        ''.join(difflib.restore(msg_diff_lines, 2)),
                        msg_diff_lines,
                        tags=flatten_tag_groups(self.tags)
                    )
                )
                msg_diff_lines = []

                if len(self._line_map) > (msg_map_idx + 1):
                    # Set up for the next message.
                    msg_map_idx += 1
                    msg_id, line_init, line_last = self._line_map[msg_map_idx]
                else:
                    # There are no existing messages. All following lines will
                    # be added to a new message.
                    new_msg = True
                    msg_id = None

            if debug:
                print(
                    (f"line: {linenum}, msg_id: {msg_id}, start: {line_init}"
                     f", stop: {line_last}, line: {line}"),
                    end=''
                )

            if (line_init <= linenum <= line_last) or new_msg:
                msg_diff_lines.append(line)

        # Store the accumulated msg diff
        log_diffs.append(
            LogDiff(
                msg_id,
                ''.join(difflib.restore(msg_diff_lines, 2)),
                msg_diff_lines,
                tags=flatten_tag_groups(self.tags)
            )
        )

        return log_diffs


class LogDiff:

    def __init__(self, msg_id, msg, diff_lines, tags=None):
        """
        mods: iterable of (change, line_num, text)
        """
        self.msg_id = msg_id
        self.msg = msg
        self.diff = ''.join(diff_lines)
        self.modified = any((
            line.startswith('- ') or line.startswith('+ ')
            for line in diff_lines
        ))
        self.is_new = msg_id is None
        self.tags = tags if tags else []

    def __str__(self):
        id_str = str(self.msg_id) if not self.is_new else 'New'
        return f"<LogDiff({id_str})>\n{self.diff}</LogDiff>"

    def update_or_create(self, conn, commit=True):
        if self.is_new:
            return self._create(conn, commit=commit)
        else:
            return self._update(conn, commit=commit)

    def _create(self, conn, commit=True):
        msg_id = insert_msg(conn, self.msg)
        self.msg_id = msg_id

        tag_ids = insert_tags(conn, self.tags)
        insert_asscs(conn, self.msg_id, tag_ids)

        if commit:
            conn.commit()

        return True

    def _update(self, conn, commit=True):
        if not self.modified:
            return False

        if not self.msg:
            # TODO: delete msg or mark as deleted?
            pass

        if not self._update_msg(conn):
            # TODO: Maybe throw a custom exception?
            return False

        if not self._update_diffs(conn):
            # TODO: Rollback? Throw exception?
            return False

        # Allow commit to be defered
        if commit:
            conn.commit()

        return True

    def _update_msg(self, conn):
        update = "UPDATE logs SET msg = ? WHERE id = ?"
        c = conn.execute(update, (self.msg, self.msg_id))
        return c.rowcount == 1

    def _update_diffs(self, conn):
        # TODO: Save diff info
        return True


def flatten_tag_groups(tag_groups):
    tags = []
    for group in tag_groups:
        tags.extend(group)
    return tags


def select_tag(conn, tag: str):
    c = conn.execute("SELECT * FROM tags WHERE tag = ?", (tag,))
    return c.fetchone()


INSERT_TAG = """
INSERT OR IGNORE INTO tags (tag) VALUES (?);
"""
def insert_tags(conn, tags):
    tag_ids = set()
    for tag in tags:
        result = select_tag(conn, tag)
        if result is None:
            c = conn.execute(INSERT_TAG, (tag,))
            tag_id = c.lastrowid
        else:
            tag_id, _ = result
        tag_ids.add(tag_id)

    conn.commit()
    return tag_ids


INSERT_LOG_TAG_ASSC = """
INSERT INTO logs_tags (log, tag) VALUES (?, ?);
"""
def insert_asscs(conn, msg_id, tag_ids):
    for tag_id in tag_ids:
        conn.execute(INSERT_LOG_TAG_ASSC, (msg_id, tag_id))
    conn.commit()
    return


def open_temp_logfile(lines=None):
    with tempfile.NamedTemporaryFile(suffix=".md") as tf:
        if lines:
            tf.writelines(line.encode('utf8') for line in lines)
            tf.flush()

        call([EDITOR, tf.name])

        tf.seek(0)
        return tf.read().decode('utf8')


def all_tags(conn):
    c = conn.execute("SELECT tags.tag FROM tags;")
    return [r[0] for r in c.fetchall()]


if __name__ == '__main__':
    args = parser.parse_args()

    dir_setup()
    conn = get_connection()
    db_setup(conn)

    if args.delete is not None:
        if msg_exists(conn, args.delete):
            delete_msg(conn, args.delete)
            print(f"Deleted message ID {args.delete}")
        else:
            print(f"No message found with ID {args.delete}")
        sys.exit()

    # Display messages
    if args.tags:
        messages = messages_with_tags(conn, args.tags)
        if not messages:
            tag_groups = (' && '.join(group) for group in args.tags)
            all_tags = (' || '.join(f"({tg})" for tg in tag_groups))
            print(f"No messages found for tags: {all_tags}")
            sys.exit()

        message_view = RenderedLog(messages, args.tags)
        edited = open_temp_logfile(message_view.rendered)
        diffs = message_view.diff(edited.splitlines(keepends=True))
        for diff in diffs:
            if diff.modified:
                # TODO: Delete msg if all lines removed?
                #print(diff)
                diff.update_or_create(conn, commit=False)
                if diff.is_new:
                    print(f"Saved additional message as ID {diff.msg_id}")
                else:
                    print(f"Saved changes to message ID {diff.msg_id}")

        conn.commit()
        sys.exit()

    # Store message
    msg = open_temp_logfile()
    if not msg:
        print("No message created...")
        sys.exit()

    msg_id = insert_msg(conn, msg)

    # Collect tags via custom prompt
    tag_prompt = TagPrompt()
    tag_prompt.populate_tags(conn)
    tag_prompt.cmdloop()

    if tag_prompt.user_tags:
        tag_ids = insert_tags(conn, tag_prompt.user_tags)
        insert_asscs(conn, msg_id, tag_ids)

    print(f"Saved as message ID {msg_id}")

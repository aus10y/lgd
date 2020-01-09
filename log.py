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


EDITOR = os.environ.get('EDITOR','vim') #that easy!


# Argparse stuff
parser = argparse.ArgumentParser(
    description="A flexible knowledge store."
)
parser.add_argument(
    '-s', '--show', action='store_true',
    help="Display logs in your editor ($EDITOR)."
)
parser.add_argument(
    '-t', '--tags', action='append', nargs='+',
    help="Tag(s) used to filter or add metadata to logs."
)
parser.add_argument(
    '-o', '--output', action="store", type=str,
    help="Specify an output file, or leave blank to output to stdio."
)
# TODO: Implement the output file redirection.


class TagPrompt(cmd.Cmd):

    intro = 'Enter comma separated tags:\n'
    prompt = '(tags) '

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._personal_tags = None
        self._final_tags = None

    @staticmethod
    def _tag_split(line):
        # Use a set in order to de-duplicate tags, then convert back to list.
        return list({tag.strip() for tag in line.split(',')})

    def default(self, line):
        self._final_tags = self._tag_split(line)

    def postcmd(self, stop, line):
        return True

    def completedefault(self, text, line, begidx, endidx):
        tag = self._tag_split(text)[-1]
        if tag:
            return [t for t in self._personal_tags if t.startswith(tag)]
        else:
            return []

    def completenames(self, text, *ignored):
        # Complete the last tag on the line
        tag = self._tag_split(text)[-1]
        if tag:
            return [t for t in self._personal_tags if t.startswith(tag)]
        else:
            return self._personal_tags

    def populate_tags(self, conn):
        c = conn.cursor()
        c.execute("SELECT tags.tag FROM tags;")
        self._personal_tags = [r[0] for r in c.fetchall()]

    @property
    def user_tags(self):
        return self._final_tags


LGD_PATH = Path.home() / Path('.lgd')
def dir_setup():
    # If our dir doesn't exist, create it.
    LGD_PATH.mkdir(mode=0o770, exist_ok=True)


DB_NAME = 'logs.db'
DB_PATH = LGD_PATH / Path(DB_NAME)
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
    FOREIGN KEY (log) REFERENCES logs(id),
    FOREIGN KEY (tag) REFERENCES tags(id)
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
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def db_setup(conn):
    c = conn.cursor()

    # Ensure logs table
    c.execute(CREATE_LOGS_TABLE)

    # Ensure tags table
    c.execute(CREATE_TAGS_TABLE)
    c.execute(CREATE_TAG_INDEX)

    # Ensure association table
    c.execute(CREATE_ASSOC_TABLE)
    c.execute(CREATE_ASSC_LOGS_INDEX)
    c.execute(CREATE_ASSC_TAGS_INDEX)

    conn.commit()


INSERT_LOG = """
INSERT into logs (created_at, msg) VALUES (CURRENT_TIMESTAMP, ?);
"""
def insert_msg(conn, msg):
    c = conn.cursor()
    c.execute(INSERT_LOG, (msg,))
    conn.commit()
    return c.lastrowid


SELECT_ALL = "SELECT * from logs;"
SELECT_WHERE_TAGS_TEMPL = """
SELECT logs.*
FROM logs
INNER JOIN logs_tags on logs_tags.log = logs.id
INNER JOIN tags on logs_tags.tag = tags.id
WHERE tags.tag in ({tags});
"""

_ = """
SELECT *
FROM tags
INNER JOIN logs_tags on logs_tags.tag = tags.id
INNER JOIN logs on logs_tags.log = logs.id
WHERE 1
"""


def messages_with_tags(conn, tags):
    # TODO: Finalize use of tags here
    # TODO: clean up
    # TODO: Have a class represent the synthetic message and ultimate diff?
    c = conn.cursor()

    if tags:
        select = SELECT_WHERE_TAGS_TEMPL.format(
            tags=', '.join('?' for tag in tags[0])
        )
        result = c.execute(select, tuple(tags[0]))
    else:
        result = c.execute(SELECT_ALL)

    return result


class RenderedLog:

    def __init__(self, logs):
        """
        messages: a list/tuple, of 2-tuples (id, message)
        """
        self.logs = list(logs)
        self._lines = []
        self._line_map = []

        self._render()

    def _render(self):
        first = True
        linenum_init = None
        linenum_last = None
        for msg_id, _, msg in self.logs:
            if first:
                first = False
            else:
                self._lines.extend(('\n', '---\n', '\n'))
                linenum_init += 3

            linenum_init = len(self._lines) + 1

            msg_lines = msg.splitlines(keepends=True)
            assert len(msg_lines) > 0, "Message must have >= 1 line!"

            self._lines.extend(msg_lines)
            linenum_last = len(self._lines)
            self._line_map.append((msg_id, linenum_init, linenum_last))

    @property
    def rendered(self):
        return self._lines

    def diff(self, other):
        """
        return an iterable of LogDiffs
        """
        linenum = 0
        msg_diff_lines = []
        msg_map_idx = 0
        msg_id, line_init, line_last = self._line_map[msg_map_idx]
        log_diffs = []

        for line in difflib.ndiff(self._lines, list(other)):
            if not (line.startswith('+ ') or line.startswith('? ')):
                linenum += 1

            if linenum > line_last:
                # Store the accumulated msg diff
                log_diffs.append(
                    LogDiff(
                        msg_id,
                        ''.join(difflib.restore(msg_diff_lines, 2)),
                        msg_diff_lines
                    )
                )
                msg_diff_lines = []

                if len(self._line_map) > (msg_map_idx + 1):
                    # Set up for the next message.
                    msg_map_idx += 1
                    msg_id, line_init, line_last = self._line_map[msg_map_idx]
                else:
                    # There are no more messages, all following lines are
                    # assumed to be synthetic and should be skipped.
                    pass

            print(f"linenum: {linenum}, msg_id: {msg_id}, start: {line_init}, stop: {line_last}, line: {line}", end='')

            if line_init <= linenum <= line_last:
                msg_diff_lines.append(line)

        # Store the accumulated msg diff
        log_diffs.append(
            LogDiff(
                msg_id,
                ''.join(difflib.restore(msg_diff_lines, 2)),
                msg_diff_lines
            )
        )

        return log_diffs


class LogDiff:

    def __init__(self, msg_id, msg, diff_lines):
        """
        mods: iterable of (change, line_num, text)
        """
        self.id = msg_id
        self.msg = msg
        self.diff = ''.join(diff_lines)
        self.modified = any((
            line.startswith('- ') or line.startswith('+ ')
            for line in diff_lines
        ))

    def __str__(self):
        return f"<LogDiff({self.id})>\n{self.diff}\n</LogDiff>"

    def save(self, conn, commit=True):
        if not self.modified:
            return

        if not self.msg:
            # TODO: delete msg or mark as deleted
            pass
        elif not self._update_msg(conn):
            # TODO: Maybe throw a custom exception?
            return False

        if not self._update_diffs(conn):
            # TODO: Rollback? Throw exception?
            return False

        # Allow commit to be defered
        if commit:
            conn.commit()

        return

    def _update_msg(self, conn):
        update = "UPDATE logs SET msg = ? WHERE id = ?"
        c = conn.cursor()
        c.execute(update, (self.msg, self.id))
        return c.rowcount == 1

    def _update_diffs(self, conn):
        # TODO: Save diff info
        return True


def select_tag(conn, tag: str):
    c = conn.cursor()
    c.execute("SELECT * FROM tags WHERE tag = ?", (tag,))
    return c.fetchone()


INSERT_TAG = """
INSERT OR IGNORE INTO tags (tag) VALUES (?);
"""
def insert_tags(conn, tags):
    c = conn.cursor()
    tag_ids = set()
    for tag in tags:
        result = select_tag(conn, tag)
        if result is None:
            c.execute(INSERT_TAG, (tag,))
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
    c = conn.cursor()
    for tag_id in tag_ids:
        c.execute(INSERT_LOG_TAG_ASSC, (msg_id, tag_id))
    conn.commit()
    return


def open_temp_logfile(lines=None):
    with tempfile.NamedTemporaryFile(suffix=".tmp") as tf:
        if lines:
            tf.writelines(line.encode('utf8') for line in lines)
            tf.flush()

        call([EDITOR, tf.name])

        tf.seek(0)
        return tf.read().decode('utf8')


def all_tags(conn):
    c = conn.cursor()
    c.execute("SELECT tags.tag FROM tags;")
    return [r[0] for r in c.fetchall()]


if __name__ == '__main__':
    args = parser.parse_args()

    dir_setup()
    conn = get_connection()
    db_setup(conn)

    # Display messages
    if args.show:
        messages = messages_with_tags(conn, args.tags)
        message_view = RenderedLog(messages)
        edited = open_temp_logfile(message_view.rendered)
        diffs = message_view.diff(edited.splitlines(keepends=True))
        for diff in diffs:
            if diff.modified:
                print(diff)
                diff.save(conn, commit=False)
            else:
                print(f"msg_id: {diff.id}, no change")
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

#!/usr/bin/env python3
import argparse
import cmd
import difflib
import re
import sys
import sqlite3
import tempfile
import uuid
import os

from datetime import datetime, timedelta
from pathlib import Path
from subprocess import call


EDITOR = os.environ.get('EDITOR','vim')
DEBUG = False


#-----------------------------------------------------------------------------
# Argparse stuff

# Capture YYYY, optional separators, opt. MM, opt. separators, opt. DD
date_regex = re.compile(
    r"(?P<year>[\d]{4})[/\-_.]?(?P<month>[\d]{2})?[/\-_.]?(?P<day>[\d]{2})?"
)

def sql_date_format(dt):
    return dt.strftime('%Y-%m-%d')

def to_datetime_range(arg):
    """Produce a (From, To) tuple of strings in YYYY-MM-DD format.
    These dates are inteded to be used in SQL queries, where the dates are
    expected to be in `YYYY-MM-DD` format, the "From" date is inclusive, and
    the "To" date is exclusive.
    """

    # Parse the date into separate fields.
    match = date_regex.match(arg)
    year, month, day = match['year'], match['month'], match['day']
    if year is None:
        raise Exception("Invalid date format")

    year = int(year)
    month = int(month) if month is not None else None
    day = int(day) if day is not None else None

    if day is not None and month is not None:
        # Full YYYY-MM-DD, increment day
        date_to = sql_date_format(
            datetime(year, month, day) + timedelta(days=1))
    elif day is None and month is not None:
        # YYYY-MM, increment month
        if month == 12:
            date_to = sql_date_format(datetime(year + 1, 1, 1))
        else:
            date_to = sql_date_format(datetime(year, month + 1, 1))
    elif day is None and month is None:
        # YYYY, increment year
        date_to = sql_date_format(datetime(year + 1, 1, 1))
    else:
        raise Exception("Invalid date format")

    date_from = sql_date_format(datetime(year, month or 1, day or 1))

    return (date_from, date_to)


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
    '-D', '--delete', action='store', type=str,
    help="Delete the message with the given ID."
)
parser.add_argument(
    '-d', '--date', action='store', type=to_datetime_range, dest='date_range',
    help=(
        "Filter by year, month, day."
        " Ex. `-d YYYYMMDD`. The year, month, day fields may optionally be"
        " separated by any of the following characters: `/`, `-`, `_`, `.`."
        " Ex. `--date YYYY/MM/DD`. The year, or year and month fields may be"
        " given without the rest of the data. Ex. `-d YYYY.MM`, `-d YYYY`."
    )
)
parser.add_argument(
    '-', dest='dash', action='store_true', default=False,
    help=(
        "Take input from STDIN, echo to STDOUT."
    )
)


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
ID = 'uuid'
LOG = 'log'
MSG = 'msg'
TAG = 'tag'
TAGS = 'tags'
CREATED_AT = 'created_at'

CREATE_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS logs (
    uuid UUID PRIMARY KEY,
    created_at int NOT NULL,
    msg TEXT NOT NULL
);
"""
CREATE_TAGS_TABLE = """
CREATE TABLE IF NOT EXISTS tags (
    uuid UUID PRIMARY KEY,
    tag TEXT NOT NULL UNIQUE
);
"""
CREATE_TAG_INDEX = """
CREATE INDEX IF NOT EXISTS tag_index ON tags (tag);
"""
CREATE_ASSOC_TABLE = """
CREATE TABLE IF NOT EXISTS logs_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_uuid UUID NOT NULL,
    tag_uuid UUID NOT NULL,
    FOREIGN KEY (log_uuid) REFERENCES logs(uuid) ON DELETE CASCADE,
    FOREIGN KEY (tag_uuid) REFERENCES tags(uuid) ON DELETE CASCADE
);
"""
CREATE_ASSC_LOGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_log_index ON logs_tags (log_uuid);
"""
CREATE_ASSC_TAGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_tag_index ON logs_tags (tag_uuid);
"""

def get_connection():
    # This creates the sqlite db if it doesn't exist.
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)

    # Register adapters and converters.
    sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes)
    sqlite3.register_converter('UUID', lambda b: uuid.UUID(bytes=b))

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

#------------------------------------------------------------------------------
# Autocompleting Tag Prompt

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

#------------------------------------------------------------------------------
# Terminal Color Helpers

class Term:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def header(cls, text):
        return f"{cls.HEADER}{text}{cls.ENDC}"

    @classmethod
    def blue(cls, text):
        return f"{cls.OKBLUE}{text}{cls.ENDC}"

    @classmethod
    def green(cls, text):
        return f"{cls.OKGREEN}{text}{cls.ENDC}"

    @classmethod
    def warning(cls, text):
        return f"{cls.WARNING}{text}{cls.ENDC}"

    @classmethod
    def error(cls, text):
        return f"{cls.ERROR}{text}{cls.ENDC}"

    @classmethod
    def bold(cls, text):
        return f"{cls.BOLD}{text}{cls.ENDC}"

    @classmethod
    def underline(cls, text):
        return f"{cls.UNDERLINE}{text}{cls.ENDC}"

    @staticmethod
    def apply_where(color_func, sub_string, text):
        return text.replace(sub_string, color_func(sub_string))

#------------------------------------------------------------------------------
# DB Row Wrappers

class DoesNotExist(Exception):
    pass


class Message:

    def __init__(self, pk: int, created_at, msg: str, tags: list):
        self.id = pk
        self.created_at = created_at
        self._msg = msg
        self._msg_new = msg
        self._tags = set(tags)
        self._tags_new = set(tags)
        self._modified = False

    @classmethod
    def new_message(cls, conn, msg, tags, commit=True):
        """Insert and return a Message."""
        msg_uuid = cls._insert_msg(conn, msg, commit=commit)
        tag_uuids = cls._insert_tags(conn, tags, commit=commit)
        cls._insert_asscs(conn, msg_uuid, tag_uuids, commit=commit)

        if commit:
            conn.commit()

        return cls.get_message(conn, msg_uuid)

    @classmethod
    def get_message(cls, conn, msg_uuid):
        """Retrieve a message."""
        sql = """
            SELECT logs.*, GROUP_CONCAT(tags.tag) as tags
            FROM logs
            LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
            LEFT JOIN tags ON tags.uuid = lt.tag_uuid
            WHERE logs.uuid = ?
            GROUP BY logs.uuid
            ORDER BY logs.created_at ASC;
        """

        msg = conn.execute(sql, (msg_uuid,)).fetchone()
        if msg is None:
            raise DoesNotExist()

        return Message(
            msg['id'],
            msg['created_at'],
            msg['msg'],
            msg['tags'].split(',') if msg['tags'] else []
        )

    @classmethod
    def all_messages(cls, conn):
        """Fetcha all messages."""
        sql = """
            SELECT logs.*, GROUP_CONCAT(tags.tag) as tags
            FROM logs
            LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
            LEFT JOIN tags ON tags.uuid = lt.tag_uuid
            GROUP BY logs.uuid
            ORDER BY logs.created_at ASC;
        """

        for msg in conn.execute(sql).fetchall():
            yield Message(
                msg['id'],
                msg['created_at'],
                msg['msg'],
                msg['tags'].split(',') if msg['tags'] else []
            )

    @classmethod
    def having_tags(cls, conn, tags, date_range=None):
        pass

    @classmethod
    def _insert_msg(cls, conn, msg, commit=True):
        INSERT_LOG = """
        INSERT into logs (uuid, created_at, msg) VALUES (?, CURRENT_TIMESTAMP, ?);
        """
        msg_uuid = uuid.uuid4()
        c = conn.execute(INSERT_LOG, (msg_uuid, msg,))
        if commit:
            conn.commit()
        return msg_uuid

    @classmethod
    def _insert_tags(cls, conn, tags, commit=True):
        INSERT_TAG = """
        INSERT OR IGNORE INTO tags (uuid, tag) VALUES (?, ?);
        """
        tag_uuids = set()
        for tag in tags:
            result = select_tag(conn, tag)
            if result is None:
                tag_uuid = uuid.uuid4()
                c = conn.execute(INSERT_TAG, (tag_uuid, tag))
            else:
                tag_uuid, _ = result
            tag_uuids.add(tag_uuid)

        if tag_uuids and commit:
            conn.commit()

        return tag_uuids

    @classmethod
    def _insert_asscs(cls, conn, msg_uuid, tag_uuids, commit=True):
        INSERT_LOG_TAG_ASSC = """
        INSERT INTO logs_tags (log_uuid, tag_uuid) VALUES (?, ?);
        """
        for tag_uuid in tag_uuids:
            conn.execute(INSERT_LOG_TAG_ASSC, (msg_uuid, tag_uuid))
        if tag_uuids and commit:
            conn.commit()
        return

    @classmethod
    def _remove_asscs(cls, conn, msg_uuid, tag_uuids, commit=True):
        ids = ', '.join('?' for _ in len(tag_uuids))
        sql = f"DELETE FROM logs_tags where log = ? and tag in ({ids});"
        conn.execute(sql, (msg_uuid, *tag_uuids))

    @classmethod
    def _update_msg(cls, conn, msg_uuid, msg, commit=True):
        update = "UPDATE logs SET msg = ? WHERE id = ?;"
        c = conn.execute(update, (msg, msg_uuid))
        return c.rowcount == 1

    @property
    def msg(self):
        return self._msg

    @msg.setter
    def msg(self, text):
        self._modified = True
        self._msg = text

    @property
    def tags(self):
        return sorted(self._tags)

    @tags.setter
    def tags(self, tags):
        self._modified = True
        self._tags_new = set(tags)

    def __str__(self):
        return self.msg

    def __repr__(self):
        msg_abbrv = self.msg if len(self.msg) < 20 else f"{self.msg[:17]}..."
        return f"<Message({self.id}: '{msg_abbrv}')>"

    def save(self, conn, commit=True):
        if not self._modified:
            return

        if self._tags != self._tags_new:
            # Remove old tag associations
            tags_to_remove = self._tags - self._tags_new
            if tags_to_remove:
                ids_to_remove = select_tags(conn, tags_to_remove)
                Message._remove_asscs(conn, self.id, ids_to_remove, commit=commit)

            # Insert new tag associations
            tags_to_add = self._tags_new - self._tags
            if tags_to_add:
                ids_to_add = Message._insert_tags(conn, tags_to_add, commit=commit)
                Message._insert_asscs(conn, self.id, ids_to_add, commit=commit)

        if self._msg != self._msg_new:
            pass


#------------------------------------------------------------------------------

INSERT_LOG = """
INSERT into logs (uuid, created_at, msg) VALUES (?, CURRENT_TIMESTAMP, ?);
"""
def insert_msg(conn, msg):
    msg_uuid = uuid.uuid4()
    c = conn.execute(INSERT_LOG, (msg_uuid, msg,))
    conn.commit()
    return msg_uuid

AND_DATE_BETWEEN_TEMPL = """
 AND {column} BETWEEN '{begin}' AND '{end}'
"""
SELECT_LOGS_HAVING_TAGS_TEMPL = """
SELECT logs.uuid
FROM logs
WHERE logs.uuid in (
    SELECT log_uuid
    FROM logs_tags
    WHERE tag_uuid in (
        SELECT uuid
        FROM tags
        WHERE tag in ({tags})
    )
    GROUP BY log_uuid
    HAVING COUNT(tag_uuid) >= ?
){date_range};
"""
SELECT_LOGS_WITH_TAGS_ALL_TEMPL = """
SELECT
    logs.uuid,
    datetime(logs.created_at, 'localtime') as created_at,
    logs.msg,
    group_concat(tags.tag) as tags
FROM logs
LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
LEFT JOIN tags ON tags.uuid = lt.tag_uuid
WHERE 1{date_range}
GROUP BY logs.uuid, logs.created_at, logs.msg
ORDER BY logs.created_at;
"""
SELECT_LOGS_AND_TAGS_TEMPL = """
SELECT
    logs.uuid,
    datetime(logs.created_at, 'localtime') as created_at,
    logs.msg,
    group_concat(tags.tag) as tags
FROM logs
LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
LEFT JOIN tags ON tags.uuid = lt.tag_uuid
WHERE logs.uuid in ({msgs})
GROUP BY logs.uuid, logs.created_at, logs.msg
ORDER BY logs.created_at;
"""


def _format_date_range(column, date_range):
    if date_range:
        return AND_DATE_BETWEEN_TEMPL.format(
            column=column,
            begin=date_range[0], end=date_range[1])
    else:
        return ''


def format_template_tags_dates(template, tags, date_col, date_range):
    tags = ', '.join('?' for _ in tags)
    dates = _format_date_range(date_col, date_range)
    return template.format(tags=tags, date_range=dates)


def _msg_uuids_having_tags(conn, tag_groups, date_range=None):
    msg_uuids = set()  # using a set in order to de-duplicate.

    for tags in tag_groups:
        select = format_template_tags_dates(
            SELECT_LOGS_HAVING_TAGS_TEMPL,
            tags,
            date_col='logs.created_at',
            date_range=date_range
        )
        for row in conn.execute(select, (*tags, len(tags))):
            msg_uuids.add(row[ID])

    return msg_uuids


def messages_with_tags(conn, tag_groups, date_range=None):
    if not tag_groups or ((len(tag_groups) == 1) and not tag_groups[0]):
        select = SELECT_LOGS_WITH_TAGS_ALL_TEMPL.format(
            date_range=_format_date_range('logs.created_at', date_range))
        return list(conn.execute(select))

    msg_uuids = _msg_uuids_having_tags(conn, tag_groups, date_range=date_range)
    select = SELECT_LOGS_AND_TAGS_TEMPL.format(
        msgs=', '.join('?' for _ in msg_uuids)
    )

    return list(conn.execute(select, tuple(msg_uuids)).fetchall())


def msg_exists(conn, msg_uuid):
    sql = 'SELECT uuid from logs where uuid = ?;'
    return conn.execute(sql, (msg_uuid,)).fetchone() is not None


def select_msgs_from_uuid_prefix(conn, uuid_prefix):
    uuid_prefix += '%'
    sql = "SELECT * from logs where hex(uuid) like ?;"
    return list(conn.execute(sql, (uuid_prefix,)).fetchall())


def delete_msg(conn, msg_uuid, commit=True):
    """Delete the message with the given UUID.

    propagate: If `True` (default), delete the associates to tags,
        but not the tags themselves.
    commit: If `True`, persist the changes to the DB.
    """
    msg_delete = "DELETE FROM logs WHERE uuid = ?;"
    c = conn.execute(msg_delete, (msg_uuid,))
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
    tag_select = "SELECT uuid FROM tags WHERE tag = ?;"
    c = conn.execute(tag_select, (tag,))
    result = c.fetchone()
    if not result:
        return False
    tag_uuid = result[0]

    # Delete the tag.
    tag_delete = "DELETE FROM tags WHERE uuid = ?;"
    c = conn.execute(tag_delete, (tag_uuid,))
    if c.rowcount != 1:
        return False

    if commit:
        conn.commit()

    return True


class RenderedLog:

    def __init__(self, logs, tags):
        """
        logs: A list/tuple, of 2-tuples (uuid, message)
        tags: The tags used to find the given logs. A list of lists of tags.
        """
        self.logs = list(logs)
        self.tag_groups = list(tags) if tags else tuple()
        self._all_tags = flatten_tag_groups(self.tag_groups)
        self._lines = []
        self._line_map = []
        self._render()  # Set up self._lines and self._lines_map

    def _render(self):
        # Header
        if self.tag_groups:
            tag_groups = (', '.join(group) for group in self.tag_groups)
            tags_together = (' || '.join(f"<{tg}>" for tg in tag_groups))
            header = f"# TAGS: {tags_together}\n"
            self._lines.append(header)

        # Body
        linenum_init, linenum_last = None, None
        for row in self.logs:
            # Set the header for each message.
            tags_str = '' if row[TAGS] is None else row[TAGS].replace(',', ', ')
            """
            self._lines.extend((
                f'\n',
                f'{79*"-"}\n',
                f'# ID: {row[ID]}\n',
                f'# Created: {row[CREATED_AT]}\n',
                f'# Tags: {tags_str}\n',
                f'\n',
            ))
            """
            self._lines.extend((
                f'\n',
                f'{79*"-"}\n',
                f'# {row[CREATED_AT]}\n',
                f'[ID: {row[ID]}]: # (Tags: {tags_str})\n',
                f'\n',
            ))

            linenum_init = len(self._lines)
            self._lines.extend(row[MSG].splitlines(keepends=True))

            linenum_last = len(self._lines) + 1
            self._line_map.append((row[ID], linenum_init, linenum_last))

        # Footer
        self._lines.extend((
            '\n',
            f'{79*"-"}\n',
            f'# Enter new log message below\n',
            f'# Tags: {", ".join(self._all_tags)}\n',
            '\n',
        ))

    @property
    def rendered(self):
        return self._lines

    @staticmethod
    def _is_addition(line):
        return line.startswith('+ ')

    @staticmethod
    def _is_removal(line):
        return line.startswith('- ')

    @staticmethod
    def _is_intraline(line):
        return line.startswith('? ')

    @staticmethod
    def _is_emptyline(line):
        return line == '  \n'

    @staticmethod
    def _is_modification(line):
        return RenderedLog._is_addition(line) or RenderedLog._is_removal(line)

    @staticmethod
    def _enumerate_diff(diff_lines):
        line_num = 0
        for line in diff_lines:
            if RenderedLog._is_intraline(line):
                # These intraline differences are not needed.
                continue

            if not RenderedLog._is_addition(line):
                line_num += 1

            yield (line_num, line)

    @staticmethod
    def _print_diff_info(line_num, msg_uuid, line_from, line_to, text, debug=False):
        msg_uuid = str(msg_uuid)
        line_from = str(line_from)
        line_to = str(line_to)
        if debug:
            print(
                (f"line: {line_num:>4}, msg_uuid: {msg_uuid},"
                 f" ({line_from:>4}, {line_to:>4}): {text}"),
                end=''
            )

    @staticmethod
    def _is_new_tag_line(line):
        TAG_LINE = '+ # Tags:'
        return line.startswith(TAG_LINE)

    @staticmethod
    def _parse_new_tags(line):
        TAG_LINE = '+ # Tags:'
        raw_tags = (t.strip() for t in line[len(TAG_LINE):].split(','))
        return [t for t in raw_tags if t]


    def diff(self, other, debug=False):
        """
        return an iterable of LogDiffs
        """
        line_num, diff_index = 0, 0
        msg_diff, log_diffs = [], []

        diff = difflib.ndiff(self._lines, list(other))
        diff = list(RenderedLog._enumerate_diff(diff))

        for msg_uuid, line_from, line_to in self._line_map:
            while True:
                if line_to < line_num:
                    # Line has moved past the current 'msg'
                    break

                if line_num in (line_from, line_to) and not RenderedLog._is_emptyline(text):
                    # Ignore the empty lines that we added, unless the user
                    # made a change.
                    # TODO: still not fully working when removing a real emptly line from bottom of message.
                    msg_diff.append(text)
                elif line_from < line_num < line_to:
                    # Line belongs to the current msg
                    msg_diff.append(text)

                if diff_index < len(diff):
                    # Line is below the current msg
                    line_num, text = diff[diff_index]
                    diff_index += 1

                    RenderedLog._print_diff_info(
                        line_num, msg_uuid, line_from, line_to, text, debug=debug
                    )

                    if RenderedLog._is_new_tag_line(text):
                        print(f"New tags for msg '{msg_uuid}', {RenderedLog._parse_new_tags(text)}")

            # TODO: Refactor LogDiff so that lines are iteratively given to it.
            log_diffs.append(LogDiff(msg_uuid, msg_diff, tags=self._all_tags))
            msg_diff = []

        # New msg
        new_tags = self._all_tags
        for line_num, text in diff[diff_index:]:
            RenderedLog._print_diff_info(
                line_num, None, None, None, text, debug=debug
            )
            if RenderedLog._is_new_tag_line(text):
                new_tags = RenderedLog._parse_new_tags(text)
            elif RenderedLog._is_addition(text):
                msg_diff.append(text)

        # Create and append the new msg, if it exists
        if msg_diff:
            log_diffs.append(LogDiff(None, msg_diff, tags=new_tags))

        return log_diffs


class LogDiff:

    def __init__(self, msg_uuid, diff_lines, tags=None):
        """
        mods: iterable of (change, line_num, text)
        """
        self.msg_uuid = msg_uuid
        self.msg = ''.join(difflib.restore(diff_lines, 2))
        self.diff = diff_lines
        self.modified = any((
            line.startswith('- ') or line.startswith('+ ')
            for line in diff_lines
        ))
        self.is_new = msg_uuid is None
        self.tags = tags if tags else []

    def __str__(self):
        return ''.join(self.diff)

    def __repr__(self):
        id_str = str(self.msg_uuid) if not self.is_new else 'New'
        return f"<LogDiff({id_str})>\n{str(self)}</LogDiff>"


    def update_or_create(self, conn, commit=True):
        if self.is_new:
            return self._create(conn, commit=commit)
        else:
            return self._update(conn, commit=commit)

    def _create(self, conn, commit=True):
        msg_uuid = insert_msg(conn, self.msg)
        self.msg_uuid = msg_uuid

        tag_uuids = insert_tags(conn, self.tags)
        insert_asscs(conn, self.msg_uuid, tag_uuids)

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
        update = "UPDATE logs SET msg = ? WHERE uuid = ?"
        c = conn.execute(update, (self.msg, self.msg_uuid))
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
    result = select_tags(conn, [tag])
    return result[0] if result else None


def select_tags(conn, tags: list):
    tag_snippet = ', '.join('?' for _ in tags)
    sql = f"SELECT * FROM tags WHERE tag in ({tag_snippet})"
    c = conn.execute(sql, tags)
    return c.fetchall()


INSERT_TAG = """
INSERT OR IGNORE INTO tags (uuid, tag) VALUES (?, ?);
"""
def insert_tags(conn, tags):
    tag_uuids = set()
    for tag in tags:
        result = select_tag(conn, tag)
        if result is None:
            tag_uuid = uuid.uuid4()
            c = conn.execute(INSERT_TAG, (tag_uuid, tag))
        else:
            tag_uuid, _ = result
        tag_uuids.add(tag_uuid)

    conn.commit()
    return tag_uuids


INSERT_LOG_TAG_ASSC = """
INSERT INTO logs_tags (log_uuid, tag_uuid) VALUES (?, ?);
"""
def insert_asscs(conn, msg_uuid, tag_uuids):
    for tag_uuid in tag_uuids:
        conn.execute(INSERT_LOG_TAG_ASSC, (msg_uuid, tag_uuid))
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


def prompt_for_delete(msg, uuid_prefix):
    # TODO: Improve uuid highlighting. Currently doesn't work for whole uuids.
    uuid_fragment = Term.apply_where(Term.green, uuid_prefix, str(msg[ID])[:8])
    msg_fragment = msg[MSG][:46].replace('\n', '\\n')

    prompt = f'{Term.warning("Delete")} {uuid_fragment}..., "{msg_fragment}..." (Y/n) '
    return input(prompt).lower() == 'y'


if __name__ == '__main__':
    args = parser.parse_args()

    dir_setup()
    conn = get_connection()
    db_setup(conn)

    if args.delete is not None:
        uuid_prefix = args.delete.replace('-', '')
        msgs = select_msgs_from_uuid_prefix(conn, uuid_prefix)
        if msgs:
            for msg in msgs:
                try:
                    confirmed = prompt_for_delete(msg, uuid_prefix)
                except EOFError:
                    print('')
                    sys.exit()

                if confirmed:
                    if delete_msg(conn, msg[ID]):
                        print(" - Deleted")
                    else:
                        print(" - Failed to delete")
        else:
            print(f"No message UUID prefixed with '{args.delete}'")
        sys.exit()

    # If reading from stdin
    if args.dash:
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            print(line, end='')
            lines.append(line)

        msg = ''.join(lines)
        msg_uuid = insert_msg(conn, msg)
        if args.tags:
            tags = flatten_tag_groups(args.tags)
            tag_uuids = insert_tags(conn, tags)
            insert_asscs(conn, msg_uuid, tag_uuids)

        print(f"Saved as message ID {msg_uuid}")
        sys.exit()

    # Display messages
    if args.tags or args.date_range:
        messages = messages_with_tags(conn, args.tags, args.date_range)
        if not messages:
            tag_groups = (' && '.join(group) for group in args.tags)
            all_tags = (' || '.join(f"({tg})" for tg in tag_groups))
            print(f"No messages found for tags: {all_tags}")
            sys.exit()

        message_view = RenderedLog(messages, args.tags)
        edited = open_temp_logfile(message_view.rendered)
        diffs = message_view.diff(edited.splitlines(keepends=True), debug=DEBUG)
        for diff in diffs:
            if diff.modified:
                if DEBUG:
                    print(repr(diff))

                # TODO: Delete msg if all lines removed?
                diff.update_or_create(conn, commit=False)
                if diff.is_new:
                    print(f"Saved additional message as ID {diff.msg_uuid}")
                else:
                    print(f"Saved changes to message ID {diff.msg_uuid}")

        conn.commit()
        sys.exit()

    # Store message
    msg = open_temp_logfile()
    if not msg:
        print("No message created...")
        sys.exit()

    msg_uuid = insert_msg(conn, msg)

    # Collect tags via custom prompt
    tag_prompt = TagPrompt()
    tag_prompt.populate_tags(conn)
    tag_prompt.cmdloop()

    if tag_prompt.user_tags:
        tag_uuids = insert_tags(conn, tag_prompt.user_tags)
        insert_asscs(conn, msg_uuid, tag_uuids)

    print(f"Saved as message ID {msg_uuid}")

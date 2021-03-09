#!/usr/bin/env python3
import argparse
import cmd
import csv
import difflib
import gzip
import io
import itertools
import os
import re
import shlex
import subprocess
from subprocess import TimeoutExpired
import sys
import sqlite3
import tempfile
import time
import uuid

from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Callable,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
)


EDITOR = os.environ.get("EDITOR", "vim")
DEBUG = False
EDITOR_POLL_PERIOD = 3


# ----------------------------------------------------------------------------
# Argparse stuff

date_regex: Pattern[str] = re.compile(
    r"(?P<year>[\d]{4})[/\-_.]?(?P<month>[\d]{2})?[/\-_.]?(?P<day>[\d]{2})?(?P<remainder>.*)?"
)


def user_date_components(
    date_str: str,
) -> Tuple[int, Union[int, None], Union[int, None]]:
    # Parse the date into separate fields.
    match = date_regex.match(date_str)
    year, month, day = match["year"], match["month"], match["day"]

    if year is None or match["remainder"]:
        raise argparse.ArgumentTypeError(f"Invalid date format '{date_str}'")

    year = int(year)
    month = int(month) if month is not None else None
    day = int(day) if day is not None else None

    return (year, month, day)


def sql_date_format(dt):
    return dt.strftime("%Y-%m-%d")


def date_range_from_single(date_str) -> tuple:
    year, month, day = user_date_components(date_str)

    if day is not None and month is not None:
        # Full YYYY-MM-DD, increment day
        date_to = datetime(year, month, day) + timedelta(days=1)
    elif day is None and month is not None:
        # YYYY-MM, increment month
        if month == 12:
            date_to = datetime(year + 1, 1, 1)
        else:
            date_to = datetime(year, month + 1, 1)
    elif day is None and month is None:
        # YYYY, increment year
        date_to = datetime(year + 1, 1, 1)
    else:
        raise Exception("Invalid date format")

    return (datetime(year, month or 1, day or 1), date_to)


def date_range_from_pair(start_str: str, end_str: str) -> Tuple[datetime, datetime]:
    # TODO: Refactor: Have one function that converts to datetime, call it twice.
    s_y, s_m, s_d = user_date_components(start_str)
    e_y, e_m, e_d = user_date_components(end_str)

    dt_start = datetime(s_y, s_m or 1, s_d or 1)
    dt_end = datetime(e_y, e_m or 1, e_d or 1)

    return (dt_start, dt_end) if dt_start < dt_end else (dt_end, dt_start)


def to_datetime_ranges(
    date_args: Union[List[List[str]], None]
) -> List[Tuple[datetime, datetime]]:
    """Produce a list of (From, To) tuples of strings in YYYY-MM-DD format.
    These dates are inteded to be used in SQL queries, where the dates are
    expected to be in `YYYY-MM-DD` format, the "From" date is inclusive, and
    the "To" date is exclusive.
    """
    if date_args is None:
        return []

    date_ranges = []
    for date_arg in date_args:
        if len(date_arg) == 1:
            date_range = date_range_from_single(*date_arg)
        elif len(date_arg) == 2:
            date_range = date_range_from_pair(*date_arg)
        else:
            raise Exception("`-d/--date` must only be given one or two values")
        date_ranges.append(date_range)

    return date_ranges


parser = argparse.ArgumentParser(description="A flexible knowledge store.")
parser.add_argument(
    "-t",
    "--tag",
    action="append",
    nargs="*",
    dest="tags",
    help=(
        "Show or tag a message or messages with the given tags.\n"
        " Filter messages by adding one or more tags separated by spaces.\n"
        " Matching messages must contain all given tags.\n"
        " Ex. `-t foo`, `-t foo bar`.\n"
        " Additional flag usage will OR the tag groups together.\n"
        " Ex. `-t foo bar -t baz`.\n"
    ),
)
parser.add_argument(
    "-D",
    "--delete",
    nargs="+",
    type=str,
    help=(
        "Delete note(s) matching the given UUID or UUID prefix. Confirmation"
        " of delete is requested for any notes found matching the value. If"
        " the -y/--yes flag is given, UUID arguments found to have one matching"
        " note will be deleted without a confirmation prompt, and UUID args"
        " matching more than one note will require confirmation from the user."
    ),
)
parser.add_argument(
    "-y",
    "--yes",
    action="store_true",
    default=False,
    dest="confirmation_override",
    help=(
        "Used to provide affirmative confirmation in place of an interactive prompt."
    ),
)
parser.add_argument(
    "-d",
    "--date",
    action="append",
    nargs="+",
    type=str,
    dest="date_ranges",
    help=(
        "Filter by year, month, day."
        " Ex. `-d YYYYMMDD`. The year, month, day fields may optionally be"
        " separated by any of the following characters: `/`, `-`, `_`, `.`."
        " Ex. `--date YYYY/MM/DD`. The year, or year and month fields may be"
        " given without the rest of the data. Ex. `-d YYYY.MM`, `-d YYYY`."
    ),
)
parser.add_argument(
    "-s",
    "--search",
    action="store",
    type=str,
    dest="search",
    help=("Search notes for the given string."),
)
parser.add_argument(
    "-",
    dest="dash",
    action="store_true",
    default=False,
    help=("Take input from STDIN, echo to STDOUT."),
)
parser.add_argument(
    "-TA",
    "--tag-associate",
    dest="tag_associate",
    action="append",
    nargs=2,
    metavar=("explicit", "denoted"),
    help=(
        "Create an association between two tags."
        " Any note tagged with the 'explicit' tag, will behave as if it is"
        " also tagged with the 'denoted' tag."
    ),
)
parser.add_argument(
    "-TD",
    "--tag-disassociate",
    dest="tag_disassociate",
    action="append",
    nargs=2,
    metavar=("explicit", "denoted"),
    help=("Remove an association between two tags."),
)
parser.add_argument(
    "--plain",
    action="store_true",
    default=False,
    help=("Disable rendering of note metadata in the editor."),
)
parser.add_argument(
    "-S",
    "--statistics",
    action="store_true",
    default=False,
    dest="tag_stats",
    help=("Print tag statistics."),
)
parser.add_argument(
    "-NI",
    "--note-import",
    dest="note_file_in",
    action="store",
    type=str,
    help=(
        "Import notes with UUIDs, created_at timestamps, note body, and tags."
        " The specified file should contain comma separated data (CSV) with"
        " the following headers: 'uuid, created_at, body, tags'."
        " Importing a particular file is an idempotent operation."
    ),
)
parser.add_argument(
    "-NE",
    "--note-export",
    dest="note_file_out",
    action="store",
    type=str,
    help=(
        "Export notes, with UUIDs, created_at timestamps, note body, and tags."
        " Notes will be exported in a comma separated format (CSV)."
    ),
)
parser.add_argument(
    "-TI",
    "--tag-import",
    dest="tag_file_in",
    action="store",
    type=str,
    help=(
        "Import tag associations."
        " The importation of tag associations is idempotent."
        " Import does not overwrite existing tag associations."
    ),
)
parser.add_argument(
    "-TE",
    "--tag-export",
    dest="tag_file_out",
    action="store",
    type=str,
    help=(
        "Export tag associations."
        " Tag associations will be exported in a comma separated format (CSV)."
    ),
)
parser.add_argument(
    "-m",
    "--meta",
    dest="metadata",
    action="store_true",
    help=(
        "Print note metadata to the terminal. Note UUIDs, created at"
        " datetimes, and tags are displayed."
    ),
)


# ----------------------------------------------------------------------------
# Path

LGD_PATH = Path.home() / Path(".lgd")


def dir_setup():
    # If our dir doesn't exist, create it.
    LGD_PATH.mkdir(mode=0o770, exist_ok=True)


# ----------------------------------------------------------------------------
# Database Setup

DB_NAME = "logs.db"
DB_PATH = LGD_PATH / Path(DB_NAME)
DB_USER_VERSION = 1

# Column Names
ID = "uuid"
LOG = "log"
MSG = "msg"
TAG = "tag"
TAGS = "tags"
CREATED_AT = "created_at"

CREATE_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS logs (
    uuid UUID PRIMARY KEY,
    created_at timestamp NOT NULL,
    msg GZIP NOT NULL
);
"""

CREATE_CREATED_AT_INDEX = """
CREATE INDEX IF NOT EXISTS created_at_idx ON logs (created_at);
"""

CREATE_TAGS_TABLE = """
CREATE TABLE IF NOT EXISTS tags (
    uuid UUID PRIMARY KEY,
    tag TEXT NOT NULL UNIQUE
);
"""

CREATE_TAG_INDEX = """
CREATE INDEX IF NOT EXISTS tag_idx ON tags (tag);
"""

CREATE_ASSOC_TABLE = """
CREATE TABLE IF NOT EXISTS logs_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_uuid UUID NOT NULL,
    tag_uuid UUID NOT NULL,
    FOREIGN KEY (log_uuid) REFERENCES logs(uuid) ON DELETE CASCADE,
    FOREIGN KEY (tag_uuid) REFERENCES tags(uuid) ON DELETE CASCADE,
    UNIQUE(log_uuid, tag_uuid)
);
"""

CREATE_ASSC_LOGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_log_idx ON logs_tags (log_uuid);
"""

CREATE_ASSC_TAGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_tag_idx ON logs_tags (tag_uuid);
"""

CREATE_TAG_RELATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS tag_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_uuid UUID NOT NULL,
    tag_uuid_denoted UUID NOT NULL,
    FOREIGN KEY (tag_uuid) REFERENCES tags(uuid) ON DELETE CASCADE,
    FOREIGN KEY (tag_uuid_denoted) REFERENCES tags(uuid) ON DELETE CASCADE,
    UNIQUE(tag_uuid, tag_uuid_denoted)
);
"""

CREATE_TAG_DIRECT_INDEX = """
CREATE INDEX IF NOT EXISTS tag_direct_idx ON tag_relations (tag_uuid);
"""

CREATE_TAG_INDIRECT_INDEX = """
CREATE INDEX IF NOT EXISTS tag_indirect_idx ON tag_relations (tag_uuid_denoted);
"""

# Create the full text search table using the External-Content-Table syntax.
CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
    note, content=''
);
"""

FTS_TRIGGERS = """
CREATE TRIGGER logs_ai AFTER INSERT ON logs BEGIN
    INSERT INTO logs_fts(rowid, note) VALUES (NEW.rowid, unzip(NEW.msg));
END;
CREATE TRIGGER logs_ad AFTER DELETE ON logs BEGIN
    INSERT INTO logs_fts(logs_fts, rowid, note) VALUES('delete', OLD.rowid, unzip(OLD.msg));
END;
CREATE TRIGGER logs_au AFTER UPDATE ON logs BEGIN
    INSERT INTO logs_fts(logs_fts, rowid, note) VALUES('delete', OLD.rowid, unzip(OLD.msg));
    INSERT INTO logs_fts(rowid, note) VALUES (NEW.rowid, unzip(NEW.msg));
END;
"""


def get_connection(db_path: str, debug=False) -> sqlite3.Connection:
    # This creates the sqlite db if it doesn't exist.
    conn = sqlite3.connect(
        db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )

    # Register adapters and converters.
    sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes)
    sqlite3.register_converter("UUID", lambda b: uuid.UUID(bytes=b))
    sqlite3.register_adapter(Gzip, lambda s: Gzip.compress_string(s))
    sqlite3.register_converter(Gzip.COL_TYPE, lambda b: Gzip.decompress_string(b))

    # Register functions
    conn.create_function("unzip", 1, Gzip.decompress_string)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    if debug:
        sqlite3.enable_callback_tracebacks(True)
        conn.set_trace_callback(print)

    return conn


def get_user_version(conn: sqlite3.Connection) -> int:
    c = conn.execute("PRAGMA user_version;")
    return c.fetchone()[0]


def set_user_version(conn: sqlite3.Connection, version: int, commit=True) -> int:
    version = int(version)
    conn.execute(f"PRAGMA user_version = {version};")
    conn.commit() if commit else None
    return version


def db_init(conn):
    # Ensure logs table
    conn.execute(CREATE_LOGS_TABLE)
    conn.execute(CREATE_CREATED_AT_INDEX)

    # Ensure tags table
    conn.execute(CREATE_TAGS_TABLE)
    conn.execute(CREATE_TAG_INDEX)

    # Ensure association table
    conn.execute(CREATE_ASSOC_TABLE)
    conn.execute(CREATE_ASSC_LOGS_INDEX)
    conn.execute(CREATE_ASSC_TAGS_INDEX)

    # Ensure tag relations table
    conn.execute(CREATE_TAG_RELATIONS_TABLE)
    conn.execute(CREATE_TAG_DIRECT_INDEX)
    conn.execute(CREATE_TAG_INDIRECT_INDEX)

    # Ensure full text search table & triggers
    conn.execute(CREATE_FTS_TABLE)
    conn.executescript(FTS_TRIGGERS)


DB_MIGRATIONS = [
    (1, db_init),
]


def db_setup(
    conn: sqlite3.Connection,
    migrations: List[Tuple[int, Callable[[sqlite3.Connection], None]]],
) -> bool:
    """Set up the database and perform necessary migrations."""
    version = get_user_version(conn)
    if version == DB_USER_VERSION:
        return True  # the DB is up to date.

    # TODO: Backup the database before migrating?

    for migration_version, migration in migrations:
        if version < migration_version:
            try:
                with conn:
                    conn.execute("BEGIN")
                    migration(conn)
            except sqlite3.Error as e:
                print(Term.error(str(e)))
                return False
            version = set_user_version(conn, version + 1)

    return True


# -----------------------------------------------------------------------------
# Types

Note = namedtuple("Note", ("uuid", "created_at", "body", "tags"))

Tag = namedtuple("Tag", ("uuid", "value"))

TagRelation = namedtuple("TagRelation", ("direct", "indirect"))

# -----------------------------------------------------------------------------
# Misc. Utilities


class TagPrompt(cmd.Cmd):

    intro = "Enter comma separated tags:"
    prompt = "(tags) "

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._personal_tags = None
        self._final_tags = None

    @staticmethod
    def _tag_split(line) -> List[str]:
        # Use a set in order to de-duplicate tags, then convert back to list.
        tags = (tag.strip() for tag in line.split(","))
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


class Term:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def header(cls, text: str) -> str:
        return f"{cls.HEADER}{text}{cls.ENDC}"

    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.OKBLUE}{text}{cls.ENDC}"

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.OKGREEN}{text}{cls.ENDC}"

    @classmethod
    def warning(cls, text: str) -> str:
        return f"{cls.WARNING}{text}{cls.ENDC}"

    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.ERROR}{text}{cls.ENDC}"

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.ENDC}"

    @classmethod
    def underline(cls, text: str) -> str:
        return f"{cls.UNDERLINE}{text}{cls.ENDC}"

    @staticmethod
    def apply_where(
        color_func: Callable[[str], str], sub_string: str, text: str
    ) -> str:
        return text.replace(sub_string, color_func(sub_string))


class Gzip(str):
    """This class exisits to aid sqlite adapters and converters for compressing
    text via gzip."""

    COL_TYPE = "GZIP"

    @staticmethod
    def compress_string(msg_str: str, **kwargs) -> bytes:
        return gzip.compress(msg_str.encode("utf8"), **kwargs)

    @staticmethod
    def decompress_string(msg_bytes: bytes) -> str:
        return gzip.decompress(msg_bytes).decode("utf8")


def flatten_tag_groups(tag_groups: Iterable[Tuple[str, ...]]) -> List[str]:
    tags = []
    for group in tag_groups:
        tags.extend(group)
    return tags


def user_confirmation(prompt: str) -> bool:
    response = input(prompt).lower()
    return response == "y" or response == "yes"


def prompt_for_delete(msg: sqlite3.Row, uuid_prefix: str) -> bool:
    # TODO: Improve uuid highlighting. Currently doesn't work for whole uuids.
    uuid_fragment = Term.apply_where(Term.green, uuid_prefix, str(msg[ID])[:8])
    msg_fragment = msg[MSG][:46].replace("\n", "\\n")[:46].ljust(49, ".")

    prompt = f'{Term.warning("Delete")} {uuid_fragment}..., "{msg_fragment}" [Y/n] '
    return user_confirmation(prompt)


def open_temp_logfile(lines: Union[List[str], None] = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
        if lines:
            tf.writelines(line.encode("utf8") for line in lines)
            tf.flush()
        tf.close()

        subprocess.call([*(shlex.split(EDITOR)), tf.name])

        with open(tf.name) as f:
            contents = f.read()

        os.unlink(tf.name)

    return contents


def editor(body: List[str]) -> Generator[List[str], None, None]:
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
        tf.writelines(line.encode("utf8") for line in body)
        tf.flush()
        tf.close()

        last_mtime = os.path.getmtime(tf.name)

        # Launch the editor in a subprocess.
        with subprocess.Popen([*(shlex.split(EDITOR)), tf.name]) as proc:
            terminated = False
            while not terminated:
                try:
                    terminated = proc.wait(timeout=EDITOR_POLL_PERIOD) is not None
                except TimeoutExpired:
                    # The editor process has not been closed yet, but the file
                    # contents may have changed.
                    pass

                # Determine if the file has been modified
                curr_mtime = os.path.getmtime(tf.name)
                if curr_mtime == last_mtime:
                    continue
                last_mtime = curr_mtime

                # Read the new contents
                with open(tf.name) as f:
                    edited = f.read()

                yield edited.splitlines(keepends=True)

        os.unlink(tf.name)


def format_tag_statistics(cur: sqlite3.Cursor) -> List[str]:
    stats: List[sqlite3.Row] = list(cur)

    TAG = "Tag"
    DENOTED_BY = "Denoted By"
    DENOTES = "Denotes"
    DIRECT = "Direct"
    INDIRECT = "Indirect"

    tag_width = len(TAG)
    denoted_by_width = len(DENOTED_BY)
    denotes_width = len(DENOTES)
    direct_width = len(DIRECT)
    indirect_width = len(INDIRECT)

    for row in stats:
        tag_width = max((tag_width, len(row["tag"])))
        denoted_by_width = max((denoted_by_width, len(row["children"])))
        denotes_width = max((denotes_width, len(row["implies"])))

    STATS_HEAD_TEMPL = " {: ^{tag_w}} | {: ^{direct_w}} | {: ^{indirect_w}} | {: ^{child_w}} | {: ^{impl_w}}"
    STATS_BODY_TEMPL = " {: <{tag_w}} | {: >{direct_w}} | {: >{indirect_w}} | {: <{child_w}} | {: <{impl_w}}"

    stats_table = [
        "",  # Empty line above table
        Term.bold(
            Term.header(
                STATS_HEAD_TEMPL.format(
                    TAG,
                    DIRECT,
                    INDIRECT,
                    DENOTED_BY,
                    DENOTES,
                    tag_w=tag_width,
                    direct_w=direct_width,
                    indirect_w=indirect_width,
                    child_w=denoted_by_width,
                    impl_w=denotes_width,
                )
            )
        ),
    ]

    for row in stats:
        stats_table.append(
            STATS_BODY_TEMPL.format(
                row["tag"],
                row["direct"],
                row["implied"],
                row["children"],
                row["implies"],
                tag_w=tag_width,
                direct_w=direct_width,
                indirect_w=indirect_width,
                child_w=denoted_by_width,
                impl_w=denotes_width,
            )
        )

    stats_table.append("")  # Empty line below table
    return stats_table


def split_tags(tags: Union[str, None]) -> FrozenSet[str]:
    return frozenset(t.strip() for t in tags.split(",")) if tags else frozenset()


def rows_to_notes(rows: List[sqlite3.Row]) -> Generator[Note, None, None]:
    return (
        Note(r["uuid"], r["created_at"], r["body"], split_tags(r["tags"])) for r in rows
    )


def ui_delete_notes(
    conn: sqlite3.Connection,
    uuid_args: List[str],
    override: bool = False,
    override_strong: bool = False,
):
    def confirm_single(note, prefix):
        try:
            return override or override_strong or prompt_for_delete(note, prefix)
        except (EOFError, KeyboardInterrupt):
            print("")
            sys.exit()

    def confirm_multi(note, prefix):
        try:
            return override_strong or prompt_for_delete(note, prefix)
        except (EOFError, KeyboardInterrupt):
            print("")
            sys.exit()

    uuid_prefixes = [uuid_arg.replace("-", "") for uuid_arg in uuid_args]
    notes_matched = [
        (uuid_prefix, select_msgs_from_uuid_prefix(conn, uuid_prefix))
        for uuid_prefix in uuid_prefixes
    ]

    for uuid_prefix, notes in notes_matched:
        if not notes:
            print(f"{uuid_prefix}, No note found")
            continue

        if len(notes) == 1:
            get_confirmation = confirm_single
        else:
            get_confirmation = confirm_multi

        for note in notes:
            if get_confirmation(note, uuid_prefix):
                if delete_msg(conn, note[ID]):
                    print(f" - Deleted {note[ID]}")
                else:
                    print(f" - Failed to delete {note[ID]}")


def stdin_note() -> str:
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        lines.append(line)

        # Print the lines back to stdout, so that this tool may be better
        # composed with other command line tools.
        print(line, end="")

    return "".join(lines)


def get_metadata(note: Note) -> str:
    tags = ",".join(sorted(note.tags))
    return f'{note.uuid},{note.created_at},"{tags}"'


# -----------------------------------------------------------------------------
# Exceptions


class LgdException(Exception):
    pass


class CSVError(LgdException):
    pass


# -----------------------------------------------------------------------------
# SQL queries and related functions

# Log / Message related

INSERT_LOG = """
INSERT into logs (uuid, created_at, msg) VALUES (?, {timestamp}, ?);
"""


def insert_msg(
    conn: sqlite3.Connection, msg, msg_uuid: uuid.UUID = None, created_at=None
) -> uuid.UUID:
    msg_uuid = uuid.uuid4() if msg_uuid is None else msg_uuid

    if created_at is None:
        insert = INSERT_LOG.format(timestamp="CURRENT_TIMESTAMP")
        params = (msg_uuid, Gzip(msg))
    else:
        insert = INSERT_LOG.format(timestamp="?")
        params = (msg_uuid, created_at, Gzip(msg))

    _ = conn.execute(insert, params)
    conn.commit()
    return msg_uuid


SELECT_NOTES_WHERE_TEMPL = """
SELECT
    logs.uuid AS uuid,
    datetime(logs.created_at{datetime_modifier}) AS "created_at [timestamp]",
    logs.msg AS body,
    group_concat(tags.tag) AS tags
FROM logs
LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
LEFT JOIN tags ON tags.uuid = lt.tag_uuid
WHERE
    ({uuid_filter})
AND ({tags_filter})
AND ({text_filter})
AND ({date_filter})
GROUP BY logs.uuid, logs.created_at, logs.msg
ORDER BY logs.created_at;
"""

_WHERE_UUIDS = """logs.uuid IN ({uuids})"""

_WHERE_TAGS_ALL = """(logs.uuid in (
    SELECT log_uuid
    FROM logs_tags
    WHERE tag_uuid in (
        SELECT uuid
        FROM tags
        WHERE tag in ({tags})
    )
    GROUP BY log_uuid
    HAVING COUNT(tag_uuid) >= ?))
"""

_WHERE_TAGS_ANY = """(logs.uuid in (
    SELECT log_uuid
    FROM logs_tags
    WHERE tag_uuid in (
        SELECT uuid
        FROM tags
        WHERE tag in ({tags})
    )
    GROUP BY log_uuid))
"""

_WHERE_NO_TAGS = """(logs.uuid in (
    SELECT logs.uuid
    FROM logs
    LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
    WHERE lt.log_uuid is NULL))
"""

_WHERE_FTS = """logs.uuid in (
    SELECT logs.uuid
    FROM logs
    INNER JOIN (
        SELECT rowid
        FROM logs_fts
        WHERE logs_fts MATCH ?
        ORDER BY rank) fts on fts.rowid = logs.rowid)
"""

_DATE_BETWEEN_FRAGMENT = "({column} BETWEEN '{begin}' AND '{end}')"


def select_notes(
    conn: sqlite3.Connection,
    uuids: Optional[List[uuid.UUID]] = None,
    tag_groups: Optional[List[Tuple[str, ...]]] = None,
    date_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
    text: Optional[str] = None,
    localtime: bool = True,
) -> List[Note]:
    params = []

    # Where having uuids
    uuid_filter = "1"
    if uuids:
        uuid_filter = _WHERE_UUIDS.format(uuids=", ".join("?" for _ in uuids))
        params.extend(uuids)

    # Where tagged with
    tags_filter = "1"
    if tag_groups and not (len(tag_groups) == 1 and not tag_groups[0]):
        tag_fragments = []

        # All tag groups containing one tag may be grouped together and OR'd.
        tags_or = [tg[0] for tg in tag_groups if len(tg) == 1 and tg != ("",)]
        tags_and = [tg for tg in tag_groups if len(tg) > 1]
        tags_none = any(tg == ("",) for tg in tag_groups)

        for tag_group in tags_and:
            tag_fragments.append(
                _WHERE_TAGS_ALL.format(tags=", ".join("?" for _ in tag_group))
            )
            params.extend((*tag_group, len(tag_group)))

        if tags_or:
            tag_fragments.append(
                _WHERE_TAGS_ANY.format(tags=", ".join("?" for _ in tags_or))
            )
            params.extend(tags_or)

        if tags_none:
            tag_fragments.append(_WHERE_NO_TAGS)

        tags_filter = " OR ".join(tag_fragments)

    # Where contains the text
    text_filter = "1"
    if text:
        text_filter = _WHERE_FTS
        params.append(text)

    # Where between dates
    date_filter = "1"
    if date_ranges:
        date_filter = " OR ".join(
            _DATE_BETWEEN_FRAGMENT.format(
                column="logs.created_at",
                begin=date_range[0],
                end=date_range[1],
            )
            for date_range in date_ranges
        )

    # Format the datetime
    datetime_modifier = ""
    if localtime:
        datetime_modifier = ", 'localtime'"

    query = SELECT_NOTES_WHERE_TEMPL.format(
        uuid_filter=uuid_filter,
        tags_filter=tags_filter,
        text_filter=text_filter,
        date_filter=date_filter,
        datetime_modifier=datetime_modifier,
    )

    return list(rows_to_notes(conn.execute(query, params).fetchall()))


UPDATE_LOG = """
UPDATE logs SET msg = ? WHERE uuid = ?
"""


def update_msg(conn: sqlite3.Connection, msg_uuid: uuid.UUID, msg: str) -> bool:
    c = conn.execute(UPDATE_LOG, (Gzip(msg), msg_uuid))
    return c.rowcount == 1


def msg_exists(conn: sqlite3.Connection, msg_uuid) -> bool:
    sql = "SELECT uuid from logs where uuid = ?;"
    return conn.execute(sql, (msg_uuid,)).fetchone() is not None


def select_msgs_from_uuid_prefix(
    conn: sqlite3.Connection, uuid_prefix: str
) -> List[sqlite3.Row]:
    uuid_prefix += "%"
    sql = "SELECT * from logs where hex(uuid) like ?;"
    return conn.execute(sql, (uuid_prefix,)).fetchall()


def delete_msg(conn: sqlite3.Connection, msg_uuid, commit=True) -> bool:
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


# Tags


def delete_tag(conn: sqlite3.Connection, tag: str, commit=True) -> bool:
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


def select_tag(conn: sqlite3.Connection, tag: str) -> Union[sqlite3.Row, None]:
    result = select_tags(conn, [tag])
    return result[0] if result else None


def select_tags(
    conn: sqlite3.Connection, tags: Union[List[str], Tuple[str, ...]]
) -> List[sqlite3.Row]:
    tag_snippet = ", ".join("?" for _ in tags)
    sql = f"SELECT * FROM tags WHERE tag in ({tag_snippet})"
    c = conn.execute(sql, tags)
    return c.fetchall()


INSERT_TAG = """
INSERT OR IGNORE INTO tags (uuid, tag) VALUES (?, ?);
"""


def insert_tags(conn: sqlite3.Connection, tags: Iterable[str]) -> Set[uuid.UUID]:
    tag_uuids = set()
    for tag in tags:
        result = select_tag(conn, tag)
        if result is None:
            tag_uuid = uuid.uuid4()
            _ = conn.execute(INSERT_TAG, (tag_uuid, tag))
        else:
            tag_uuid, _ = result
        tag_uuids.add(tag_uuid)

    conn.commit()
    return tag_uuids


def select_all_tags(conn: sqlite3.Connection) -> List[str]:
    c = conn.execute("SELECT tags.tag FROM tags;")
    return [r[0] for r in c.fetchall()]


# Log-Tag associations

INSERT_LOG_TAG_ASSC = """
INSERT INTO logs_tags (log_uuid, tag_uuid) VALUES (?, ?);
"""


def insert_asscs(
    conn: sqlite3.Connection, msg_uuid: uuid.UUID, tag_uuids: Iterable[uuid.UUID]
) -> None:
    for tag_uuid in tag_uuids:
        try:
            conn.execute(INSERT_LOG_TAG_ASSC, (msg_uuid, tag_uuid))
        except sqlite3.IntegrityError as e:
            if "unique" in str(e).lower():
                continue
            else:
                raise e
    conn.commit()
    return


def remove_asscs(
    conn: sqlite3.Connection, msg_uuid: uuid.UUID, tag_uuids: Iterable[uuid.UUID]
) -> None:
    if not tag_uuids:
        return

    sql = "DELETE FROM logs_tags WHERE log_uuid = ? AND tag_uuid in ({tags})".format(
        tags=",".join("?" for _ in tag_uuids)
    )
    conn.execute(sql, (msg_uuid, *tag_uuids))
    return


# Tag Relations

INSERT_TAG_RELATION = """
INSERT INTO tag_relations (tag_uuid, tag_uuid_denoted) VALUES (?, ?);
"""


def insert_tag_relation(
    conn: sqlite3.Connection, explicit: str, implicit: str, quiet=False
):
    tags = select_tags(conn, (explicit, implicit))
    tags = {t[TAG]: t[ID] for t in tags}

    if explicit not in tags:
        explicit_uuid = insert_tags(conn, (explicit,)).pop()
        if not quiet:
            print(f"- Inserted '{explicit}' tag'")
    else:
        explicit_uuid = tags[explicit]

    if implicit not in tags:
        implicit_uuid = insert_tags(conn, (implicit,)).pop()
        if not quiet:
            print(f"- Inserted '{implicit}' tag'")
    else:
        implicit_uuid = tags[implicit]

    with conn:
        conn.execute(INSERT_TAG_RELATION, (explicit_uuid, implicit_uuid))

    return


REMOVE_TAG_RELATION = """
DELETE from tag_relations WHERE tag_uuid = ? AND tag_uuid_denoted = ?;
"""


def remove_tag_relation(conn: sqlite3.Connection, explicit: str, implicit: str) -> bool:
    tags = select_tags(conn, (explicit, implicit))
    tags = {t[TAG]: t[ID] for t in tags}

    if explicit not in tags:
        raise LgdException(f"Relation not removed: Tag '{explicit}' not found!")
    if implicit not in tags:
        raise LgdException(f"Relation not removed: Tag '{implicit}' not found!")

    explicit_uuid = tags[explicit]
    implicit_uuid = tags[implicit]

    with conn:
        result = conn.execute(REMOVE_TAG_RELATION, (explicit_uuid, implicit_uuid))

    return result.rowcount == 1


SELECT_TAG_RELATIONS_ALL = """
SELECT t1.tag AS tag_direct, t2.tag AS tag_indirect
FROM tag_relations tr
INNER JOIN tags t1 ON t1.uuid = tr.tag_uuid
INNER JOIN tags t2 ON t2.uuid = tr.tag_uuid_denoted;
"""


def select_related_tags_all(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cursor = conn.execute(SELECT_TAG_RELATIONS_ALL)
    return cursor.fetchall()


SELECT_TAG_RELATIONS = """
WITH RECURSIVE relations (tag, tag_uuid, tag_denoted, tag_uuid_denoted) AS (
  SELECT tags.tag, tr.tag_uuid, tags_from.tag, tr.tag_uuid_denoted
  FROM tag_relations tr
    INNER JOIN tags as tags_from on tags_from.uuid = tr.tag_uuid_denoted
    INNER JOIN tags on tags.uuid = tr.tag_uuid
  WHERE tags_from.tag = ?

  UNION

  SELECT tags.tag, tr.tag_uuid, relations.tag, tr.tag_uuid_denoted
  FROM tag_relations tr
    INNER JOIN relations on relations.tag_uuid = tr.tag_uuid_denoted
    INNER JOIN tags on tags.uuid = tr.tag_uuid
)
SELECT * from relations;
"""


def select_related_tags(conn: sqlite3.Connection, tag) -> Set:
    """Select tags associated with the given tag."""
    tags = {tag}
    with conn:
        results = conn.execute(SELECT_TAG_RELATIONS, (tag,))
        tags.update({r["tag"] for r in results})
    return tags


def expand_tag_groups(
    conn: sqlite3.Connection, tag_groups: List[List[str]]
) -> List[Tuple[str, ...]]:
    """
    Given a set of "tag groups" (a list of lists of tags), expand those tags to
    include related tags, while maintaing the appropriate AND and OR
    relationships between the groups.

    We operate with the assumption that the sub-lists shall be OR'd together
    while the tags within each sub-list shall be AND'd.
    """
    expanded_groups = []
    for tag_group in tag_groups:
        related_groups = []
        for tag in tag_group:
            # Expand the tag into it's related tags
            related_groups.append(select_related_tags(conn, tag))

        # Find the product of the groups of related tags.
        expanded_groups.extend(list(itertools.product(*related_groups)))

    # Due to tag-associations, it's possible for an expanded sub-tuple to
    # consist of a repeated tag. We'll take a step here to reduce these groups
    # as there is no benefit to this repetition.
    groups: List[Tuple[str, ...]] = [
        group if len(set(group)) > 1 else (group[0],) for group in expanded_groups
    ]

    return groups


SELECT_TAG_STATISTICS = """
WITH RECURSIVE relations (tag_FROM, tag, tag_uuid, tag_uuid_denoted) AS (
    SELECT tags_FROM.tag, tags.tag, tr.tag_uuid, tr.tag_uuid_denoted
        FROM tag_relations tr
    INNER JOIN tags AS tags_FROM ON tags_FROM.uuid = tr.tag_uuid_denoted
    INNER JOIN tags ON tags.uuid = tr.tag_uuid

    UNION

    SELECT relations.tag_FROM, tags.tag, tr.tag_uuid, tr.tag_uuid_denoted
        FROM tag_relations tr
    INNER JOIN relations ON relations.tag_uuid = tr.tag_uuid_denoted
    INNER JOIN tags ON tags.uuid = tr.tag_uuid
)

SELECT
    t1.tag,
    COALESCE((
        SELECT
            COUNT(*)
        FROM tags
            INNER JOIN logs_tags lt ON lt.tag_uuid = tags.uuid
            INNER JOIN logs ON logs.uuid = lt.log_uuid
        WHERE tags.tag = t1.tag
        GROUP BY tags.tag
    ), 0) AS direct,
    COALESCE((
        SELECT
            count(*) AS cnt
        FROM logs
        INNER JOIN logs_tags lt ON logs.uuid = lt.log_uuid
        INNER JOIN tags ON lt.tag_uuid = tags.uuid
        WHERE tags.tag in (
            SELECT tag
            FROM relations
            WHERE tag_FROM = t1.tag
        )
    ), 0) AS implied,
    COALESCE(REPLACE(GROUP_CONCAT(DISTINCT r.tag), ',', ', '), '') AS children,
    COALESCE(REPLACE(GROUP_CONCAT(DISTINCT r2.tag_FROM), ',', ', '), '') AS implies
FROM
    tags t1
LEFT JOIN
    relations r ON r.tag_FROM = t1.tag
LEFT JOIN
    relations r2 ON r2.tag = t1.tag
GROUP BY t1.tag
ORDER BY implied DESC, direct DESC, t1.tag ASC;
"""


def tag_statistics(conn: sqlite3.Connection) -> sqlite3.Cursor:
    with conn:
        results = conn.execute(SELECT_TAG_STATISTICS)
    return results


# -----------------------------------------------------------------------------
# Rendering and Diffing logs


class RenderedLog:

    _TAG_REGEX: Pattern[str] = re.compile(r".*\([Tt]ags:\s*(?P<tags>.*)\)")

    def __init__(
        self,
        notes: Iterable[Note],
        tag_groups: List[Tuple[str, ...]],
        expanded_tag_groups: List[Tuple[str, ...]],
        style: bool = True,
        top_header: bool = True,
    ):
        """
        logs: A list/tuple, of 2-tuples (uuid, message)
        tags: The tags used to find the given logs. A list of lists of tags.
        """
        self.notes = list(notes)
        self.tag_groups = tag_groups
        self.expanded_tag_groups = expanded_tag_groups
        self._tags_flat = flatten_tag_groups(tag_groups)
        self._styled = style
        self._lines = []
        self._line_map = []
        self._render(style, top_header)  # Set up self._lines and self._lines_map

    def _render(self, style: bool, top_header: bool):
        # Header
        if style and top_header and self.tag_groups:
            self._lines.append(RenderedLog._editor_header(self.expanded_tag_groups))

        # Body
        linenum_init, linenum_last = None, None
        for note in self.notes:
            # Set the header for each message.
            if style:
                self._lines.extend(RenderedLog._note_header(note))
                linenum_init = len(self._lines) - 1
            else:
                linenum_init = len(self._lines)

            self._lines.extend(note.body.splitlines(keepends=True))

            if style:
                self._lines.extend(RenderedLog._note_footer(note))

            linenum_last = len(self._lines)

            self._line_map.append((note.uuid, linenum_init, linenum_last))

        # Footer
        if style:
            self._lines.extend(RenderedLog._editor_footer(set(self._tags_flat)))

    @staticmethod
    def _editor_header(tag_groups: List[Tuple[str, ...]]):
        _tag_groups = (", ".join(group) for group in tag_groups)
        tags_together = " || ".join(f"<{tg}>" for tg in _tag_groups)
        header = f"# TAGS: {tags_together}\n"
        return header

    @staticmethod
    def _note_header(note: Note) -> Tuple[str, ...]:
        tags_str = "" if not note.tags else ", ".join(sorted(note.tags))
        id_str = str(note.uuid)[:8]  # Only show first eight digits of UUID
        header = (
            f'{79*"-"}\n',
            f"# {note.created_at}\n",
            f"[ID: {id_str}]: # (Tags: {tags_str})\n",
            f"\n",
        )
        return header

    @staticmethod
    def _note_footer(note: Note) -> Tuple[str, ...]:
        # Add a newline, but only if there's not already an empty line at the
        # end of the note body.
        if note.body[-2:] == "\n\n":
            return ()
        return ("\n",)

    @staticmethod
    def _editor_footer(tags: Iterable[str]) -> Tuple[str, ...]:
        footer = (
            f'{79*"-"}\n',
            f"# Enter new log message below\n",
            f'# Tags: {", ".join(tags)}\n',
            "\n",
        )
        return footer

    @property
    def rendered(self) -> List[str]:
        return self._lines

    @staticmethod
    def _is_addition(line: str) -> bool:
        return line.startswith("+ ")

    @staticmethod
    def _is_removal(line: str) -> bool:
        return line.startswith("- ")

    @staticmethod
    def _is_intraline(line: str) -> bool:
        return line.startswith("? ")

    @staticmethod
    def _is_emptyline(line: str) -> bool:
        return line == "  \n"

    @staticmethod
    def _is_modification(line: str) -> bool:
        return RenderedLog._is_addition(line) or RenderedLog._is_removal(line)

    @staticmethod
    def _enumerate_diff(
        diff_lines: Iterable[str],
    ) -> Generator[Tuple[int, str], None, None]:
        first_line = True
        line_num = 0

        for line in diff_lines:
            if RenderedLog._is_intraline(line):
                # These intraline differences are not needed.
                continue

            if RenderedLog._is_addition(line):
                yield (line_num, line)
            else:
                if first_line:
                    yield (line_num, line)
                    first_line = False
                else:
                    line_num += 1
                    yield (line_num, line)

    @staticmethod
    def _print_diff_info(
        line_num: int,
        msg_uuid: uuid.UUID,
        line_from: int,
        line_to: int,
        text: str,
        debug: bool = False,
    ):
        if debug:
            print(
                (
                    f"line: {line_num:>4}, msg_uuid: {str(msg_uuid)},"
                    f" ({str(line_from):>4}, {str(line_to):>4}): {text}"
                ),
                end="",
            )

    @staticmethod
    def _is_new_tag_line(line: str) -> bool:
        TAG_LINE = "+ # Tags:"
        return line.startswith(TAG_LINE)

    @classmethod
    def _parse_tags(cls, line) -> Union[None, Set[str]]:
        match = cls._TAG_REGEX.match(line)
        if match is None:
            return None
        raw_tags = (t.strip() for t in match["tags"].split(","))
        return {t for t in raw_tags if t}

    @staticmethod
    def _parse_new_tags(line: str) -> Set[str]:
        TAG_LINE = "+ # Tags:"
        raw_tags = (t.strip() for t in line[len(TAG_LINE) :].split(","))
        return {t for t in raw_tags if t}

    def diff(self, other: Iterable[str], debug=False) -> List["LogDiff"]:
        """
        return an iterable of LogDiffs
        """
        line_num, diff_index = 0, 0
        msg_diff = []
        log_diffs: List[LogDiff] = []

        diff = difflib.ndiff(self._lines, list(other))
        diff = list(RenderedLog._enumerate_diff(diff))

        line_num, text = diff[diff_index]

        for msg_uuid, line_from, line_to in self._line_map:
            tags_original, tags_updated = None, None

            advance = 0
            for line_num, text in diff[diff_index:]:
                if line_num < line_from:
                    # Check for tag changes
                    tags = RenderedLog._parse_tags(text)
                    if tags is not None:
                        if text.startswith("-"):
                            tags_original = tags
                        elif text.startswith("+"):
                            tags_updated = tags
                        else:
                            tags_original = tags
                elif line_num == line_from:
                    # Handle leading synthetic newline
                    if self._styled:
                        if RenderedLog._is_addition(text):
                            msg_diff.append(text)
                    else:
                        msg_diff.append(text)
                elif line_from < line_num < (line_to - 1):
                    # Handle body of note
                    msg_diff.append(text)
                elif line_num == (line_to - 1):
                    # Handle trailing synthetic newline
                    if self._styled:
                        if RenderedLog._is_addition(text):
                            msg_diff.append(text)
                    else:
                        msg_diff.append(text)
                elif line_to <= line_num:
                    break

                RenderedLog._print_diff_info(
                    line_num, msg_uuid, line_from, line_to, text, debug=debug
                )

                advance += 1

            diff_index += advance

            # TODO: Refactor LogDiff so that lines are iteratively given to it.
            log_diffs.append(
                LogDiff(
                    msg_uuid,
                    msg_diff,
                    tags_original=tags_original,
                    tags_updated=tags_updated,
                )
            )
            msg_diff = []

        # New msg
        new_tags = set(self._tags_flat)
        for line_num, text in diff[diff_index:]:
            RenderedLog._print_diff_info(line_num, None, None, None, text, debug=debug)
            if RenderedLog._is_new_tag_line(text):
                new_tags = RenderedLog._parse_new_tags(text)
            elif RenderedLog._is_addition(text):
                msg_diff.append(text)

        # Create and append the new msg, if it exists
        if msg_diff:
            log_diffs.append(LogDiff(None, msg_diff, tags_original=new_tags))

        return log_diffs


class LogDiff:
    def __init__(
        self,
        msg_uuid: uuid.UUID,
        diff_lines: List[str],
        tags_original: Union[Set[str], None] = None,
        tags_updated: Union[Set[str], None] = None,
    ):
        """
        mods: iterable of (change, line_num, text)
        """
        self.msg_uuid = msg_uuid
        self.msg = "".join(difflib.restore(diff_lines, 2))
        self.diff = diff_lines
        self._note_modified = any(
            (line.startswith("- ") or line.startswith("+ ") for line in diff_lines)
        )
        self.is_new = msg_uuid is None

        self.tags_original = tags_original
        self.tags_updated = tags_updated
        self._tags_modified = tags_updated is not None and (
            tags_original != tags_updated
        )

    def __str__(self):
        return "".join(self.diff)

    def __repr__(self):
        id_str = str(self.msg_uuid) if not self.is_new else "New"
        return f"<LogDiff({id_str})>\n{str(self)}</LogDiff>"

    @property
    def modified(self):
        return self._note_modified or self._tags_modified

    @property
    def tags(self):
        if self.tags_updated is not None:
            return self.tags_updated
        return self.tags_original or set()

    def update_or_create(self, conn: sqlite3.Connection, commit: bool = True):
        if self.is_new:
            return self._create(conn, commit=commit)
        else:
            return self._update(conn, commit=commit)

    def _create(self, conn: sqlite3.Connection, commit: bool = True):
        msg_uuid = insert_msg(conn, self.msg)
        self.msg_uuid = msg_uuid

        tag_uuids = insert_tags(conn, self.tags)
        insert_asscs(conn, self.msg_uuid, tag_uuids)

        if commit:
            conn.commit()

        return True

    def _update(self, conn: sqlite3.Connection, commit: bool = True):
        if not self.modified:
            return True

        if not self.msg:
            # TODO: delete msg or mark as deleted?
            pass

        if self._note_modified:
            if not self._update_msg(conn):
                # TODO: Maybe throw a custom exception?
                return False

        if self._tags_modified:
            if not self._update_tags(conn):
                return False

        if not self._update_diffs(conn):
            # TODO: Rollback? Throw exception?
            return False

        # Allow commit to be defered
        if commit:
            conn.commit()

        return True

    def _update_msg(self, conn: sqlite3.Connection):
        return update_msg(conn, self.msg_uuid, self.msg)

    def _update_tags(self, conn: sqlite3.Connection):
        tags_add = self.tags_updated - self.tags_original
        if tags_add:
            tag_uuids = insert_tags(conn, tags_add)
            insert_asscs(conn, self.msg_uuid, tag_uuids)

        tags_sub = self.tags_original - self.tags_updated
        if tags_sub:
            tag_uuids = {t[0] for t in select_tags(conn, tuple(tags_sub))}
            remove_asscs(conn, self.msg_uuid, tag_uuids)

        return True

    def _update_diffs(self, conn: sqlite3.Connection):
        # TODO: Save diff info
        return True


# -----------------------------------------------------------------------------


def handle_tag_associate(
    conn: sqlite3.Connection,
    to_associate: Iterable[Tuple[str, str]],
    quiet: bool = False,
) -> Tuple[int, int]:
    inserted, existing = 0, 0
    for explicit, implicit in to_associate:
        try:
            insert_tag_relation(conn, explicit, implicit, quiet=True)
        except LgdException as e:
            print(Term.warning(str(e)))
        except sqlite3.IntegrityError as e:
            existing += 1
            if "unique" in str(e).lower() and not quiet:
                print(
                    Term.warning(
                        f"Tag relation '{explicit}' -> '{implicit}' already exists!"
                    )
                )
        else:
            inserted += 1
            if not quiet:
                print(Term.green(f"Created '{explicit}' -> '{implicit}' relation"))
    return (inserted, existing)


def handle_tag_disassociate(
    conn: sqlite3.Connection, to_disassociate: Iterable[List[str]]
) -> None:
    for explicit, implicit in to_disassociate:
        try:
            removed = remove_tag_relation(conn, explicit, implicit)
        except LgdException as e:
            print(Term.warning(str(e)))
        else:
            if removed:
                print(Term.green(f"Removed '{explicit}' -> '{implicit}' relation"))
            else:
                print(
                    Term.warning(
                        f"Relation '{explicit}' -> '{implicit}' doesn't exist!"
                    )
                )


def note_export(
    conn: sqlite3.Connection, notes: List[Note], outfile: io.TextIOWrapper
) -> int:
    writer = csv.DictWriter(outfile, Note._fields)
    writer.writeheader()
    for note in notes:
        # For the CSV file, for the tags to be a comma separated str.
        note = note._replace(tags=",".join(note.tags))
        writer.writerow(note._asdict())

    return len(notes)


def note_import(conn: sqlite3.Connection, infile: io.TextIOWrapper) -> Tuple[int, int]:
    inserted = 0
    updated = 0

    reader = csv.DictReader(infile)
    if set(reader.fieldnames) != set(Note._fields):
        raise CSVError(
            "Invalid CSV columns; columns must be: uuid,created_at,body,tags"
        )

    for row in reader:
        try:
            row[CREATED_AT] = datetime.strptime(row[CREATED_AT], "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise CSVError(
                "Invalid 'created_at' format; timestamp must be 'YYYY-MM-DD HH:MM:SS'."
            ) from e

        try:
            row[ID] = uuid.UUID(row[ID])
        except ValueError as e:
            raise CSVError(
                "Invalid 'uuid' format; value must be string of hex digits. Curly braces, hyphens, and a URN prefix are all optional."
            ) from e

        try:
            note = Note(**row)
        except TypeError as e:
            raise CSVError(
                "Invalid CSV columns; columns must be: uuid,created_at,body,tags"
            ) from e

        tags = note.tags.split(",")

        if msg_exists(conn, note.uuid):
            # Update
            updated += int(update_msg(conn, note.uuid, note.body))
        else:
            # Insert
            _ = insert_msg(
                conn,
                note.body,
                msg_uuid=note.uuid,
                created_at=note.created_at,
            )
            inserted += 1

        tag_uuids = insert_tags(conn, tags)
        insert_asscs(conn, note.uuid, tag_uuids)

    return (inserted, updated)


def tag_export(conn: sqlite3.Connection, outfile: io.TextIOWrapper) -> int:
    tag_relations = select_related_tags_all(conn)
    writer = csv.writer(outfile)
    writer.writerow(("tag_direct", "tag_indirect"))
    writer.writerows(tag_relations)
    return len(tag_relations)


def tag_import(conn: sqlite3.Connection, infile: io.TextIOWrapper) -> Tuple[int, int]:
    reader = csv.DictReader(infile)
    relations = ((row["tag_direct"], row["tag_indirect"]) for row in reader)
    return handle_tag_associate(conn, relations, quiet=True)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    args.date_ranges = to_datetime_ranges(args.date_ranges)

    dir_setup()
    conn = get_connection(str(DB_PATH))

    if not db_setup(conn, DB_MIGRATIONS):
        print("Failed to finish database setup!")
        sys.exit()

    # ------------------------------------------------------------------------
    # Apply the users filters to select the notes/messages
    tag_groups = [tg for tg in (args.tags or []) if tg]
    expanded_tag_groups = expand_tag_groups(conn, tag_groups)

    messages = select_notes(
        conn,
        tag_groups=expanded_tag_groups,
        date_ranges=args.date_ranges,
        text=args.search,
    )

    if args.note_file_in:
        with open(args.note_file_in, "r") as infile:
            try:
                inserted, updated = note_import(conn, infile)
            except CSVError as e:
                sys.exit(Term.error(str(e)))
        print(
            f" - Inserted {inserted}, updated {updated} notes from {args.note_file_in}"
        )

    if args.tag_file_in:
        with open(args.tag_file_in, "r") as infile:
            inserted, existing = tag_import(conn, infile)
            total = inserted + existing
        print(f" - Inserted {inserted} of {total} relations from {args.tag_file_in}")

    if args.note_file_out:
        with open(args.note_file_out, "w") as outfile:
            num = note_export(conn, messages, outfile)
        print(f" - Exported {num} notes to {args.note_file_out}")

    if args.tag_file_out:
        with open(args.tag_file_out, "w") as outfile:
            num = tag_export(conn, outfile)
        print(f" - Exported {num} tag relations to {args.tag_file_out}")

    if args.tag_stats:
        stats = format_tag_statistics(tag_statistics(conn))
        for line in stats:
            print(line)

    if args.tag_associate or args.tag_disassociate:
        handle_tag_associate(conn, (args.tag_associate or []))
        handle_tag_disassociate(conn, (args.tag_disassociate or []))

    if args.delete is not None:
        ui_delete_notes(conn, args.delete, args.confirmation_override)

    if any(
        (
            args.note_file_in,
            args.note_file_out,
            args.tag_file_in,
            args.tag_file_out,
            args.tag_stats,
            args.delete,
        )
    ):
        sys.exit()

    # ------------------------------------------------------------------------
    # If reading from stdin
    if args.dash:
        msg = stdin_note()
        msg_uuid = insert_msg(conn, msg)
        if args.tags:
            tags = flatten_tag_groups(args.tags)
            tag_uuids = insert_tags(conn, tags)
            insert_asscs(conn, msg_uuid, tag_uuids)

        print(f"Saved as message ID {msg_uuid}")
        conn.close()
        sys.exit()

    # ------------------------------------------------------------------------
    # Print Note metadata (uuid, created_at, tags)
    if args.metadata:
        for msg in messages:
            print(get_metadata(msg))
        sys.exit()

    # ------------------------------------------------------------------------
    # Display notes matching filters
    if args.tags or args.date_ranges or args.search:

        # The users editor is opened here, and may be open for an extended
        # period of time. Close the database and get a new connection
        # afterward.
        conn.close()

        message_view = RenderedLog(
            messages, tag_groups, expanded_tag_groups, style=(not args.plain)
        )

        new_note = None
        notifications = []

        for body_lines in editor(message_view.rendered):
            diffs = message_view.diff(body_lines, debug=DEBUG)
            if diffs:
                if diffs[-1].is_new:
                    new_note = diffs[-1]
                elif new_note is not None:
                    new_note = None

            # Produce the note diffs. We filter out the new note, if it
            # exists, because we don't want to defer it's creation until
            # the editor process has exited.
            changes = [d for d in diffs if d.modified and not d.is_new]

            if not changes:
                continue

            conn = get_connection(str(DB_PATH))

            for change in changes:
                if DEBUG:
                    notifications.append(repr(change))

                # TODO: Delete msg if all lines removed?
                change.update_or_create(conn, commit=False)
                notifications.append(f"Saved changes to message ID {change.msg_uuid}")

            conn.commit()

            # Reset the RenderedLog to account for the modified notes.
            message_view = RenderedLog(
                select_notes(conn, uuids=[m.uuid for m in messages]),
                tag_groups,
                expanded_tag_groups,
                style=(not args.plain),
            )

            conn.close()

        if new_note is not None:
            conn = get_connection(str(DB_PATH))
            new_note.update_or_create(conn, commit=True)
            notifications.append(f"Saved additional message as ID {new_note.msg_uuid}")
            conn.close()

        # Messages / notification to user are deferred until the editor is
        # closed, so that output to stdout (via print) will not interfere
        # with terminal editors.
        for notification in notifications:
            print(notification)

        sys.exit()

    # ------------------------------------------------------------------------
    # Quick note
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
    conn.close()

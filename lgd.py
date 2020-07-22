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
import sys
import sqlite3
import tempfile
import uuid

from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import call
from typing import Callable, FrozenSet, Iterable, List, Pattern, Set, Tuple, Union


EDITOR = os.environ.get("EDITOR", "vim")
DEBUG = False


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
        " Ex. `-s foo`, `-s foo bar`.\n"
        " Additional flag usage will OR the tag groups together.\n"
        " Ex. `-s foo bar -s baz`.\n"
    ),
)
parser.add_argument(
    "-D",
    "--delete",
    action="store",
    type=str,
    help="Delete the message with the given ID.",
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
    "-T",
    "--tags",
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
    created_at int NOT NULL,
    msg GZIP NOT NULL
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
    FOREIGN KEY (tag_uuid) REFERENCES tags(uuid) ON DELETE CASCADE,
    UNIQUE(log_uuid, tag_uuid)
);
"""

CREATE_ASSC_LOGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_log_index ON logs_tags (log_uuid);
"""

CREATE_ASSC_TAGS_INDEX = """
CREATE INDEX IF NOT EXISTS assc_tag_index ON logs_tags (tag_uuid);
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


def get_connection(db_path: str) -> sqlite3.Connection:
    # This creates the sqlite db if it doesn't exist.
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    # Register adapters and converters.
    sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes)
    sqlite3.register_converter("UUID", lambda b: uuid.UUID(bytes=b))
    sqlite3.register_adapter(Gzip, lambda s: Gzip.compress_string(s))
    sqlite3.register_converter(Gzip.COL_TYPE, lambda b: Gzip.decompress_string(b))

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
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

    # Ensure tags table
    conn.execute(CREATE_TAGS_TABLE)
    conn.execute(CREATE_TAG_INDEX)

    # Ensure association table
    conn.execute(CREATE_ASSOC_TABLE)
    conn.execute(CREATE_ASSC_LOGS_INDEX)
    conn.execute(CREATE_ASSC_TAGS_INDEX)

    # Ensure tag relations table
    conn.execute(CREATE_TAG_RELATIONS_TABLE)


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
# Autocompleting Tag Prompt


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


# -----------------------------------------------------------------------------
# Terminal Color Helpers


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


# -----------------------------------------------------------------------------
# Misc. Utilities


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


def flatten_tag_groups(tag_groups: Tuple[Tuple[str, ...]]) -> List:
    tags = []
    for group in tag_groups:
        tags.extend(group)
    return tags


def prompt_for_delete(msg: sqlite3.Row, uuid_prefix: str) -> bool:
    # TODO: Improve uuid highlighting. Currently doesn't work for whole uuids.
    uuid_fragment = Term.apply_where(Term.green, uuid_prefix, str(msg[ID])[:8])
    msg_fragment = msg[MSG][:46].replace("\n", "\\n")

    prompt = f'{Term.warning("Delete")} {uuid_fragment}..., "{msg_fragment}..." (Y/n) '
    return input(prompt).lower() == "y"


def open_temp_logfile(lines: Union[List[str], None] = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
        if lines:
            tf.writelines(line.encode("utf8") for line in lines)
            tf.flush()
        tf.close()

        call([EDITOR, tf.name])

        with open(tf.name) as f:
            contents = f.read()

        os.unlink(tf.name)

    return contents


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


def split_tags(tags: str) -> FrozenSet[str]:
    return frozenset(t.strip() for t in tags.split(","))


def rows_to_notes(rows):
    return (
        Note(r["uuid"], r["created_at"], r["body"], split_tags(r["tags"])) for r in rows
    )


# -----------------------------------------------------------------------------
# Types

Note = namedtuple("Note", ("uuid", "created_at", "body", "tags"))

Tag = namedtuple("Tag", ("uuid", "value"))

TagRelation = namedtuple("TagRelation", ("direct", "indirect"))


# -----------------------------------------------------------------------------
# SQL queries and related functions


class LgdException(Exception):
    pass


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


UPDATE_LOG = """
UPDATE logs SET msg = ? WHERE uuid = ?
"""


def update_msg(conn: sqlite3.Connection, msg_uuid: uuid.UUID, msg: str) -> bool:
    c = conn.execute(UPDATE_LOG, (Gzip(msg), msg_uuid))
    return c.rowcount == 1


AND_DATE_BETWEEN_TEMPL = """
 AND ({ranges})
"""

_DATE_BETWEEN_FRAGMENT = """
({column} BETWEEN '{begin}' AND '{end}')
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
    logs.uuid AS uuid,
    datetime(logs.created_at{datetime_modifier}) AS created_at,
    logs.msg AS body,
    group_concat(tags.tag) AS tags
FROM logs
LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
LEFT JOIN tags ON tags.uuid = lt.tag_uuid
WHERE 1{date_range}
GROUP BY logs.uuid, logs.created_at, logs.msg
ORDER BY logs.created_at;
"""

SELECT_LOGS_AND_TAGS_TEMPL = """
SELECT
    logs.uuid AS uuid,
    datetime(logs.created_at{datetime_modifier}) AS created_at,
    logs.msg AS body,
    group_concat(tags.tag) AS tags
FROM logs
LEFT JOIN logs_tags lt ON lt.log_uuid = logs.uuid
LEFT JOIN tags ON tags.uuid = lt.tag_uuid
WHERE logs.uuid in ({msgs})
GROUP BY logs.uuid, logs.created_at, logs.msg
ORDER BY logs.created_at;
"""


def _format_date_range(column: str, date_ranges) -> str:
    if not date_ranges:
        return ""

    ranges = " OR ".join(
        _DATE_BETWEEN_FRAGMENT.format(
            column=column, begin=date_range[0], end=date_range[1]
        )
        for date_range in date_ranges
    )

    return AND_DATE_BETWEEN_TEMPL.format(ranges=ranges)


def format_template_tags_dates(template: str, tags, date_col, date_range) -> str:
    tags = ", ".join("?" for _ in tags)
    dates = _format_date_range(date_col, date_range)
    return template.format(tags=tags, date_range=dates)


def _msg_uuids_having_tags(
    conn: sqlite3.Connection, tag_groups, date_range=None
) -> Set[uuid.UUID]:
    msg_uuids = set()  # using a set in order to de-duplicate.

    for tags in tag_groups:
        select = format_template_tags_dates(
            SELECT_LOGS_HAVING_TAGS_TEMPL,
            tags,
            date_col="logs.created_at",
            date_range=date_range,
        )
        for row in conn.execute(select, (*tags, len(tags))):
            msg_uuids.add(row[ID])

    return msg_uuids


def messages_with_tags(
    conn: sqlite3.Connection, tag_groups, date_range=None, localtime=True
) -> List[Note]:
    if not tag_groups or ((len(tag_groups) == 1) and not tag_groups[0]):
        select = SELECT_LOGS_WITH_TAGS_ALL_TEMPL.format(
            date_range=_format_date_range("logs.created_at", date_range),
            datetime_modifier=", 'localtime'" if localtime else "",
        )
        cursor = conn.execute(select)
    else:
        msg_uuids = _msg_uuids_having_tags(conn, tag_groups, date_range=date_range)
        select = SELECT_LOGS_AND_TAGS_TEMPL.format(
            msgs=", ".join("?" for _ in msg_uuids),
            datetime_modifier=", 'localtime'" if localtime else "",
        )
        cursor = conn.execute(select, tuple(msg_uuids))

    return list(rows_to_notes(cursor.fetchall()))


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


def select_tags(conn: sqlite3.Connection, tags: Iterable) -> List[sqlite3.Row]:
    tag_snippet = ", ".join("?" for _ in tags)
    sql = f"SELECT * FROM tags WHERE tag in ({tag_snippet})"
    c = conn.execute(sql, tags)
    return c.fetchall()


INSERT_TAG = """
INSERT OR IGNORE INTO tags (uuid, tag) VALUES (?, ?);
"""


def insert_tags(conn: sqlite3.Connection, tags):
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


def select_all_tags(conn: sqlite3.Connection) -> List[str]:
    c = conn.execute("SELECT tags.tag FROM tags;")
    return [r[0] for r in c.fetchall()]


# Log-Tag associations

INSERT_LOG_TAG_ASSC = """
INSERT INTO logs_tags (log_uuid, tag_uuid) VALUES (?, ?);
"""


def insert_asscs(conn: sqlite3.Connection, msg_uuid, tag_uuids):
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
WITH RECURSIVE relations (tag, tag_uuid, tag_uuid_denoted) AS (
  SELECT tags.tag, tr.tag_uuid, tr.tag_uuid_denoted
  FROM tag_relations tr
    INNER JOIN tags as tags_from on tags_from.uuid = tr.tag_uuid_denoted
    INNER JOIN tags on tags.uuid = tr.tag_uuid
  WHERE tags_from.tag = ?

  UNION

  SELECT tags.tag, tr.tag_uuid, tr.tag_uuid_denoted
  FROM tag_relations tr
    INNER JOIN relations on relations.tag_uuid = tr.tag_uuid_denoted
    INNER JOIN tags on tags.uuid = tr.tag_uuid
)
SELECT tag from relations;
"""


def select_related_tags(conn: sqlite3.Connection, tag):
    """Select tags associated with the given tag."""
    tags = {tag}
    with conn:
        results = conn.execute(SELECT_TAG_RELATIONS, (tag,))
        tags.update({r["tag"] for r in results})
    return tags


def expand_tag_groups(conn: sqlite3.Connection, tag_groups):
    """
    Given a set of "tag groups" (a list of lists of tags), expand those tags to
    include related tags, while maintaing the appropriate AND and OR
    relationships between the groups.
    """
    expanded_groups = []
    for tag_group in tag_groups:
        related_groups = []
        for tag in tag_group:
            # Expand the tag into it's related tags
            related_groups.append(select_related_tags(conn, tag))

        # Find the product of the groups of related tags.
        expanded_groups.extend(list(itertools.product(*related_groups)))

    return expanded_groups


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
    def __init__(self, notes, tags, style=True):
        """
        logs: A list/tuple, of 2-tuples (uuid, message)
        tags: The tags used to find the given logs. A list of lists of tags.
        """
        self.notes = list(notes)
        self.tag_groups = list(tags) if tags else tuple()
        self._styled = style
        self._all_tags = flatten_tag_groups(self.tag_groups)
        self._lines = []
        self._line_map = []
        self._render(style)  # Set up self._lines and self._lines_map

    def _render(self, style):
        # Header
        if style and self.tag_groups:
            self._lines.append(RenderedLog._editor_header(self.tag_groups))

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
            self._lines.extend(RenderedLog._editor_footer(set(self._all_tags)))

    @staticmethod
    def _editor_header(tag_groups):
        tag_groups = (", ".join(group) for group in tag_groups)
        tags_together = " || ".join(f"<{tg}>" for tg in tag_groups)
        header = f"# TAGS: {tags_together}\n"
        return header

    @staticmethod
    def _note_header(note: Note):
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
    def _note_footer(note: Note):
        return ("\n",)

    @staticmethod
    def _editor_footer(tags):
        footer = (
            f'{79*"-"}\n',
            f"# Enter new log message below\n",
            f'# Tags: {", ".join(tags)}\n',
            "\n",
        )
        return footer

    @property
    def rendered(self):
        return self._lines

    @staticmethod
    def _is_addition(line):
        return line.startswith("+ ")

    @staticmethod
    def _is_removal(line):
        return line.startswith("- ")

    @staticmethod
    def _is_intraline(line):
        return line.startswith("? ")

    @staticmethod
    def _is_emptyline(line):
        return line == "  \n"

    @staticmethod
    def _is_modification(line):
        return RenderedLog._is_addition(line) or RenderedLog._is_removal(line)

    @staticmethod
    def _enumerate_diff(diff_lines):
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
    def _print_diff_info(line_num, msg_uuid, line_from, line_to, text, debug=False):
        msg_uuid = str(msg_uuid)
        line_from = str(line_from)
        line_to = str(line_to)
        if debug:
            print(
                (
                    f"line: {line_num:>4}, msg_uuid: {msg_uuid},"
                    f" ({line_from:>4}, {line_to:>4}): {text}"
                ),
                end="",
            )

    @staticmethod
    def _is_new_tag_line(line):
        TAG_LINE = "+ # Tags:"
        return line.startswith(TAG_LINE)

    @staticmethod
    def _parse_new_tags(line):
        TAG_LINE = "+ # Tags:"
        raw_tags = (t.strip() for t in line[len(TAG_LINE) :].split(","))
        return [t for t in raw_tags if t]

    def diff(self, other, debug=False):
        """
        return an iterable of LogDiffs
        """
        line_num, diff_index = 0, 0
        msg_diff, log_diffs = [], []

        diff = difflib.ndiff(self._lines, list(other))
        diff = list(RenderedLog._enumerate_diff(diff))

        line_num, text = diff[diff_index]

        for msg_uuid, line_from, line_to in self._line_map:
            advance = 0
            for line_num, text in diff[diff_index:]:
                if line_num < line_from:
                    # Do nothing
                    pass
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

                if RenderedLog._is_new_tag_line(text):
                    print(
                        f"New tags for msg '{msg_uuid}', {RenderedLog._parse_new_tags(text)}"
                    )

                advance += 1

            diff_index += advance

            # TODO: Refactor LogDiff so that lines are iteratively given to it.
            log_diffs.append(LogDiff(msg_uuid, msg_diff, tags=self._all_tags))
            msg_diff = []

        # New msg
        new_tags = self._all_tags
        for line_num, text in diff[diff_index:]:
            RenderedLog._print_diff_info(line_num, None, None, None, text, debug=debug)
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
        self.msg = "".join(difflib.restore(diff_lines, 2))
        self.diff = diff_lines
        self.modified = any(
            (line.startswith("- ") or line.startswith("+ ") for line in diff_lines)
        )
        self.is_new = msg_uuid is None
        self.tags = tags if tags else []

    def __str__(self):
        return "".join(self.diff)

    def __repr__(self):
        id_str = str(self.msg_uuid) if not self.is_new else "New"
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
        return update_msg(conn, self.msg_uuid, self.msg)

    def _update_diffs(self, conn):
        # TODO: Save diff info
        return True


# -----------------------------------------------------------------------------


def handle_tag_associate(
    conn: sqlite3.Connection, to_associate, quiet=False
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


def handle_tag_disassociate(conn: sqlite3.Connection, to_disassociate) -> None:
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


def note_export(conn: sqlite3.Connection, outfile: io.TextIOWrapper) -> int:
    notes = messages_with_tags(conn, None, localtime=False)

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
    for row in reader:
        note = Note(**row)
        note = note._replace(uuid=uuid.UUID(note.uuid))
        tags = note.tags.split(",")

        if msg_exists(conn, note.uuid):
            # Update
            updated += int(update_msg(conn, note.uuid, note.body))
        else:
            # Insert
            _ = insert_msg(
                conn, note.body, msg_uuid=note.uuid, created_at=note.created_at,
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

    if args.note_file_out:
        with open(args.note_file_out, "w") as outfile:
            num = note_export(conn, outfile)
        print(f" - Exported {num} notes to {args.note_file_out}")
        sys.exit()

    if args.note_file_in:
        with open(args.note_file_in, "r") as infile:
            inserted, updated = note_import(conn, infile)
        print(
            f" - Inserted {inserted}, updated {updated} notes from {args.note_file_in}"
        )
        sys.exit()

    if args.tag_file_out:
        with open(args.tag_file_out, "w") as outfile:
            num = tag_export(conn, outfile)
        print(f" - Exported {num} tag relations to {args.tag_file_out}")
        sys.exit()

    if args.tag_file_in:
        with open(args.tag_file_in, "r") as infile:
            inserted, existing = tag_import(conn, infile)
            total = inserted + existing
        print(f" - Inserted {inserted} of {total} relations from {args.tag_file_in}")
        sys.exit()

    if args.tag_stats:
        stats = format_tag_statistics(tag_statistics(conn))
        for line in stats:
            print(line)
        sys.exit()

    if args.tag_associate or args.tag_disassociate:
        handle_tag_associate(conn, (args.tag_associate or []))
        handle_tag_disassociate(conn, (args.tag_disassociate or []))
        sys.exit()

    if args.delete is not None:
        uuid_prefix = args.delete.replace("-", "")
        msgs = select_msgs_from_uuid_prefix(conn, uuid_prefix)
        if msgs:
            for msg in msgs:
                try:
                    confirmed = prompt_for_delete(msg, uuid_prefix)
                except EOFError:
                    print("")
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
            print(line, end="")
            lines.append(line)

        msg = "".join(lines)
        msg_uuid = insert_msg(conn, msg)
        if args.tags:
            tags = flatten_tag_groups(args.tags)
            tag_uuids = insert_tags(conn, tags)
            insert_asscs(conn, msg_uuid, tag_uuids)

        print(f"Saved as message ID {msg_uuid}")
        sys.exit()

    # Display messages
    if args.tags or args.date_ranges:
        tag_groups = [tg for tg in args.tags if tg]
        expanded_tag_groups = expand_tag_groups(conn, tag_groups)

        messages = messages_with_tags(conn, expanded_tag_groups, args.date_ranges)
        message_view = RenderedLog(
            messages, expanded_tag_groups, style=(not args.plain)
        )
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

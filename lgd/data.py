import gzip
import itertools
import sqlite3
import uuid

from collections import namedtuple
from datetime import datetime
from typing import (
    Any,
    Callable,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from lgd.exceptions import LgdException


# -----------------------------------------------------------------------------
# Types

Note = namedtuple("Note", ("uuid", "created_at", "body", "tags"))


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
            except sqlite3.Error:
                return False

            version = set_user_version(conn, version + 1)

    return True


# -----------------------------------------------------------------------------
# SQL queries and related functions


def split_tags(tags: Union[str, None]) -> FrozenSet[str]:
    return frozenset(t.strip() for t in tags.split(",")) if tags else frozenset()


def rows_to_notes(rows: List[sqlite3.Row]) -> Iterator[Note]:
    return (
        Note(r["uuid"], r["created_at"], r["body"], split_tags(r["tags"])) for r in rows
    )


# Log / Message related

INSERT_LOG = """
INSERT into logs (uuid, created_at, msg) VALUES (?, {timestamp}, ?);
"""


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


UPDATE_LOG = """
UPDATE logs SET msg = ? WHERE uuid = ?
"""


class NoteX:
    __slots__ = ("uuid", "created_at", "body", "tags")

    def __init__(self, uuid, created_at, body, tags):
        self.uuid = uuid
        self.created_at = created_at
        self.body = body
        self.tags = tags

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return ""


T = TypeVar("T")


class Query(Generic[T]):
    __slots__ = ("sql", "params", "adapter")

    def __init__(
        self,
        sql: str,
        params: Iterable[Any],
        adapter: Callable[[sqlite3.Cursor], T] = lambda x: x,
    ):
        """
        adapter:  callable accepting a cursor and returning some type.
        """
        self.sql = sql
        self.params = params
        self.adapter = adapter

    def __str__(self):
        return self.sql

    def __repr__(self):
        return f"Query({self.sql})"

    def execute(self, conn: sqlite3.Connection) -> T:
        result = conn.execute(self.sql, self.params)
        return self.adapter(result)


class NoteQuery:
    def __init__(self):
        pass

    def __str__(self):
        return "TODO: __str__"

    def __repr__(self):
        return "TODO: __repr__"

    @staticmethod
    def _uuid_filter(uuids: Union[None, List[uuid.UUID]]) -> Tuple[str, list]:
        uuid_filter = "1"
        params = []

        if uuids is not None:
            uuid_filter = _WHERE_UUIDS.format(uuids=", ".join("?" for _ in uuids))
            params.extend(uuids)

        return (uuid_filter, params)

    @staticmethod
    def _tag_groups_filter(
        tag_groups: Union[None, List[Tuple[str, ...]]]
    ) -> Tuple[str, list]:
        tags_filter = "1"
        params = []

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

        return (tags_filter, params)

    @staticmethod
    def _text_filter(text: Union[None, str]) -> Tuple[str, List[str]]:
        text_filter = "1"
        params = []

        if text:
            text_filter = _WHERE_FTS
            params.append(text)

        return (text_filter, params)

    @staticmethod
    def _date_filter(
        date_ranges: Union[None, List[Tuple[datetime, datetime]]]
    ) -> Tuple[str, list]:
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

        return (date_filter, [])

    @staticmethod
    def _localtime_correction(localtime: Union[None, bool]) -> str:
        datetime_modifier = ", 'localtime'" if localtime else ""
        return datetime_modifier

    @classmethod
    def insert(
        cls,
        note: str,
        note_uuid: Union[uuid.UUID, None] = None,
        created_at: Union[datetime, None] = None,
    ) -> Query[uuid.UUID]:
        """
        created_at is assumed to be in UTC.
        """
        note_uuid = uuid.uuid4() if note_uuid is None else note_uuid

        if created_at is None:
            insert = INSERT_LOG.format(timestamp="CURRENT_TIMESTAMP")
            params = (note_uuid, Gzip(note))
        else:
            insert = INSERT_LOG.format(timestamp="?")
            params = (note_uuid, created_at, Gzip(note))

        return Query(insert, params, adapter=lambda _: note_uuid)

    @classmethod
    def select(
        cls,
        uuids: Optional[List[uuid.UUID]] = None,
        tag_groups: Optional[List[Tuple[str, ...]]] = None,
        date_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
        text: Optional[str] = None,
        localtime: Optional[bool] = None,
    ) -> Query[Iterator[Note]]:
        uuid_filter, uuid_params = cls._uuid_filter(uuids)
        tags_filter, tags_params = cls._tag_groups_filter(tag_groups)
        text_filter, text_params = cls._text_filter(text)
        date_filter, date_params = cls._date_filter(date_ranges)
        datetime_modifier = cls._localtime_correction(localtime)

        params = list(
            itertools.chain(uuid_params, tags_params, text_params, date_params)
        )

        query = SELECT_NOTES_WHERE_TEMPL.format(
            uuid_filter=uuid_filter,
            tags_filter=tags_filter,
            text_filter=text_filter,
            date_filter=date_filter,
            datetime_modifier=datetime_modifier,
        )

        return Query(query, params, adapter=lambda c: rows_to_notes(c.fetchall()))

    @classmethod
    def update(cls, msg_uuid: uuid.UUID, msg: str) -> Query[bool]:
        return Query(
            UPDATE_LOG, (Gzip(msg), msg_uuid), adapter=lambda c: c.rowcount == 1
        )

    @classmethod
    def delete(cls, msg_uuid: uuid.UUID) -> Query[bool]:
        msg_delete = "DELETE FROM logs WHERE uuid = ?;"
        return Query(msg_delete, (msg_uuid,), adapter=lambda c: c.rowcount == 1)

    @classmethod
    def msg_exists(cls, msg_uuid: uuid.UUID) -> Query[bool]:
        sql = "SELECT uuid from logs where uuid = ?;"
        params = (msg_uuid,)
        return Query(sql, params, adapter=lambda c: c.fetchone() is not None)

    @classmethod
    def select_msgs_from_uuid_prefix(cls, uuid_prefix: str) -> Query:
        uuid_prefix += "%"
        sql = "SELECT * from logs where hex(uuid) like ?;"
        params = (uuid_prefix,)
        return Query(sql, params, adapter=lambda c: c.fetchall())

    @classmethod
    def associate_tags(cls, msg_uuid: uuid.UUID, tags: Iterator[str]) -> Query:
        return Query("", [])

    @classmethod
    def disassociate_tags(cls, msg_uuid: uuid.UUID, tags: Iterator[str]) -> Query:
        return Query("", [])


# class TagQuery:
#     def __init__(self):
#         pass

#     @classmethod
#     def select(cls, tag: str) -> Query[Union[sqlite3.Row, None]]:
#         pass

#     @classmethod
#     def select_all(cls) -> Query[List[str]]:
#         pass

#     @classmethod
#     def insert(cls, tags: Iterable[str]) -> Query[Set[uuid.UUID]]:
#         pass

#     @classmethod
#     def delete(cls, tag: str) -> Query[bool]:
#         pass


# -----------------------------------------------------------------------------
# Tags


def delete_tag(conn: sqlite3.Connection, tag: str) -> bool:
    """Delete the tag with the given value.

    propagate: If `True` (default), delete the associates to logs,
        but not the logs themselves.
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


def select_related_tags(conn: sqlite3.Connection, parent_tag: str) -> Set:
    """Select tags associated with the given tag.

    Returned tags are leaf nodes / child tags.
    """
    tags = {parent_tag}
    results = conn.execute(SELECT_TAG_RELATIONS, (parent_tag,))
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

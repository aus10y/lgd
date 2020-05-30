#!/usr/bin/env python3
import csv
import sqlite3
import sys
import uuid

from pathlib import Path


DB_PATH = Path.home() / Path('.lgd/logs.db')
CREATED_AT = 'created_at'
MSG = 'msg'
TAGS = 'tags'


def get_connection():
    # This creates the sqlite db if it doesn't exist.
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)

    # Register adapters and converters.
    sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes)
    sqlite3.register_converter('UUID', lambda b: uuid.UUID(bytes=b))

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


INSERT_LOG = """
INSERT into logs (uuid, created_at, msg) VALUES (?, ?, ?);
"""
def insert_msg(conn, msg, created_at):
    msg_uuid = uuid.uuid4()
    c = conn.execute(INSERT_LOG, (msg_uuid, created_at, msg))
    conn.commit()
    return msg_uuid


def select_tag(conn, tag: str):
    c = conn.execute("SELECT * FROM tags WHERE tag = ?", (tag,))
    return c.fetchone()


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Must give an input file name...")
        sys.exit()

    filename = sys.argv[1]

    if not DB_PATH.exists():
        print(f"Could not find '{DB_PATH}'")
        sys.exit()

    con = get_connection()

    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            msg = row[MSG]
            created_at = row[CREATED_AT]
            tags = row[TAGS].split(',')

            print(f"{msg[:10]}, {created_at}, {tags}")

            msg_uuid = insert_msg(con, msg, created_at)
            tag_uuids = insert_tags(con, tags)
            insert_asscs(con, msg_uuid, tag_uuids)

    con.commit()

    print("done!")

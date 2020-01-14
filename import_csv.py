#!/usr/bin/env python3
import csv
import sqlite3
import sys

from pathlib import Path


DB_PATH = Path.home() / Path('.lgd/logs.db')
CREATED_AT = 'created_at'
MSG = 'msg'
TAGS = 'tags'


def get_connection():
    # This creates the sqlite db if it doesn't exist.
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_messages_tags(con):
    sql = """
    SELECT
        logs.created_at,
        logs.msg,
        group_concat(tags.tag) as tags
    FROM logs
    INNER JOIN logs_tags lt on lt.log = logs.id
    INNER JOIN tags tags on tags.id = lt.tag
    GROUP BY logs.id
    ORDER BY logs.created_at;
    """
    return con.execute(sql)


def row_to_dict(row):
    return {key:row[key] for key in row.keys()}


def dict_to_tuple(row_dict):
    return (row_dict[CREATED_AT], row_dict[MSGS])


INSERT_LOG = """
INSERT into logs (created_at, msg) VALUES (?, ?);
"""
def insert_msg(conn, msg, created_at):
    c = conn.execute(INSERT_LOG, (created_at, msg))
    conn.commit()
    return c.lastrowid


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

            msg_id = insert_msg(con, msg, created_at)
            tag_ids = insert_tags(con, tags)
            insert_asscs(con, msg_id, tag_ids)

    con.commit()

    print("done!")

#!/usr/bin/env python3
import csv
import sqlite3
import sys

from pathlib import Path


DB_PATH = Path.home() / Path('.lgd/logs.db')


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
    INNER JOIN logs_tags lt on lt.log_uuid = logs.uuid
    INNER JOIN tags tags on tags.uuid = lt.tag_uuid
    GROUP BY logs.uuid
    ORDER BY logs.created_at;
    """
    return con.execute(sql)


def row_to_dict(row):
    return {key:row[key] for key in row.keys()}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Must give an output file name...")
        sys.exit()

    filename = sys.argv[1]

    if not DB_PATH.exists():
        print(f"Could not find '{DB_PATH}'")
        sys.exit()

    con = get_connection()
    rows = get_messages_tags(con)
    field_names = [desc[0] for desc in rows.description]

    with open(filename, 'w') as out:
        writer = csv.DictWriter(out, field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row))

    print("done!")

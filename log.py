import argparse, sys, sqlite3, tempfile, os

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
SELECT *
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

def show_msgs(conn, tags):
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

    lines = []
    for row in result:
        msg = row[2]
        lines.extend(msg.splitlines(keepends=True))

    return open_temp_logfile(lines=lines) 


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

        # do the parsing with `tf` using regular File operations.
        # for instance:
        tf.seek(0)
        return (l.decode('utf8') for l in tf.readlines())


if __name__ == '__main__':
    args = parser.parse_args()
    print(f"tags: {args.tags}")

    dir_setup()
    conn = get_connection()
    db_setup(conn)
    print('foo')

    # Display messages
    if args.show:
        show_msgs(conn, args.tags)
        sys.exit()

    # Store message
    """
    with tempfile.NamedTemporaryFile(suffix=".tmp") as tf:
        #tf.write(initial_message.encode('utf8'))
        #tf.flush()
        call([EDITOR, tf.name])

        # do the parsing with `tf` using regular File operations.
        # for instance:
        tf.seek(0)
        edited_message = tf.read()
    """
    msg_lines = open_temp_logfile()
    msg = '\n'.join(msg_lines)

    msg_id = insert_msg(conn, msg.decode('utf8'))

    tags = input("Add tags? (comma separated): ")
    if tags:
        tags = {t.strip() for t in tags.split(',')}
        tag_ids = insert_tags(conn, tags)
        insert_asscs(conn, msg_id, tag_ids)

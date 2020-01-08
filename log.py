import argparse, sys, sqlite3, tempfile, os

from difflib import ndiff
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
        self._lines = None
        self._line_map = {}

        self._render()

    def _render(self):
        self._lines = []
        first = True
        linenum_curr = 1
        for msg_id, _, msg in self.logs:
            if first:
                first = False
            else:
                self._lines.extend(('\n', '---\n', '\n'))
                linenum_curr += 3

            msg_lines = msg.splitlines(keepends=True)
            self._lines.extend(msg_lines)

            # TODO: this is an inefficient way to achieve this mapping
            for i in range(len(msg_lines)):
                linenum_curr += i
                self._line_map[linenum_curr] = msg_id

    @property
    def rendered(self):
        return self._lines

    def diff_rendered(self, other):
        """
        return an iterable of LogDiffs
        """
        i = 0  # represents the line num from the original
        for line in ndiff(self._lines, list(other)):
            if not line.startswith('+ '):
                i += 1
            print(f"{i:3>}: {line}", end='')

            # TODO: Turn the rendered line num into an absolute line num for
            #       each log message.
            pass


class LogDiff:

    def __init__(self, msg_id, msg, mods):
        """
        mods: iterable of (change, line_num, text)
        """
        self.id = msg_id
        self.msg = msg

        # `mods` should be an iterable of:
        # (line_num, change, text) tuples
        self.mods = mods

    def save(self, conn, commit=True):
        if not self._update_msg(conn):
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
        return (l.decode('utf8') for l in tf.readlines())


if __name__ == '__main__':
    args = parser.parse_args()

    dir_setup()
    conn = get_connection()
    db_setup(conn)

    # Display messages
    if args.show:
        #show_msgs(conn, args.tags)
        messages = messages_with_tags(conn, args.tags)
        message_view = RenderedLog(messages)
        edited = open_temp_logfile(message_view.rendered)
        message_view.diff_rendered(edited)
        sys.exit()

    # Store message
    msg_lines = open_temp_logfile()
    msg = '\n'.join(msg_lines)

    msg_id = insert_msg(conn, msg)

    tags = input("Add tags? (comma separated): ")
    if tags:
        tags = {t.strip() for t in tags.split(',')}
        tag_ids = insert_tags(conn, tags)
        insert_asscs(conn, msg_id, tag_ids)

#!/usr/bin/env python3

import csv
import datetime
import gzip
import io
import random
import sqlite3
import string
import unittest
import uuid

import lgd


DB_IN_MEM = ":memory:"


def tag_factory(num):
    return [
        ''.join(random.choices(string.ascii_letters, k=random.randrange(3, 10)))
        for _ in range(num)
    ]


def tag_relation_factory(tags, num):
    return set(tuple(random.sample(tags, 2)) for _ in range(num))


def body_factory():
    return ''.join(random.choices(string.printable, k=random.randrange(20, 100)))


def date_factory(multiple=1):
    now = datetime.datetime.now()
    delta = datetime.timedelta(seconds=random.randrange(0, 3600))
    return now - (multiple * delta)


def note_factory(num_notes, tags, min_tags=0, max_tags=4):
    def choose_tags():
        return frozenset(random.choices(tags, k=random.randrange(min_tags, max_tags)))

    notes = []
    for i in range(num_notes):
        note = lgd.Note(
            uuid=uuid.uuid4(),
            created_at=date_factory(multiple=i),
            body=body_factory(),
            tags=choose_tags(),
        )
        notes.append(note)
    return sorted(notes, key=lambda n: n.created_at)


def note_csv_factory(notes):
    outfile = io.StringIO()
    writer = csv.DictWriter(outfile, lgd.Note._fields)
    writer.writeheader()
    for note in notes:
        # For the CSV file, for the tags to be a comma separated str.
        note = note._replace(tags=','.join(note.tags))
        writer.writerow(note._asdict())
    outfile.seek(0)
    return outfile


def tag_csv_factory(tag_relations):
    outfile = io.StringIO()
    writer = csv.writer(outfile)
    writer.writerow(('tag_direct', 'tag_indirect'))
    writer.writerows(tag_relations)
    outfile.seek(0)
    return outfile


def get_note_csv():
    _NOTE_CSV.seek(0)
    return _NOTE_CSV


def get_tag_csv():
    _TAG_CSV.seek(0)
    return _TAG_CSV


TAGS = tag_factory(50)
NOTES = note_factory(100, TAGS)
TAG_RELATIONS = tag_relation_factory(TAGS, 20)
_NOTE_CSV = note_csv_factory(NOTES)
_TAG_CSV = tag_csv_factory(TAG_RELATIONS)


class TestFlattenTagGroups(unittest.TestCase):

    def test_no_groups(self):
        self.assertEqual([], lgd.flatten_tag_groups([]))

    def test_empty_groups(self):
        self.assertEqual([], lgd.flatten_tag_groups([[]]))

    def test_single_group(self):
        self.assertEqual(
            ['a'],
            lgd.flatten_tag_groups([['a']])
        )
        self.assertEqual(
            ['a', 'b', 'c'],
            lgd.flatten_tag_groups([['a', 'b', 'c']])
        )

    def test_multiple_groups(self):
        self.assertEqual(
            ['a', 'b', 'c', 'd'],
            lgd.flatten_tag_groups([['a', 'b'], ['c', 'd']])
        )


class TestGzip(unittest.TestCase):

    def setUp(self):
        self.text = "The quick brown fox jumps over the lazy dog"
        self.data = gzip.compress(self.text.encode('utf8'))

    def test_compress_string(self):
        compressed = lgd.Gzip.compress_string(self.text)

        self.assertIsInstance(compressed, bytes)
        self.assertEqual(
            self.text,
            gzip.decompress(compressed).decode('utf8')
        )

    def test_decompress_string(self):
        decompressed = lgd.Gzip.decompress_string(self.data)

        self.assertIsInstance(decompressed, str)
        self.assertEqual(
            self.text,
            decompressed
        )

    def test_col_type(self):
        self.assertTrue(hasattr(lgd.Gzip, 'COL_TYPE'))
        self.assertEqual(lgd.Gzip.COL_TYPE, 'GZIP')


class TestDBSetup(unittest.TestCase):

    def setUp(self):
        self.conn = lgd.get_connection(DB_IN_MEM)

    def test_migrations(self):
        # App version before migrations
        current_version = lgd.get_user_version(self.conn)
        self.assertNotEqual(current_version, lgd.DB_USER_VERSION)

        success = lgd.db_setup(self.conn, lgd.DB_MIGRATIONS)
        self.assertTrue(success, "DB migrations failed!")

        # After migrations
        current_version = lgd.get_user_version(self.conn)
        self.assertEqual(current_version, lgd.DB_USER_VERSION)

    def test_migration_fail(self):
        # Test that a sqlite3 exception will be caught.

        def raise_err(err):
            raise err

        migrations = [
            (1, lambda c: raise_err(sqlite3.Error))
        ]

        success = lgd.db_setup(self.conn, migrations)
        self.assertFalse(success, "db_setup failed to catch migration failure")

    def test_migration_intermediate(self):
        BAD_TABLE = 'fail'

        def bad_migration(conn):
            conn.execute(f"CREATE TABLE {BAD_TABLE} (id INT PRIMARY KEY, data BLOB);")
            conn.execute(f"INSERT INTO {BAD_TABLE} (id, data) VALUES (1, 'abcd');")
            cursor = conn.execute("SELECT * FROM fail WHERE id = ?", (1,))
            self.assertEqual(cursor.fetchone()[1], 'abcd')
            raise sqlite3.Error

        def good_migration(conn):
            conn.execute("CREATE table success (id INT PRIMARY KEY);")

        migrations = [
            (1, good_migration),
            (2, bad_migration),
        ]

        # Expect failure to finish all migrations

        success = lgd.db_setup(self.conn, migrations)
        self.assertFalse(success)

        # Ensure good_migration succeeded.

        db_version = lgd.get_user_version(self.conn)
        self.assertEqual(db_version, 1)

        # Ensure bad_migration failed and did not fail in an intermediate state

        cursor = self.conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' and name = ?",
            (BAD_TABLE,)
        )
        count = cursor.fetchone()[0]
        table_exists = bool(count)

        self.assertFalse(table_exists, f"count == {count}; table exists!")

    def test_db_version(self):
        version = lgd.get_user_version(self.conn)
        self.assertEqual(version, 0)

        lgd.set_user_version(self.conn, 1)
        version = lgd.get_user_version(self.conn)
        self.assertEqual(version, 1)

        lgd.set_user_version(self.conn, 3)
        version = lgd.get_user_version(self.conn)
        self.assertEqual(version, 3)


class TestDBNotes(unittest.TestCase):

    def setUp(self):
        self.conn = lgd.get_connection(DB_IN_MEM)
        lgd.db_setup(self.conn, lgd.DB_MIGRATIONS)

    def test_note_insert(self):
        pass


class TestCSVNoteInsert(unittest.TestCase):

    def setUp(self):
        self.conn = lgd.get_connection(DB_IN_MEM)
        lgd.db_setup(self.conn, lgd.DB_MIGRATIONS)

    def test_note_import_new(self):
        inserted, updated = lgd.note_import(self.conn, get_note_csv())
        self.assertEqual(inserted, len(NOTES))
        self.assertEqual(updated, 0)

    def test_note_import_equality(self):
        pass


class TestCSVNoteUpdate(unittest.TestCase):

    def setUp(self):
        self.conn = lgd.get_connection(DB_IN_MEM)
        lgd.db_setup(self.conn, lgd.DB_MIGRATIONS)
        self.inserted, self.updated = lgd.note_import(self.conn, get_note_csv())

    def test_note_import_update(self):
        inserted, updated = lgd.note_import(self.conn, get_note_csv())
        self.assertEqual(inserted + updated, len(NOTES))
        self.assertEqual(inserted, 0)
        self.assertEqual(updated, self.inserted)
        self.assertEqual(updated, len(NOTES), f'Number updated: {updated}')


class TestCSVNoteExport(unittest.TestCase):

    def setUp(self):
        self.conn = lgd.get_connection(DB_IN_MEM)
        lgd.db_setup(self.conn, lgd.DB_MIGRATIONS)
        self.inserted, self.updated = lgd.note_import(self.conn, get_note_csv())

    def test_note_export(self):
        # Check number of exported notes
        outfile = io.StringIO()
        num_exported = lgd.note_export(self.conn, outfile)

        self.assertEqual(num_exported, len(NOTES))

    def test_note_export_equality(self):
        # Retrieve Notes inserted during setup.
        notes1 = set(lgd.messages_with_tags(self.conn, None))
        self.assertEqual(len(notes1), len(NOTES))

        # Export from the setUp DB.
        notefile = io.StringIO()
        _ = lgd.note_export(self.conn, notefile)
        notefile.seek(0)

        # Set up a second DB
        conn2 = lgd.get_connection(DB_IN_MEM)
        lgd.db_setup(conn2, lgd.DB_MIGRATIONS)

        # Import the Notes from the 1st DB into the 2nd.
        _, _ = lgd.note_import(conn2, notefile)

        # Retrieve the Notes from the 2nd DB.
        # Check that the notes retrieved from both DBs are equal.
        notes2 = set(lgd.messages_with_tags(conn2, None))
        self.assertEqual(notes1, notes2)


class TestCSVTagImportExport(unittest.TestCase):

    def setUp(self):
        self.conn = lgd.get_connection(DB_IN_MEM)
        lgd.db_setup(self.conn, lgd.DB_MIGRATIONS)

    def test_tag_relation_import_new(self):
        inserted, existing = lgd.tag_import(self.conn, get_tag_csv())
        self.assertEqual(inserted + existing, len(TAG_RELATIONS))
        self.assertEqual(inserted, len(TAG_RELATIONS))
        self.assertEqual(existing, 0)

    def test_tag_relation_import_existing(self):
        # Initial import of tag relations
        _, _ = lgd.tag_import(self.conn, get_tag_csv())

        inserted, existing = lgd.tag_import(self.conn, get_tag_csv())
        self.assertEqual(inserted + existing, len(TAG_RELATIONS))
        self.assertEqual(inserted, 0)
        self.assertEqual(existing, len(TAG_RELATIONS))

    def test_tag_relation_export(self):
        # Insert and then retrieve tag-relations.
        _, _ = lgd.tag_import(self.conn, get_tag_csv())
        tag_relations1 = set(lgd.select_related_tags_all(self.conn))
        self.assertEqual(len(tag_relations1), len(TAG_RELATIONS))

        # Export tag-relations from the initial DB.
        tagfile = io.StringIO()
        _ = lgd.tag_export(self.conn, tagfile)
        tagfile.seek(0)

        # Set up a second DB.
        # Import tag-relations from the 1st DB into the 2nd.
        conn2 = lgd.get_connection(DB_IN_MEM)
        lgd.db_setup(conn2, lgd.DB_MIGRATIONS)
        _, _ = lgd.tag_import(conn2, tagfile)

        # Retrieve the tag-relations from the 2nd DB.
        # Check that the tag-relations retrieved from both DBs are equal.
        tag_relations2 = set(lgd.select_related_tags_all(conn2))
        self.assertEqual(tag_relations1, tag_relations2)


if __name__ == '__main__':
    unittest.main()

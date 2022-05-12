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

from typing import List, Set, Tuple

from context import lgd
from lgd import data
from lgd import exceptions


DB_IN_MEM = ":memory:"


def tag_factory(num: int) -> List[str]:
    return [
        "".join(random.choices(string.ascii_letters, k=random.randrange(3, 10)))
        for _ in range(num)
    ]


def tag_relation_factory(tags: str, num: int) -> Set[Tuple]:
    return set(tuple(random.sample(tags, 2)) for _ in range(num))


def body_factory() -> str:
    return "".join(random.choices(string.printable, k=random.randrange(20, 100)))


def date_factory(multiple: int = 1) -> datetime.datetime:
    now = datetime.datetime.now().replace(microsecond=0)
    delta = datetime.timedelta(seconds=random.randrange(0, 3600))
    return now - (multiple * delta)


def note_factory(
    num_notes: int, tags: List[str], min_tags: int = 0, max_tags: int = 4
) -> List[data.Note]:
    def choose_tags():
        return frozenset(random.choices(tags, k=random.randrange(min_tags, max_tags)))

    notes: List[data.Note] = []
    for i in range(num_notes):
        note = data.Note(
            uuid=uuid.uuid4(),
            created_at=date_factory(multiple=i),
            body=body_factory(),
            tags=choose_tags(),
        )
        notes.append(note)
    return sorted(notes, key=lambda n: n.created_at)


def note_csv_factory(notes: List[data.Note]) -> io.StringIO:
    outfile = io.StringIO()
    writer = csv.DictWriter(outfile, data.Note._fields)
    writer.writeheader()
    for note in notes:
        # For the CSV file, for the tags to be a comma separated str.
        note = note._replace(tags=",".join(note.tags))
        writer.writerow(note._asdict())
    outfile.seek(0)
    return outfile


def tag_csv_factory(tag_relations) -> io.StringIO:
    outfile = io.StringIO()
    writer = csv.writer(outfile)
    writer.writerow(("tag_direct", "tag_indirect"))
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
        self.assertEqual(["a"], lgd.flatten_tag_groups([["a"]]))
        self.assertEqual(["a", "b", "c"], lgd.flatten_tag_groups([["a", "b", "c"]]))

    def test_multiple_groups(self):
        self.assertEqual(
            ["a", "b", "c", "d"], lgd.flatten_tag_groups([["a", "b"], ["c", "d"]])
        )


class TestGzip(unittest.TestCase):
    def setUp(self):
        self.text = "The quick brown fox jumps over the lazy dog"
        self.data = gzip.compress(self.text.encode("utf8"))

    def test_compress_string(self):
        compressed = data.Gzip.compress_string(self.text)

        self.assertIsInstance(compressed, bytes)
        self.assertEqual(self.text, gzip.decompress(compressed).decode("utf8"))

    def test_decompress_string(self):
        decompressed = data.Gzip.decompress_string(self.data)

        self.assertIsInstance(decompressed, str)
        self.assertEqual(self.text, decompressed)

    def test_col_type(self):
        self.assertTrue(hasattr(data.Gzip, "COL_TYPE"))
        self.assertEqual(data.Gzip.COL_TYPE, "GZIP")


class TestDBSetup(unittest.TestCase):
    def setUp(self):
        self.conn = data.get_connection(DB_IN_MEM)

    def test_migrations(self):
        # App version before migrations
        current_version = data.get_user_version(self.conn)
        self.assertNotEqual(current_version, data.DB_USER_VERSION)

        success = data.db_setup(self.conn, data.DB_MIGRATIONS)
        self.assertTrue(success, "DB migrations failed!")

        # After migrations
        current_version = data.get_user_version(self.conn)
        self.assertEqual(current_version, data.DB_USER_VERSION)

    def test_migration_fail(self):
        # Test that a sqlite3 exception will be caught.

        def raise_err(err):
            raise err

        migrations = [(1, lambda c: raise_err(sqlite3.Error))]

        success = data.db_setup(self.conn, migrations)
        self.assertFalse(success, "db_setup failed to catch migration failure")

    def test_migration_intermediate(self):
        BAD_TABLE = "fail"

        def good_migration(conn):
            conn.execute("CREATE table success (id INT PRIMARY KEY);")

        def bad_migration(conn):
            conn.execute(f"CREATE TABLE {BAD_TABLE} (id INT PRIMARY KEY, data BLOB);")
            conn.execute(f"INSERT INTO {BAD_TABLE} (id, data) VALUES (1, 'abcd');")
            cursor = conn.execute("SELECT * FROM fail WHERE id = ?", (1,))
            self.assertEqual(cursor.fetchone()[1], "abcd")
            raise sqlite3.Error

        migrations = [
            (1, good_migration),
            (2, bad_migration),
        ]

        # Expect failure to finish all migrations

        success = data.db_setup(self.conn, migrations)
        self.assertFalse(success)

        # Ensure good_migration succeeded.

        db_version = data.get_user_version(self.conn)
        self.assertEqual(db_version, 1)

        # Ensure bad_migration failed and did not fail in an intermediate state

        cursor = self.conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' and name = ?",
            (BAD_TABLE,),
        )
        count = cursor.fetchone()[0]
        table_exists = bool(count)

        self.assertFalse(table_exists, f"count == {count}; table exists!")

    def test_db_version(self):
        version = data.get_user_version(self.conn)
        self.assertEqual(version, 0)

        data.set_user_version(self.conn, 1)
        version = data.get_user_version(self.conn)
        self.assertEqual(version, 1)

        data.set_user_version(self.conn, 3)
        version = data.get_user_version(self.conn)
        self.assertEqual(version, 3)


class TestNoteInsert(unittest.TestCase):
    def setUp(self):
        self.conn = data.get_connection(DB_IN_MEM)
        data.db_setup(self.conn, data.DB_MIGRATIONS)

    def test_note_insert(self):
        note_body = body_factory()

        # Insert a note
        note_uuid = data.insert_note(self.conn, note_body)

        # Show that the note may be retrieved
        notes = data.Notes(uuids=[note_uuid]).fetch(self.conn)

        self.assertEqual(len(notes), 1)
        note = notes[0]

        self.assertEqual(note.body, note_body)

    def test_note_insert_with_uuid(self):
        note_uuid_orig = uuid.uuid4()
        note_body = body_factory()

        # Insert a note, passing a pre-existing uuid.
        note_uuid = data.insert_note(self.conn, note_body, note_uuid=note_uuid_orig)
        self.assertEqual(note_uuid, note_uuid_orig)

        # Retrieve the note and show that the uuid is the same.
        notes = data.Notes(uuids=[note_uuid]).fetch(self.conn)

        self.assertEqual(len(notes), 1)
        note = notes[0]

        self.assertEqual(note.body, note_body)
        self.assertEqual(note.uuid, note_uuid_orig)

    def test_note_insert_with_timestamp(self):
        created_at_orig = date_factory()
        note_body = body_factory()

        # Insert a note, passing a pre-existing timestamp.
        note_uuid = data.insert_note(self.conn, note_body, created_at=created_at_orig)

        # Retrieve the note and show that the created_at timestamp is the same.
        notes = data.Notes(uuids=[note_uuid], localtime=False).fetch(self.conn)

        self.assertEqual(len(notes), 1)
        note = notes[0]

        self.assertEqual(note.body, note_body)
        self.assertEqual(note.uuid, note_uuid)
        self.assertEqual(note.created_at, created_at_orig)


class TestNoteSelect(unittest.TestCase):
    def setUp(self):
        self.conn = data.get_connection(DB_IN_MEM)
        data.db_setup(self.conn, data.DB_MIGRATIONS)
        # TODO: insert some notes

    def test_select_uuids(self):
        note_uuids = [data.insert_note(self.conn, body_factory()) for _ in range(5)]

        # Select zero uuids
        notes = data.Notes(uuids=[]).fetch(self.conn)
        self.assertEqual(notes, [])

        # Select one uuid
        notes = data.Notes(uuids=[note_uuids[0]]).fetch(self.conn)
        self.assertEqual(len(notes), 1)
        self.assertEqual([n.uuid for n in notes], [note_uuids[0]])

        # Select multiple uuids
        notes = data.Notes(uuids=note_uuids).fetch(self.conn)
        self.assertEqual(len(notes), len(note_uuids))
        self.assertEqual({n.uuid for n in notes}, set(note_uuids))

    def test_select_tag_groups(self):
        """
        msg_uuid = lgd.insert_note(self.conn, new_note.body)
        tag_uuids = lgd.insert_tags(self.conn, new_note.tags)
        lgd.insert_asscs(self.conn, msg_uuid, tag_uuids)
        """

    def test_select_date_ranges(self):
        pass

    def test_select_text(self):
        pass

    def test_select_complex(self):
        pass


class TestCSVNoteInsert(unittest.TestCase):
    def setUp(self):
        self.conn = data.get_connection(DB_IN_MEM)
        data.db_setup(self.conn, data.DB_MIGRATIONS)

    def test_note_import_new(self):
        inserted, updated = lgd.note_import(self.conn, get_note_csv())
        self.assertEqual(inserted, len(NOTES))
        self.assertEqual(updated, 0)

    def test_invalid_fieldnames(self):
        infile = io.StringIO("a,b,c,d\n" "1,2,3,4\n")
        with self.assertRaises(exceptions.CSVError):
            _, _ = lgd.note_import(self.conn, infile)

    def test_invalid_uuid(self):
        notes = note_factory(1, TAGS)

        uuids = [1, "2", "asdf", hex(random.randint(0, 1_000_000))]

        for uuid_bad in uuids:
            with self.subTest(uuid_bad=uuid_bad):
                notes[0] = notes[0]._replace(uuid=uuid_bad)
                infile = note_csv_factory(notes)
                with self.assertRaises(exceptions.CSVError):
                    _, _ = lgd.note_import(self.conn, infile)

    def test_invalid_created_at(self):
        notes = note_factory(1, TAGS)

        dates = [
            1,
            "1",
            "asdf",
            "2020-04-01",
            "2020-04-01 12",
            "2020-04-01 12:00",
            "2020-04-01 12:00:00.123456",
        ]

        for date_bad in dates:
            with self.subTest(date_bad=date_bad):
                notes[0] = notes[0]._replace(created_at=date_bad)
                infile = note_csv_factory(notes)
                with self.assertRaises(exceptions.CSVError):
                    _, _ = lgd.note_import(self.conn, infile)


class TestCSVNoteUpdate(unittest.TestCase):
    def setUp(self):
        self.conn = data.get_connection(DB_IN_MEM)
        data.db_setup(self.conn, data.DB_MIGRATIONS)
        self.inserted, self.updated = lgd.note_import(self.conn, get_note_csv())

    def test_note_import_update(self):
        inserted, updated = lgd.note_import(self.conn, get_note_csv())
        self.assertEqual(inserted + updated, len(NOTES))
        self.assertEqual(inserted, 0)
        self.assertEqual(updated, self.inserted)
        self.assertEqual(updated, len(NOTES), f"Number updated: {updated}")


class TestCSVNoteExport(unittest.TestCase):
    def setUp(self):
        self.conn = data.get_connection(DB_IN_MEM)
        data.db_setup(self.conn, data.DB_MIGRATIONS)
        self.inserted, self.updated = lgd.note_import(self.conn, get_note_csv())

    def test_note_export(self):
        # Check number of exported notes
        notes = data.Notes(localtime=False).fetch(self.conn)
        outfile = io.StringIO()
        num_exported = lgd.note_export(self.conn, notes, outfile)

        self.assertEqual(num_exported, len(NOTES))

    def test_note_export_equality(self):
        # Retrieve Notes inserted during setup.
        notes1 = set(data.Notes().fetch(self.conn))
        self.assertEqual(len(notes1), len(NOTES))

        # Export from the setUp DB.
        notes = data.Notes(localtime=False).fetch(self.conn)
        notefile = io.StringIO()
        _ = lgd.note_export(self.conn, notes, notefile)
        notefile.seek(0)

        # Set up a second DB
        conn2 = data.get_connection(DB_IN_MEM)
        data.db_setup(conn2, data.DB_MIGRATIONS)

        # Import the Notes from the 1st DB into the 2nd.
        _, _ = lgd.note_import(conn2, notefile)

        # Retrieve the Notes from the 2nd DB.
        # Check that the notes retrieved from both DBs are equal.
        notes2 = set(data.Notes().fetch(conn2))
        self.assertEqual(notes1, notes2)


class TestCSVTagImportExport(unittest.TestCase):
    def setUp(self):
        self.conn = data.get_connection(DB_IN_MEM)
        data.db_setup(self.conn, data.DB_MIGRATIONS)

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
        tag_relations1 = set(data.select_related_tags_all(self.conn))
        self.assertEqual(len(tag_relations1), len(TAG_RELATIONS))

        # Export tag-relations from the initial DB.
        tagfile = io.StringIO()
        _ = lgd.tag_export(self.conn, tagfile)
        tagfile.seek(0)

        # Set up a second DB.
        # Import tag-relations from the 1st DB into the 2nd.
        conn2 = data.get_connection(DB_IN_MEM)
        data.db_setup(conn2, data.DB_MIGRATIONS)
        _, _ = lgd.tag_import(conn2, tagfile)

        # Retrieve the tag-relations from the 2nd DB.
        # Check that the tag-relations retrieved from both DBs are equal.
        tag_relations2 = set(data.select_related_tags_all(conn2))
        self.assertEqual(tag_relations1, tag_relations2)


if __name__ == "__main__":
    unittest.main(buffer=True)

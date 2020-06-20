#!/usr/bin/env python3

import gzip
import sqlite3
import unittest

import lgd


DB_IN_MEM = ":memory:"


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


if __name__ == '__main__':
    unittest.main()

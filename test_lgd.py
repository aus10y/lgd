import gzip
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

    def test_db_version(self):
        # App version before migrations
        current_version = lgd.get_user_version(self.conn)
        self.assertNotEqual(current_version, lgd.DB_USER_VERSION)

        lgd.db_setup(self.conn)

        # After migrations
        current_version = lgd.get_user_version(self.conn)
        self.assertEqual(current_version, lgd.DB_USER_VERSION)


class TestDBNotes(unittest.TestCase):

    def setUp(self):
        self.conn = lgd.get_connection(DB_IN_MEM)

    def test_note_insert(self):
        pass


if __name__ == '__main__':
    unittest.main()

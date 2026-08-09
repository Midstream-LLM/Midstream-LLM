import struct
import unittest


class AneIndexFormatTest(unittest.TestCase):
    def test_frozen_header_and_record_sizes(self):
        header = struct.Struct("<8sIIIIQ6dII")
        record = struct.Struct("<IQIB3x")
        self.assertEqual(header.size, 88)
        self.assertEqual(record.size, 20)
        packed = record.pack(3, 123456789, 4096, 6)
        self.assertEqual(record.unpack(packed), (3, 123456789, 4096, 6))


if __name__ == "__main__":
    unittest.main()

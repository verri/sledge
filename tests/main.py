import unittest
import pandas as pd
import numpy as np

from sledge import sledge_score_clusters


class TestSimpleCalculations(unittest.TestCase):

    def test_basic(self):
        X = pd.DataFrame.from_dict({
            'A': [1, 0, 1, 0, 0, 0, 0, 0],
            'B': [1, 0, 0, 0, 0, 0, 0, 1],
            'C': [1, 1, 1, 1, 1, 1, 0, 0],
            'D': [0, 0, 0, 1, 0, 1, 1, 0],
            'E': [0, 0, 0, 1, 1, 0, 1, 0]})

        labels = [0, 0, 0, 1, 1, 1, 1, 2]

        score = sledge_score_clusters(X, labels)
        self.assertEqual(3, len(score))


if __name__ == '__main__':
    unittest.main()

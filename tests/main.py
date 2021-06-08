import unittest
import pandas as pd
import numpy as np

from sledge import sledge_score_clusters, semantic_descriptors


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
        print(score)
        # TODO: calculate SLEDge "by hand" and compare values here

    def test_zero_descriptors(self):
        X = pd.DataFrame.from_dict({
            'A': [1, 1, 1, 1, 1, 0],
            'B': [1, 1, 0, 1, 0, 0],
            'C': [0, 1, 1, 1, 0, 1],
            'D': [0, 0, 0, 0, 0, 0],
            'E': [0, 0, 0, 0, 0, 0]})

        labels = [0, 0, 0, 0, 1, 1]

        descriptors = semantic_descriptors(X, labels)

        self.assertLess(0, np.max(descriptors.transpose()[0]))
        self.assertEqual(0, np.max(descriptors.transpose()[1]))

        score = sledge_score_clusters(X, labels)
        self.assertEqual(0, score[1])
        print(score)
        # TODO: calculate SLEDge "by hand" and compare values here

    def test_sledge_full_matrix(self):
        X = pd.DataFrame.from_dict({
            'A': [1, 1, 1, 1, 1, 0],
            'B': [1, 1, 0, 1, 0, 0],
            'C': [0, 1, 1, 1, 0, 1],
            'D': [0, 0, 0, 0, 0, 0],
            'E': [0, 0, 0, 0, 0, 0]})

        labels = [0, 0, 0, 0, 1, 1]

        score_matrix = sledge_score_clusters(X, labels, aggregation=None)
        print(score_matrix)
        values = score_matrix.to_dict('records')

        self.assertLess(0, values[0]['S'])
        self.assertEqual(1, values[0]['L'])
        self.assertEqual(1, values[0]['E'])
        self.assertLess(0, values[0]['D'])

        self.assertEqual(0, values[1]['S'])
        self.assertEqual(0, values[1]['L'])
        self.assertEqual(0, values[1]['E'])
        self.assertEqual(0, values[1]['D'])


if __name__ == '__main__':
    unittest.main()

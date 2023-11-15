import torch
import unittest

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder(['h', 'e', 'l', 'p'])
        probs = torch.tensor([[0.1, 0.4, 0.3, 0.1, 0.1],
                             [0.1, 0.1, 0.6, 0.1, 0.1],
                             [0.1, 0.1, 0.1, 0.6, 0.1],
                             [0.1, 0.1, 0.1, 0.1, 0.6],
                             [0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.1, 0.1, 0.1, 0.1, 0.1]])

        result = text_encoder.ctc_beam_search(probs, 4, 2)
        text_result = (result[0].text, result[1].text)
        expected_result = ("help", "elp")
        self.assertEqual(text_result, expected_result)




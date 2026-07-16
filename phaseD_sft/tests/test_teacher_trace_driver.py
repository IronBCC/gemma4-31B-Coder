import unittest

from phaseD_sft.teacher_trace_driver import (
    CREDIT_EXHAUSTED_ERROR,
    credit_exhausted_in_stream,
    next_credit_streak,
)


class TeacherTraceCreditGuardTests(unittest.TestCase):
    def test_detects_credit_wall_even_when_cli_marks_result_success(self):
        stream = (
            '{"type":"result","subtype":"success",'
            '"result":"Error: out of usage credits. Try again later."}\n'
        )

        self.assertTrue(credit_exhausted_in_stream(stream))

    def test_credit_streak_resets_after_a_non_credit_result(self):
        self.assertEqual(next_credit_streak(2, CREDIT_EXHAUSTED_ERROR), 3)
        self.assertEqual(next_credit_streak(3, None), 0)


if __name__ == "__main__":
    unittest.main()

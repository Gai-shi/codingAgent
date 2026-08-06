from __future__ import annotations

import unittest
from unittest.mock import patch

from ai_job.terminal_input import AllowInputEcho, SuppressInputEchoAndDiscard


class FakeStdin:
    def fileno(self):
        return 7


class FakeTermiosModule:
    ECHO = 0b1000
    TCSADRAIN = 1
    TCIFLUSH = 2
    error = OSError

    def __init__(self, initial_attrs):
        self.initial_attrs = list(initial_attrs)
        self.set_attrs = []
        self.flush_calls = []
        self.getattr_calls = []

    def tcgetattr(self, stdin_fd):
        self.getattr_calls.append(stdin_fd)
        return list(self.initial_attrs)

    def tcsetattr(self, stdin_fd, when, attrs):
        self.set_attrs.append((stdin_fd, when, list(attrs)))

    def tcflush(self, stdin_fd, queue):
        self.flush_calls.append((stdin_fd, queue))


class TerminalInputModeTest(unittest.TestCase):
    def test_allow_input_echo_sets_echo_and_restores_original_attrs(self):
        fake_termios = FakeTermiosModule([0, 0, 0, 0])

        with patch.dict("sys.modules", {"termios": fake_termios}):
            with patch("ai_job.terminal_input.terminal_input_mode.os.isatty", return_value=True):
                with AllowInputEcho(stdin=FakeStdin()):
                    pass

        self.assertEqual(fake_termios.getattr_calls, [7])
        self.assertEqual(fake_termios.set_attrs[0], (7, fake_termios.TCSADRAIN, [0, 0, 0, fake_termios.ECHO]))
        self.assertEqual(fake_termios.set_attrs[1], (7, fake_termios.TCSADRAIN, [0, 0, 0, 0]))
        self.assertEqual(fake_termios.flush_calls, [])

    def test_suppress_input_echo_clears_echo_flushes_input_and_restores_attrs(self):
        fake_termios = FakeTermiosModule([0, 0, 0, FakeTermiosModule.ECHO])

        with patch.dict("sys.modules", {"termios": fake_termios}):
            with patch("ai_job.terminal_input.terminal_input_mode.os.isatty", return_value=True):
                with SuppressInputEchoAndDiscard(stdin=FakeStdin()):
                    pass

        self.assertEqual(fake_termios.set_attrs[0], (7, fake_termios.TCSADRAIN, [0, 0, 0, 0]))
        self.assertEqual(fake_termios.flush_calls, [(7, fake_termios.TCIFLUSH)])
        self.assertEqual(
            fake_termios.set_attrs[1],
            (7, fake_termios.TCSADRAIN, [0, 0, 0, FakeTermiosModule.ECHO]),
        )

    def test_non_tty_stdin_degrades_to_noop(self):
        fake_termios = FakeTermiosModule([0, 0, 0, FakeTermiosModule.ECHO])

        with patch.dict("sys.modules", {"termios": fake_termios}):
            with patch("ai_job.terminal_input.terminal_input_mode.os.isatty", return_value=False):
                with SuppressInputEchoAndDiscard(stdin=FakeStdin()):
                    pass

        self.assertEqual(fake_termios.getattr_calls, [])
        self.assertEqual(fake_termios.set_attrs, [])
        self.assertEqual(fake_termios.flush_calls, [])

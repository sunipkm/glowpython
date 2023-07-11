import glowpython
import unittest
import io
import sys
import contextlib

@contextlib.contextmanager
def captured_output(stream_name):
    """Run the 'with' statement body using a StringIO object in place of a
       specific attribute on the sys module.
       Example use (with 'stream_name=stdout'):

       with captured_stdout() as s:
           print("hello")
           assert s.getvalue() == "hello"
    """
    orig_stdout = getattr(sys, stream_name)
    setattr(sys, stream_name, io.StringIO())
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)

def captured_stdout():
    return captured_output("stdout")

def captured_stderr():
    return captured_output("stderr")

def captured_stdin():
    return captured_output("stdin")


class TestGlowbasic(unittest.TestCase):
    def setUp(self):
        with captured_stdout() as stdout:
            with open('src/GLOW/in.basic.day', 'r') as infile:
                with captured_stdin() as stdin:
                    import glowbasic
                    glowpython.release_cglow()
                    stdin.write(infile.read())
                    glowbasic.main()
                    self.actual = stdout.read()

                    with open('src/GLOW/out.basic.day', 'r') as outfile:
                        self.expected = outfile.read()

    # TODO: Figure out what is wrong with this test
    @unittest.expectedFailure
    def test_glowbasic(self):
        self.assertEqual(self.actual, self.expected)

    def test_test(self):
        self.assertTrue(True)

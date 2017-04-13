"""Exceptions for ReSpecTh Parser.

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""

class ParseError(Exception):
    """Base class for errors."""
    pass


class KeywordError(ParseError):
    """Raised for errors in keyword parsing."""

    def __init__(self, *keywords):
        self.keywords = keywords

    def __str__(self):
        return repr('Error: {}.'.format(self.keywords))


class UndefinedElementError(KeywordError):
    """Raised for undefined elements."""

    def __str__(self):
        return repr('Error: Element not defined.\n{}'.format(self.keywords))


class MissingElementError(KeywordError):
    """Raised for missing required elements."""

    def __str__(self):
        return repr('Error: Required element {} is missing.'.format(
            self.keywords))


class MissingAttributeError(KeywordError):
    """Raised for missing required attribute."""

    def __str__(self):
        return repr('Error: Required attribute {} of {} is missing.'.format(
            self.keywords))


class UndefinedKeywordError(KeywordError):
    """Raised for undefined keywords."""

    def __str__(self):
        return repr('Error: Keyword not defined: {}'.format(self.keywords))

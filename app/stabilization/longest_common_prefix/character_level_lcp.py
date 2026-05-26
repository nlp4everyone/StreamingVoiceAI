"""
Character-level LCP strategy.

Suitable for Latin-script languages where every byte of a word is
meaningful and a single wrong character invalidates only from that
byte onward.  For languages with tonal diacritics (e.g. Vietnamese)
prefer WordLevelLCP, which avoids over-trimming when a diacritic
shifts the byte boundary mid-word.
"""


class CharacterLevelLCP:
    """LCP strategy that compares transcripts character by character."""

    def lcp(self, s1: str, s2: str) -> str:
        """Return the longest common character prefix of *s1* and *s2*.

        Scans both strings simultaneously and stops at the first
        differing character.

        Args:
            s1: First transcript string.
            s2: Second transcript string.

        Returns:
            The shared leading substring, empty string if none.
        """
        i, min_len = 0, min(len(s1), len(s2))
        while i < min_len and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    def starts_with(self, s: str, prefix: str) -> bool:
        """Return True if *s* starts with *prefix* at the character level.

        Args:
            s: The string to test.
            prefix: The expected leading substring.

        Returns:
            True when *s* begins with every character of *prefix*.
        """
        return s.startswith(prefix)

    def suffix_after(self, s: str, prefix: str) -> str:
        """Return the portion of *s* that follows *prefix*.

        Args:
            s: The full string.
            prefix: A leading substring of *s*.

        Returns:
            Characters of *s* after the end of *prefix*; empty string
            when *prefix* spans the whole of *s*.
        """
        return s[len(prefix):]

"""
Word-level LCP strategy.

Splits transcripts on whitespace before comparing, so a single
misrecognised diacritic only invalidates the prefix from that *word*
onward rather than from the first differing byte.  This makes it the
recommended strategy for Vietnamese and other tonal languages where
character-level diffing would discard too much stable context.
"""


class WordLevelLCP:
    """LCP strategy that compares transcripts word by word.

    Recommended for Vietnamese: a misrecognised diacritic on one word only
    invalidates the prefix from that word onward, not from the first differing
    byte.
    """

    def lcp(self, s1: str, s2: str) -> str:
        """Return the longest common word-level prefix of *s1* and *s2*.

        Tokenises both strings by whitespace, then walks the token
        lists in parallel until the first non-matching word.

        Args:
            s1: First transcript string.
            s2: Second transcript string.

        Returns:
            Space-joined shared leading words; empty string if the
            first words already differ.
        """
        words1, words2 = s1.split(), s2.split()
        i = 0
        while i < min(len(words1), len(words2)) and words1[i] == words2[i]:
            i += 1
        return " ".join(words1[:i])

    def starts_with(self, s: str, prefix: str) -> bool:
        """Return True if *s* starts with *prefix* at the word level.

        An empty *prefix* always matches.  Otherwise checks that the
        leading tokens of *s* are identical to every token in *prefix*.

        Args:
            s: The string to test.
            prefix: The expected leading words (space-separated).

        Returns:
            True when all words of *prefix* appear at the start of *s*.
        """
        if not prefix:
            return True
        s_words, p_words = s.split(), prefix.split()
        return len(s_words) >= len(p_words) and s_words[:len(p_words)] == p_words

    def suffix_after(self, s: str, prefix: str) -> str:
        """Return the portion of *s* that follows *prefix* at the word level.

        Skips the words that constitute *prefix*, then rejoins the
        remainder with a single leading space so the result can be
        directly appended to the prefix string.

        Args:
            s: The full transcript string.
            prefix: A leading word sequence of *s* (space-separated).

        Returns:
            Remaining words prefixed with a space, or empty string when
            no words remain after *prefix*.
        """
        if not prefix:
            return s
        p_words = prefix.split()
        remaining = s.split()[len(p_words):]
        return (" " + " ".join(remaining)) if remaining else ""

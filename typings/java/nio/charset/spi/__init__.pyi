from typing import Protocol

import java.lang
import java.nio.charset
import java.util

class CharsetProvider:
    def charsetForName(
        self, string: java.lang.String | str
    ) -> java.nio.charset.Charset: ...
    def charsets(self) -> java.util.Iterator[java.nio.charset.Charset]: ...

class __module_protocol__(Protocol):
    # A module protocol which reflects the result of ``jp.JPackage("java.nio.charset.spi")``.

    CharsetProvider: type[CharsetProvider]

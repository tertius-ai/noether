Why the GPL text is present
===========================

Noether itself is MIT-licensed, and the default wheel links no GPL library.
The wheel does convey FLINT, GMP, and MPFR under LGPL-3.0-or-later.

LGPL version 3 is written as additional permissions on top of GPL version 3,
so a complete LGPL-3.0 notice consists of both the LGPL-3.0 and GPL-3.0 texts.
The GPL-3.0.txt file is included for that reason; it is not a declaration that
Noether or its conveyed native libraries are GPL-licensed.

NATIVE_LIBRARIES.txt identifies the libraries actually conveyed, their chosen
licenses, source archives, checksums, and replacement instructions. The wheel
audit rejects linked GPL libraries and forbidden GPL symbols.

"""A boiled down version of UniqueNameGenerator from github.com/inducer/pytools"""

__copyright__ = "Copyright (C) 2009-2013 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Dict, Optional, Set, Iterable, Tuple


# {{{ unique name generation

def generate_unique_names(prefix):
    yield prefix

    try_num = 0
    while True:
        yield "%s_%d" % (prefix, try_num)
        try_num += 1


def generate_numbered_unique_names(
        prefix: str, num: Optional[int] = None) -> Iterable[Tuple[int, str]]:
    if num is None:
        yield (0, prefix)
        num = 0

    while True:
        name = "%s_%d" % (prefix, num)
        num += 1
        yield (num, name)


class UniqueNameGenerator:
    """
    .. automethod:: is_name_conflicting
    .. automethod:: __call__
    """
    def __init__(self,
            existing_names: Optional[Set[str]] = None,
            forced_prefix: str = ""):
        if existing_names is None:
            existing_names = set()

        self.existing_names = existing_names.copy()
        self.forced_prefix = forced_prefix
        self.prefix_to_counter: Dict[str, int] = {}

    def is_name_conflicting(self, name: str) -> bool:
        return name in self.existing_names

    def __call__(self, based_on: str) -> str:
        based_on = self.forced_prefix + based_on

        counter = self.prefix_to_counter.get(based_on, None)

        for counter, var_name in generate_numbered_unique_names(based_on, counter):  # noqa: B007,E501
            if not self.is_name_conflicting(var_name):
                break

        self.prefix_to_counter[based_on] = counter

        self.existing_names.add(var_name)
        return var_name

# }}}

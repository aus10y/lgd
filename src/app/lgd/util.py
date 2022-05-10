#!/usr/bin/env python3
import argparse
import cmd
import csv
import difflib
import io
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import uuid

from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import TimeoutExpired
from typing import (
    Callable,
    Generator,
    Iterable,
    List,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Union,
)

from lgd import data, exceptions

EDITOR = os.environ.get("EDITOR", "vim")
DEBUG = False
EDITOR_POLL_PERIOD = 3


# ----------------------------------------------------------------------------
# Argparse stuff

date_regex: Pattern[str] = re.compile(
    r"(?P<year>[\d]{4})[/\-_.]?(?P<month>[\d]{2})?[/\-_.]?(?P<day>[\d]{2})?(?P<remainder>.*)?"
)


def user_date_components(
    date_str: str,
) -> Tuple[int, Union[int, None], Union[int, None]]:
    # Parse the date into separate fields.
    match = date_regex.match(date_str)
    year, month, day = match["year"], match["month"], match["day"]

    if year is None or match["remainder"]:
        raise argparse.ArgumentTypeError(f"Invalid date format '{date_str}'")

    year = int(year)
    month = int(month) if month is not None else None
    day = int(day) if day is not None else None

    return (year, month, day)


def sql_date_format(dt):
    return dt.strftime("%Y-%m-%d")


def date_range_from_single(date_str) -> Tuple[datetime, datetime]:
    year, month, day = user_date_components(date_str)

    if day is not None and month is not None:
        # Full YYYY-MM-DD, increment day
        date_to = datetime(year, month, day) + timedelta(days=1)
    elif day is None and month is not None:
        # YYYY-MM, increment month
        if month == 12:
            date_to = datetime(year + 1, 1, 1)
        else:
            date_to = datetime(year, month + 1, 1)
    elif day is None and month is None:
        # YYYY, increment year
        date_to = datetime(year + 1, 1, 1)
    else:
        raise Exception("Invalid date format")

    return (datetime(year, month or 1, day or 1), date_to)


def date_range_from_pair(start_str: str, end_str: str) -> Tuple[datetime, datetime]:
    s_y, s_m, s_d = user_date_components(start_str)
    e_y, e_m, e_d = user_date_components(end_str)

    dt_start = datetime(s_y, s_m or 1, s_d or 1)
    dt_end = datetime(e_y, e_m or 1, e_d or 1)

    return (dt_start, dt_end) if dt_start < dt_end else (dt_end, dt_start)


def _date_range_as_utc(
    datetimes: Tuple[datetime, datetime]
) -> Tuple[datetime, datetime]:
    return (
        datetimes[0].astimezone(timezone.utc),
        datetimes[1].astimezone(timezone.utc),
    )


def to_datetime_ranges(
    date_args: Union[List[List[str]], None]
) -> List[Tuple[datetime, datetime]]:
    """Produce a list of (From, To) tuples of strings in YYYY-MM-DD format.
    These dates are inteded to be used in SQL queries, where the dates are
    expected to be in `YYYY-MM-DD` format, the "From" date is inclusive, and
    the "To" date is exclusive.
    """
    if date_args is None:
        return []

    date_ranges = []
    for date_arg in date_args:
        if len(date_arg) == 1:
            date_range = date_range_from_single(*date_arg)
        elif len(date_arg) == 2:
            date_range = date_range_from_pair(*date_arg)
        else:
            raise Exception("`-d/--date` must only be given one or two values")
        date_ranges.append(_date_range_as_utc(date_range))

    return date_ranges


parser = argparse.ArgumentParser(description="A flexible knowledge store.")
parser.add_argument(
    "-t",
    "--tag",
    action="append",
    nargs="*",
    dest="tags",
    help=(
        "Show or tag a message or messages with the given tags.\n"
        " Filter messages by adding one or more tags separated by spaces.\n"
        " Matching messages must contain all given tags.\n"
        " Ex. `-t foo`, `-t foo bar`.\n"
        " Additional flag usage will OR the tag groups together.\n"
        " Ex. `-t foo bar -t baz`.\n"
    ),
)
parser.add_argument(
    "-D",
    "--delete",
    nargs="*",
    type=str,
    help=(
        "Delete note(s) matching the given UUID or UUID prefix. Confirmation"
        " of delete is requested for any notes found matching the value. If"
        " the -y/--yes flag is given, UUID arguments found to have one matching"
        " note will be deleted without a confirmation prompt, and UUID args"
        " matching more than one note will require confirmation from the user."
    ),
)
parser.add_argument(
    "-y",
    "--yes",
    action="store_true",
    default=False,
    dest="confirmation_override",
    help=(
        "Used to provide affirmative confirmation in place of an interactive prompt."
    ),
)
parser.add_argument(
    "-d",
    "--date",
    action="append",
    nargs="+",
    type=str,
    dest="date_ranges",
    help=(
        "Filter by year, month, day."
        " Ex. `-d YYYYMMDD`. The year, month, day fields may optionally be"
        " separated by any of the following characters: `/`, `-`, `_`, `.`."
        " Ex. `--date YYYY/MM/DD`. The year, or year and month fields may be"
        " given without the rest of the data. Ex. `-d YYYY.MM`, `-d YYYY`."
    ),
)
parser.add_argument(
    "-s",
    "--search",
    action="store",
    type=str,
    dest="search",
    help=("Search notes for the given string."),
)
parser.add_argument(
    "-",
    dest="dash",
    action="store_true",
    default=False,
    help=("Take input from STDIN, echo to STDOUT."),
)
parser.add_argument(
    "--tag-associate",
    dest="tag_associate",
    action="append",
    nargs=2,
    metavar=("explicit", "denoted"),
    help=(
        "Create an association between two tags."
        " Any note tagged with the 'explicit' tag, will behave as if it is"
        " also tagged with the 'denoted' tag."
    ),
)
parser.add_argument(
    "--tag-disassociate",
    dest="tag_disassociate",
    action="append",
    nargs=2,
    metavar=("explicit", "denoted"),
    help=("Remove an association between two tags."),
)
parser.add_argument(
    "--plain",
    action="store_true",
    default=False,
    help=("Disable rendering of note metadata in the editor."),
)
parser.add_argument(
    "-S",
    "--statistics",
    action="store_true",
    default=False,
    dest="tag_stats",
    help=("Print tag statistics."),
)
parser.add_argument(
    "--note-import",
    dest="note_file_in",
    action="store",
    type=str,
    help=(
        "Import notes with UUIDs, created_at timestamps, note body, and tags."
        " The specified file should contain comma separated data (CSV) with"
        " the following headers: 'uuid, created_at, body, tags'."
        " Importing a particular file is an idempotent operation."
    ),
)
parser.add_argument(
    "--note-export",
    dest="note_file_out",
    action="store",
    type=str,
    help=(
        "Export notes, with UUIDs, created_at timestamps, note body, and tags."
        " Notes will be exported in a comma separated format (CSV)."
    ),
)
parser.add_argument(
    "--tag-import",
    dest="tag_file_in",
    action="store",
    type=str,
    help=(
        "Import tag associations."
        " The importation of tag associations is idempotent."
        " Import does not overwrite existing tag associations."
    ),
)
parser.add_argument(
    "--tag-export",
    dest="tag_file_out",
    action="store",
    type=str,
    help=(
        "Export tag associations."
        " Tag associations will be exported in a comma separated format (CSV)."
    ),
)
parser.add_argument(
    "-m",
    "--meta",
    dest="metadata",
    action="store_true",
    help=(
        "Print note metadata to the terminal. Note UUIDs, created at"
        " datetimes, and tags are displayed."
    ),
)
parser.add_argument(
    "--stdout",
    dest="stdout",
    action="store_true",
    default=False,
    help=("Print notes to STDOUT."),
)


# ----------------------------------------------------------------------------
# Path

LGD_PATH = Path.home() / Path(".lgd")


def dir_setup():
    # If our dir doesn't exist, create it.
    LGD_PATH.mkdir(mode=0o770, exist_ok=True)


# ----------------------------------------------------------------------------
# Database Setup

_DB_NAME = "logs.db"
_DB_PATH = os.getenv("LGD_DB_PATH")

if _DB_PATH is None:
    DB_PATH = LGD_PATH / Path(_DB_NAME)
else:
    DB_PATH = Path(_DB_PATH).expanduser()

# -----------------------------------------------------------------------------
# Misc. Utilities


class TagPrompt(cmd.Cmd):

    intro = "Enter comma separated tags:"
    prompt = "(tags) "

    def __init__(self, tags: List[str], *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._personal_tags = tags
        self._final_tags = None

    @staticmethod
    def _tag_split(line) -> List[str]:
        # Use a set in order to de-duplicate tags, then convert back to list.
        tags = (tag.strip() for tag in line.split(","))
        tags = {t for t in tags if t}
        return list(tags)

    def default(self, line):
        self._final_tags = self._tag_split(line)

    def postcmd(self, stop, line):
        return True

    def completedefault(self, text, line, begidx, endidx):
        tag = self._tag_split(text)[-1]
        if tag:
            return [t for t in self._personal_tags if t.startswith(tag)]
        else:
            return self._personal_tags

    def completenames(self, text, *ignored):
        # Complete the last tag on the line
        tag = self._tag_split(text)[-1]
        if tag:
            return [t for t in self._personal_tags if t.startswith(tag)]
        else:
            return self._personal_tags

    @property
    def user_tags(self):
        return self._final_tags


class Term:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def header(cls, text: str) -> str:
        return f"{cls.HEADER}{text}{cls.ENDC}"

    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.OKBLUE}{text}{cls.ENDC}"

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.OKGREEN}{text}{cls.ENDC}"

    @classmethod
    def warning(cls, text: str) -> str:
        return f"{cls.WARNING}{text}{cls.ENDC}"

    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.ERROR}{text}{cls.ENDC}"

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.ENDC}"

    @classmethod
    def underline(cls, text: str) -> str:
        return f"{cls.UNDERLINE}{text}{cls.ENDC}"

    @staticmethod
    def apply_where(
        color_func: Callable[[str], str], sub_string: str, text: str
    ) -> str:
        return text.replace(sub_string, color_func(sub_string))


def flatten_tag_groups(tag_groups: Iterable[Tuple[str, ...]]) -> List[str]:
    tags = []
    for group in tag_groups:
        tags.extend(group)
    return tags


def user_confirmation(prompt: str) -> bool:
    response = input(prompt).lower()
    return response == "y" or response == "yes"


def prompt_for_delete(uuid_prefix: str, uuid_full: uuid.UUID, note_body) -> bool:
    # TODO: Improve uuid highlighting. Currently doesn't work for whole uuids.
    uuid_fragment = Term.apply_where(Term.green, uuid_prefix, str(uuid_full)[:8])
    msg_fragment = note_body[:46].replace("\n", "\\n")[:46].ljust(49, ".")

    prompt = f'{Term.warning("Delete")} {uuid_fragment}..., "{msg_fragment}" [Y/n] '
    return user_confirmation(prompt)


def open_temp_logfile(lines: Union[List[str], None] = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
        if lines:
            tf.writelines(line.encode("utf8") for line in lines)
            tf.flush()
        tf.close()

        subprocess.call([*(shlex.split(EDITOR)), tf.name])

        with open(tf.name) as f:
            contents = f.read()

        os.unlink(tf.name)

    return contents


def editor(body: List[str]) -> Generator[Generator[str, None, None], None, None]:
    """
    Launch the users editor and yield the body of the editor as lines of text
    when a change is detected.
    """
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
        tf.writelines(line.encode("utf8") for line in body)
        tf.flush()
        tf.close()

        last_mtime = os.path.getmtime(tf.name)

        # Launch the editor in a subprocess.
        with subprocess.Popen([*(shlex.split(EDITOR)), tf.name]) as proc:
            terminated = False
            while not terminated:
                try:
                    terminated = proc.wait(timeout=EDITOR_POLL_PERIOD) is not None
                except TimeoutExpired:
                    # The editor process has not been closed yet, but the file
                    # contents may have changed.
                    pass

                # Determine if the file has been modified
                curr_mtime = os.path.getmtime(tf.name)
                if curr_mtime == last_mtime:
                    continue
                last_mtime = curr_mtime

                # Read the new contents
                with open(tf.name) as f:
                    yield (line for line in f)

        os.unlink(tf.name)


def format_tag_statistics(cur: sqlite3.Cursor) -> List[str]:
    stats: List[sqlite3.Row] = list(cur)

    TAG = "Tag"
    DENOTED_BY = "Denoted By"
    DENOTES = "Denotes"
    DIRECT = "Direct"
    INDIRECT = "Indirect"

    tag_width = len(TAG)
    denoted_by_width = len(DENOTED_BY)
    denotes_width = len(DENOTES)
    direct_width = len(DIRECT)
    indirect_width = len(INDIRECT)

    for row in stats:
        tag_width = max((tag_width, len(row["tag"])))
        denoted_by_width = max((denoted_by_width, len(row["children"])))
        denotes_width = max((denotes_width, len(row["implies"])))

    STATS_HEAD_TEMPL = " {: ^{tag_w}} | {: ^{direct_w}} | {: ^{indirect_w}} | {: ^{child_w}} | {: ^{impl_w}}"
    STATS_BODY_TEMPL = " {: <{tag_w}} | {: >{direct_w}} | {: >{indirect_w}} | {: <{child_w}} | {: <{impl_w}}"

    stats_table = [
        "",  # Empty line above table
        Term.bold(
            Term.header(
                STATS_HEAD_TEMPL.format(
                    TAG,
                    DIRECT,
                    INDIRECT,
                    DENOTED_BY,
                    DENOTES,
                    tag_w=tag_width,
                    direct_w=direct_width,
                    indirect_w=indirect_width,
                    child_w=denoted_by_width,
                    impl_w=denotes_width,
                )
            )
        ),
    ]

    for row in stats:
        stats_table.append(
            STATS_BODY_TEMPL.format(
                row["tag"],
                row["direct"],
                row["implied"],
                row["children"],
                row["implies"],
                tag_w=tag_width,
                direct_w=direct_width,
                indirect_w=indirect_width,
                child_w=denoted_by_width,
                impl_w=denotes_width,
            )
        )

    stats_table.append("")  # Empty line below table
    return stats_table


def ui_delete_notes(
    conn: sqlite3.Connection,
    uuid_args: List[str],
    notes_given: List[data.Note],
    override: bool = False,
    override_strong: bool = False,
):
    uuid_prefixes = [uuid_arg.replace("-", "") for uuid_arg in uuid_args]

    def confirm_single(uuid_prefix, uuid_full, note_body):
        try:
            return (
                override
                or override_strong
                or prompt_for_delete(uuid_prefix, uuid_full, note_body)
            )
        except (EOFError, KeyboardInterrupt):
            print("")
            sys.exit()

    def confirm_multi(uuid_prefix, uuid_full, note_body):
        try:
            return override_strong or prompt_for_delete(
                uuid_prefix, uuid_full, note_body
            )
        except (EOFError, KeyboardInterrupt):
            print("")
            sys.exit()

    notes_matched = [(str(n.uuid), ((n.uuid, n.body),)) for n in notes_given]
    notes_matched.extend(
        (
            uuid_prefix,
            tuple(
                (n[data.ID], n[data.MSG])
                for n in data.select_msgs_from_uuid_prefix(conn, uuid_prefix)
            ),
        )
        for uuid_prefix in uuid_prefixes
    )

    if not notes_matched:
        print(" - No notes or UUIDs given")

    for uuid_prefix, matches in notes_matched:
        if not matches:
            print(f"'{uuid_prefix}', No note found")
            continue

        if len(matches) == 1:
            get_confirmation = confirm_single
        else:
            get_confirmation = confirm_multi

        for uuid_full, note_body in matches:
            if get_confirmation(uuid_prefix, uuid_full, note_body):
                if data.delete_msg(conn, uuid_full):
                    print(f" - Deleted {uuid_full}")
                else:
                    print(f" - Failed to delete {uuid_full}")


def stdin_note() -> str:
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        lines.append(line)

        # Print the lines back to stdout, so that this tool may be better
        # composed with other command line tools.
        print(line, end="")

    return "".join(lines)


def get_metadata(note: data.Note) -> str:
    tags = ",".join(sorted(note.tags))
    return f'{note.uuid},{note.created_at},"{tags}"'


# -----------------------------------------------------------------------------
# Rendering and Diffing logs


class EditorView:

    _TAG_REGEX: Pattern[str] = re.compile(r".*\([Tt]ags:\s*(?P<tags>.*)\)")

    def __init__(
        self,
        notes: Iterable[data.Note],
        tag_groups: List[Tuple[str, ...]],
        expanded_tag_groups: List[Tuple[str, ...]],
        style: bool = True,
        top_header: bool = True,
    ):
        """
        logs: A list/tuple, of 2-tuples (uuid, message)
        tags: The tags used to find the given logs. A list of lists of tags.
        """
        self.notes = list(notes)
        self.tag_groups = tag_groups
        self.expanded_tag_groups = expanded_tag_groups
        self._tags_flat = flatten_tag_groups(tag_groups)
        self._styled = style
        self._lines = []
        self._line_map = []
        self._render(style, top_header)  # Set up self._lines and self._lines_map

    def _render(self, style: bool, top_header: bool):
        # Header
        if style and top_header and self.tag_groups:
            self._lines.append(EditorView._editor_header(self.expanded_tag_groups))

        # Body
        linenum_init, linenum_last = None, None
        for note in self.notes:
            # Set the header for each message.
            if style:
                self._lines.extend(EditorView._note_header(note))
                linenum_init = len(self._lines) - 1
            else:
                linenum_init = len(self._lines)

            self._lines.extend(note.body.splitlines(keepends=True))

            if style:
                self._lines.extend(EditorView._note_footer(note))

            linenum_last = len(self._lines)

            self._line_map.append((note.uuid, linenum_init, linenum_last))

        # Footer
        if style:
            self._lines.extend(EditorView._editor_footer(set(self._tags_flat)))

    @staticmethod
    def _editor_header(tag_groups: List[Tuple[str, ...]]):
        _tag_groups = (", ".join(group) for group in tag_groups)
        tags_together = " || ".join(f"<{tg}>" for tg in _tag_groups)
        header = f"# TAGS: {tags_together}\n"
        return header

    @staticmethod
    def _note_header(note: data.Note) -> Tuple[str, ...]:
        tags_str = "" if not note.tags else ", ".join(sorted(note.tags))
        id_str = str(note.uuid)[:8]  # Only show first eight digits of UUID
        header = (
            f'{79*"-"}\n',
            f"# {note.created_at}\n",
            f"[ID: {id_str}]: # (Tags: {tags_str})\n",
            f"\n",
        )
        return header

    @staticmethod
    def _note_footer(note: data.Note) -> Tuple[str, ...]:
        # Add a newline, but only if there's not already an empty line at the
        # end of the note body.
        if note.body[-2:] == "\n\n":
            return ()
        return ("\n",)

    @staticmethod
    def _editor_footer(tags: Iterable[str]) -> Tuple[str, ...]:
        footer = (
            f'{79*"-"}\n',
            f"# Enter new log message below\n",
            f'# Tags: {", ".join(tags)}\n',
            "\n",
        )
        return footer

    @property
    def rendered(self) -> List[str]:
        return self._lines

    @staticmethod
    def _is_addition(line: str) -> bool:
        return line.startswith("+ ")

    @staticmethod
    def _is_removal(line: str) -> bool:
        return line.startswith("- ")

    @staticmethod
    def _is_intraline(line: str) -> bool:
        return line.startswith("? ")

    @staticmethod
    def _is_emptyline(line: str) -> bool:
        return line == "  \n"

    @staticmethod
    def _is_modification(line: str) -> bool:
        return EditorView._is_addition(line) or EditorView._is_removal(line)

    @staticmethod
    def _enumerate_diff(
        diff_lines: Iterable[str],
    ) -> Generator[Tuple[int, str], None, None]:
        first_line = True
        line_num = 0

        for line in diff_lines:
            if EditorView._is_intraline(line):
                # These intraline differences are not needed.
                continue

            if EditorView._is_addition(line):
                yield (line_num, line)
            else:
                if first_line:
                    yield (line_num, line)
                    first_line = False
                else:
                    line_num += 1
                    yield (line_num, line)

    @staticmethod
    def _print_diff_info(
        line_num: int,
        msg_uuid: uuid.UUID,
        line_from: int,
        line_to: int,
        text: str,
        debug: bool = False,
    ):
        if debug:
            print(
                (
                    f"line: {line_num:>4}, msg_uuid: {str(msg_uuid)},"
                    f" ({str(line_from):>4}, {str(line_to):>4}): {text}"
                ),
                end="",
            )

    @staticmethod
    def _is_new_tag_line(line: str) -> bool:
        TAG_LINE = "+ # Tags:"
        return line.startswith(TAG_LINE)

    @classmethod
    def _parse_tags(cls, line) -> Union[None, Set[str]]:
        match = cls._TAG_REGEX.match(line)
        if match is None:
            return None
        raw_tags = (t.strip() for t in match["tags"].split(","))
        return {t for t in raw_tags if t}

    @staticmethod
    def _parse_new_tags(line: str) -> Set[str]:
        TAG_LINE = "+ # Tags:"
        raw_tags = (t.strip() for t in line[len(TAG_LINE) :].split(","))
        return {t for t in raw_tags if t}

    @staticmethod
    def _diff_has_modifications(lines: List[str]) -> bool:
        return any(line.startswith("- ") or line.startswith("+ ") for line in lines)

    def diff(
        self, other: Sequence[str], debug=False
    ) -> Generator[Tuple[Union[None, data.Note], data.Note], None, None]:
        line_num, diff_index = 0, 0

        diff = difflib.ndiff(self._lines, other)
        diff = list(EditorView._enumerate_diff(diff))

        line_num, text = diff[diff_index]

        for msg_uuid, line_from, line_to in self._line_map:
            msg_diff = []
            tags_original, tags_updated = None, None

            advance = 0
            for line_num, text in diff[diff_index:]:
                if line_num < line_from:
                    # Check for tag changes
                    tags = EditorView._parse_tags(text)
                    if tags is not None:
                        if text.startswith("-"):
                            tags_original = tags
                        elif text.startswith("+"):
                            tags_updated = tags
                        else:
                            tags_original = tags
                elif line_num == line_from:
                    # Handle leading synthetic newline
                    if self._styled:
                        if EditorView._is_addition(text):
                            msg_diff.append(text)
                    else:
                        msg_diff.append(text)
                elif line_from < line_num < (line_to - 1):
                    # Handle body of note
                    msg_diff.append(text)
                elif line_num == (line_to - 1):
                    # Handle trailing synthetic newline
                    if self._styled:
                        if EditorView._is_addition(text):
                            msg_diff.append(text)
                    else:
                        msg_diff.append(text)
                elif line_to <= line_num:
                    break

                EditorView._print_diff_info(
                    line_num, msg_uuid, line_from, line_to, text, debug=debug
                )

                advance += 1

            diff_index += advance

            # Continue on if no change detected
            tags_updated = tags_updated if tags_updated else tags_original

            if (
                EditorView._diff_has_modifications(msg_diff)
                or tags_original != tags_updated
            ):
                yield (
                    data.Note(
                        msg_uuid,
                        None,
                        "".join(difflib.restore(msg_diff, 1)),
                        tags_original,
                    ),
                    data.Note(
                        msg_uuid,
                        None,
                        "".join(difflib.restore(msg_diff, 2)),
                        tags_updated,
                    ),
                )
            else:
                continue

        # New msg
        msg_diff = []
        new_tags = set(self._tags_flat)
        for line_num, text in diff[diff_index:]:
            EditorView._print_diff_info(line_num, None, None, None, text, debug=debug)
            if EditorView._is_new_tag_line(text):
                new_tags = EditorView._parse_new_tags(text)
            elif EditorView._is_addition(text):
                msg_diff.append(text)

        # Create and append the new msg, if it exists
        if msg_diff:
            yield (
                None,
                data.Note(None, None, "".join(difflib.restore(msg_diff, 2)), new_tags),
            )


# -----------------------------------------------------------------------------


def handle_tag_associate(
    conn: sqlite3.Connection,
    to_associate: Iterable[Tuple[str, str]],
    quiet: bool = False,
) -> Tuple[int, int]:
    inserted, existing = 0, 0
    for explicit, implicit in to_associate:
        try:
            data.insert_tag_relation(conn, explicit, implicit, quiet=True)
        except exceptions.LgdException as e:
            print(Term.warning(str(e)))
        except sqlite3.IntegrityError as e:
            existing += 1
            if "unique" in str(e).lower() and not quiet:
                print(
                    Term.warning(
                        f"Tag relation '{explicit}' -> '{implicit}' already exists!"
                    )
                )
        else:
            inserted += 1
            if not quiet:
                print(Term.green(f"Created '{explicit}' -> '{implicit}' relation"))
    return (inserted, existing)


def handle_tag_disassociate(
    conn: sqlite3.Connection, to_disassociate: Iterable[List[str]]
) -> None:
    for explicit, implicit in to_disassociate:
        try:
            removed = data.remove_tag_relation(conn, explicit, implicit)
        except exceptions.LgdException as e:
            print(Term.warning(str(e)))
        else:
            if removed:
                print(Term.green(f"Removed '{explicit}' -> '{implicit}' relation"))
            else:
                print(
                    Term.warning(
                        f"Relation '{explicit}' -> '{implicit}' doesn't exist!"
                    )
                )


def note_export(
    conn: sqlite3.Connection, notes: List[data.Note], outfile: io.TextIOWrapper
) -> int:
    writer = csv.DictWriter(outfile, data.Note._fields)
    writer.writeheader()
    for note in notes:
        # For the CSV file, for the tags to be a comma separated str.
        note = note._replace(tags=",".join(note.tags))
        writer.writerow(note._asdict())

    return len(notes)


def note_import(conn: sqlite3.Connection, infile: io.TextIOWrapper) -> Tuple[int, int]:
    inserted = 0
    updated = 0

    reader = csv.DictReader(infile)
    if reader.fieldnames is None or set(reader.fieldnames) != set(data.Note._fields):
        raise exceptions.CSVError(
            "Invalid CSV columns; columns must be: uuid,created_at,body,tags"
        )

    for row in reader:
        try:
            row[data.CREATED_AT] = datetime.strptime(
                row[data.CREATED_AT], "%Y-%m-%d %H:%M:%S"
            )
        except ValueError as e:
            raise exceptions.CSVError(
                "Invalid 'created_at' format; timestamp must be 'YYYY-MM-DD HH:MM:SS'."
            ) from e

        try:
            row[data.ID] = uuid.UUID(row[data.ID])
        except ValueError as e:
            raise exceptions.CSVError(
                "Invalid 'uuid' format; value must be string of hex digits. Curly braces, hyphens, and a URN prefix are all optional."
            ) from e

        try:
            note = data.Note(**row)
        except TypeError as e:
            raise exceptions.CSVError(
                "Invalid CSV columns; columns must be: uuid,created_at,body,tags"
            ) from e

        tags = note.tags.split(",")

        if data.msg_exists(conn, note.uuid):
            # Update
            updated += int(data.update_msg(conn, note.uuid, note.body))
        else:
            # Insert
            _ = data.insert_note(
                conn,
                note.body,
                note_uuid=note.uuid,
                created_at=note.created_at,
            )
            inserted += 1

        tag_uuids = data.insert_tags(conn, tags)
        data.insert_asscs(conn, note.uuid, tag_uuids)

    return (inserted, updated)


def tag_export(conn: sqlite3.Connection, outfile: io.TextIOWrapper) -> int:
    tag_relations = data.select_related_tags_all(conn)
    writer = csv.writer(outfile)
    writer.writerow(("tag_direct", "tag_indirect"))
    writer.writerows(tag_relations)
    return len(tag_relations)


def tag_import(conn: sqlite3.Connection, infile: io.TextIOWrapper) -> Tuple[int, int]:
    reader = csv.DictReader(infile)
    relations = ((row["tag_direct"], row["tag_indirect"]) for row in reader)
    return handle_tag_associate(conn, relations, quiet=True)

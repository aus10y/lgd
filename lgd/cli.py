#!/usr/bin/env python3
import argparse

import os
import sqlite3
import sys

from lgd import data
from lgd import util as lgd
from lgd.exceptions import CSVError


EDITOR = os.environ.get("EDITOR", "vim")
DEBUG = False
EDITOR_POLL_PERIOD = 3


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


def main():
    args = parser.parse_args()
    args.date_ranges = lgd.to_datetime_ranges(args.date_ranges)

    lgd.dir_setup()
    conn = data.get_connection(str(lgd.DB_PATH))

    try:
        data.db_setup(conn, data.DB_MIGRATIONS)
    except sqlite3.Error as e:
        print(lgd.Term.error(str(e)))
        print("Failed to finish database setup!")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # Apply the users filters to select the notes/messages
    tag_groups = [tg for tg in (args.tags or []) if tg]
    expanded_tag_groups = data.expand_tag_groups(conn, tag_groups)

    messages = list(
        data.NoteQuery.select(
            tag_groups=expanded_tag_groups,
            date_ranges=args.date_ranges,
            text=args.search,
        ).execute(conn)
    )

    notes_requested = any((tag_groups, args.date_ranges, args.search))

    if args.note_file_in:
        with open(args.note_file_in, "r") as infile:
            try:
                with conn:
                    inserted, updated = lgd.note_import(conn, infile)
            except CSVError as e:
                sys.exit(lgd.Term.error(str(e)))
        print(
            f" - Inserted {inserted}, updated {updated} notes from {args.note_file_in}"
        )

    if args.tag_file_in:
        with open(args.tag_file_in, "r") as infile:
            with conn:
                inserted, existing = lgd.tag_import(conn, infile)
                total = inserted + existing
        print(f" - Inserted {inserted} of {total} relations from {args.tag_file_in}")

    if args.note_file_out:
        with open(args.note_file_out, "w") as outfile:
            num = lgd.note_export(conn, messages, outfile)
        print(f" - Exported {num} notes to {args.note_file_out}")

    if args.tag_file_out:
        with open(args.tag_file_out, "w") as outfile:
            num = lgd.tag_export(conn, outfile)
        print(f" - Exported {num} tag relations to {args.tag_file_out}")

    if args.tag_stats:
        stats = lgd.format_tag_statistics(data.tag_statistics(conn))
        for line in stats:
            print(line)

    if args.tag_associate or args.tag_disassociate:
        with conn:
            lgd.handle_tag_associate(conn, (args.tag_associate or []))
            lgd.handle_tag_disassociate(conn, (args.tag_disassociate or []))

    if args.delete is not None:
        notes_searched = messages if notes_requested else []
        with conn:
            lgd.ui_delete_notes(
                conn, args.delete, notes_searched, args.confirmation_override
            )

    if any(
        (
            args.note_file_in,
            args.note_file_out,
            args.tag_file_in,
            args.tag_file_out,
            args.tag_stats,
            args.delete is not None,
            args.tag_associate or args.tag_disassociate,
        )
    ):
        conn.close()
        sys.exit()

    # ------------------------------------------------------------------------
    # If reading from stdin
    if args.dash:
        msg = lgd.stdin_note()

        with conn:
            msg_uuid = data.NoteQuery.insert(msg).execute(conn)
            if args.tags:
                tags = lgd.flatten_tag_groups(args.tags)
                tag_uuids = data.insert_tags(conn, tags)
                data.insert_asscs(conn, msg_uuid, tag_uuids)

        print(f"Saved as message ID {msg_uuid}")
        conn.close()
        sys.exit()

    # ------------------------------------------------------------------------
    # Print Note metadata (uuid, created_at, tags)
    if args.metadata:
        for msg in messages:
            print(lgd.get_metadata(msg))
        conn.close()
        sys.exit()

    # ------------------------------------------------------------------------
    # Display notes matching filters
    if args.tags or args.date_ranges or args.search:

        # The users editor is opened here, and may be open for an extended
        # period of time. Close the database and get a new connection
        # afterward.
        conn.close()

        editor_view = lgd.EditorView(
            messages, tag_groups, expanded_tag_groups, style=(not args.plain)
        )

        if args.stdout:
            # Print to stdout and then exit
            print("".join(editor_view.rendered))
            sys.exit()

        new_note = None
        notifications = []

        for body_lines in lgd.editor(editor_view.rendered):
            conn = data.get_connection(str(lgd.DB_PATH))

            diffs = editor_view.diff(list(body_lines), debug=DEBUG)
            for note_orig, note_mod in diffs:
                if note_orig is None:
                    new_note = note_mod
                    continue

                assert note_orig.uuid == note_mod.uuid

                # Begin transaction to update note
                with conn:
                    if note_orig.body != note_mod.body:
                        # Update body of note
                        data.NoteQuery.update(note_mod.uuid, note_mod.body).execute(
                            conn
                        )

                    if note_orig.tags != note_mod.tags:
                        # Update associated tags
                        tags_add = note_mod.tags - note_orig.tags
                        if tags_add:
                            tag_uuids = data.insert_tags(conn, tags_add)
                            data.insert_asscs(conn, note_mod.uuid, tag_uuids)

                        tags_sub = note_orig.tags - note_mod.tags
                        if tags_sub:
                            tag_uuids = {
                                t[0] for t in data.select_tags(conn, tuple(tags_sub))
                            }
                            data.remove_asscs(conn, note_mod.uuid, tag_uuids)

                notifications.append(f"Saved changes to message ID {note_mod.uuid}")

            # Reset the EditorView to account for the modified notes.
            editor_view = lgd.EditorView(
                data.NoteQuery.select(uuids=[m.uuid for m in messages]).execute(conn),
                tag_groups,
                expanded_tag_groups,
                style=(not args.plain),
            )

            conn.close()

        if new_note is not None:
            conn = data.get_connection(str(lgd.DB_PATH))
            with conn:
                msg_uuid = data.NoteQuery.insert(new_note.body).execute(conn)
                tag_uuids = data.insert_tags(conn, new_note.tags)
                data.insert_asscs(conn, msg_uuid, tag_uuids)

                notifications.append(f"Saved additional message as ID {msg_uuid}")
            conn.close()

        # Messages / notification to user are deferred until the editor is
        # closed, so that output to stdout (via print) will not interfere
        # with terminal editors.
        for notification in notifications:
            print(notification)

        sys.exit()

    # ------------------------------------------------------------------------
    # Quick note
    msg = lgd.open_temp_logfile()
    if not msg:
        print("No message created...")
        sys.exit()

    msg_uuid = data.NoteQuery.insert(msg).execute(conn)
    conn.commit()

    # Collect tags via custom prompt
    tag_prompt = lgd.TagPrompt(data.select_all_tags(conn))
    tag_prompt.cmdloop()

    if tag_prompt.user_tags:
        with conn:
            tag_uuids = data.insert_tags(conn, tag_prompt.user_tags)
            data.insert_asscs(conn, msg_uuid, tag_uuids)

    print(f"Saved as message ID {msg_uuid}")
    conn.close()

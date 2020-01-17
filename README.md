# lgd
Personal Knowledge Store

lgd ('logged') is a command line tool for recording and tagging thoughts, notes, or anything text that you want to capture.
This is an experiment in scratching my own itch, by building what I think I want in a notes/journal tool.
There are many tools like it, but this one is mine.

## Goals

I want lgd to be simple, effective, hackable, and easy to get your data out of.

To that end, lgd currently:
- Is built in a single python file
- Uses only the standard library.
- Launches your shell's `$EDITOR`, with which you record your notes.
- Store your notes in a sqlite database.

These details may someday change, but not without a compelling reason.

## Installation & Use

Currently only supports Linux (due to standard lib dependency upon the `readline` lib).  
Python 3.6+

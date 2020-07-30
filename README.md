# lgd

**Your Personal Knowledge Store**

lgd ('logged') is a command line tool for capturing and tagging notes (or anything text).

## Features

- Tags
  - Tag your notes
  - Create tag hierarchies
- Search & Filtering
  - Complex filtering by tags and date
  - Full text search (Coming Soon)
- Supports Markdown*
- Use your favorite terminal text editor for note capture
- Record notes via stdin
- Edit notes or their associated tags
- Simple, no lock-in
  - Single Python module, no 3rd party dependencies
  - Export your notes and tag hierarchies to CSV
  - Sqlite database under the hood

*Notes are opened in your text editor as a temporary file with a `.md` file extension, leveraging your editors support for Markdown.

## Examples

### Record a Note

**Create a quick note:**  
`$ lgd`  
**TODO: gif**

**Add a note while showing existing notes for a given tag:**  
`$ lgd -t example`  
**TODO: gif**

### Filter Notes

**Filter by tags:**  
`$ lgd -t foo bar`, notes must have both tags (AND)  
`$ lgd -t foo -t bar`, notes must have either tag (OR)  
**TODO: gif**

**Filter notes by date created:**  
Filter by year, year-month, year-month-day, or custom ranges.  
`$ lgd -d 2020`  
`$ lgd -d 20200401`  
`$ lgd -d 20200123 20200125`  
**TODO: gif**

**Filter by tags and dates:**  
`$ lgd -t foo bar -d 2019`  
**TODO: gif**

### Editing Notes

Edit multiple notes in your text editor, the changes will be saved back to each individual note in the database.  
**TODO: gif**

Edit the tags associated with a note.  
**TODO: gif**

## Installation

Requires Python 3.6+.  

**Install**  
- `$ git clone git@github.com:aus10y/lgd.git` (Clone with SSH)
- `$ cd lgd`
- `$ make install`

**Uninstall**  
- `$ make uninstall`

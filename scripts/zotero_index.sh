#!/usr/bin/env bash
# Zotero library index — resolves each paper to its PDF on disk.
#
# Reads the local Zotero SQLite catalog READ-ONLY (immutable=1), so it is safe
# to run while Zotero is open and never modifies the library.
#
# Usage:
#   scripts/zotero_index.sh                 # all papers with a PDF
#   scripts/zotero_index.sh tesis           # only papers in the "tesis" collection
#
# Output columns (tab-separated): YEAR <TAB> AUTHORS <TAB> TITLE <TAB> ABS_PDF_PATH

set -euo pipefail

ZOTERO_DIR="${ZOTERO_DIR:-$HOME/Zotero}"
DB="file:${ZOTERO_DIR}/zotero.sqlite?immutable=1"
COLLECTION="${1:-}"

collection_filter=""
if [[ -n "$COLLECTION" ]]; then
  # Restrict to attachments whose parent item is in the named collection.
  collection_filter="AND ia.parentItemID IN (
      SELECT ci.itemID FROM collectionItems ci
      JOIN collections c ON ci.collectionID = c.collectionID
      WHERE c.collectionName = '${COLLECTION//\'/\'\'}'
  )"
fi

sqlite3 "$DB" <<SQL | sed "s#\tstorage/#\t${ZOTERO_DIR}/storage/#"
.mode tabs
SELECT
  COALESCE(yv.value, '')                                   AS year,
  COALESCE((
     SELECT GROUP_CONCAT(cr.lastName, ', ')
     FROM itemCreators ic
     JOIN creators cr ON ic.creatorID = cr.creatorID
     WHERE ic.itemID = ia.parentItemID
     ORDER BY ic.orderIndex
     LIMIT 3), '')                                         AS authors,
  COALESCE(tv.value, '(no title)')                         AS title,
  'storage/' || ai.key || '/' || substr(ia.path, 9)        AS pdf
FROM itemAttachments ia
JOIN items ai ON ia.itemID = ai.itemID
LEFT JOIN itemData td ON td.itemID = ia.parentItemID
     AND td.fieldID = (SELECT fieldID FROM fields WHERE fieldName='title')
LEFT JOIN itemDataValues tv ON td.valueID = tv.valueID
LEFT JOIN itemData yd ON yd.itemID = ia.parentItemID
     AND yd.fieldID = (SELECT fieldID FROM fields WHERE fieldName='date')
LEFT JOIN itemDataValues yvraw ON yd.valueID = yvraw.valueID
LEFT JOIN (SELECT valueID, substr(value,1,4) AS value FROM itemDataValues) yv
     ON yd.valueID = yv.valueID
WHERE ia.contentType = 'application/pdf'
  AND ia.path LIKE 'storage:%'
  ${collection_filter}
ORDER BY title;
SQL

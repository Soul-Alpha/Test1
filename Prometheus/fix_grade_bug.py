"""Fix Bug 1: insert grade win/loss tracking block in the close event handler."""
fp = r'C:\Users\Chaba\Documents\tradingBots\Prometheus\live_bot\trader.py'

with open(fp, encoding='utf-8') as f:
    txt = f.read()

# The block to find ends just before OB learning comment.
# We use a search string that avoids the garbled em-dash characters by searching
# for the ob_dir line which is plain ASCII.
search = (
    '                ds["wins" if won else "losses"] += 1\n'
    '\n'
    '                # OB learning'
)

insert_block = (
    '                ds["wins" if won else "losses"] += 1\n'
    '\n'
    '                # per-grade win/loss tracking (Bug 1 fix)\n'
    '                _g = self._entry_grade.pop(deal.position_id, None)\n'
    '                if _g:\n'
    '                    _gs = _LEARNING["grade_stats"].setdefault(\n'
    '                        _g, {"seen": 0, "acted": 0, "wins": 0, "losses": 0}\n'
    '                    )\n'
    '                    _gs.setdefault("wins", 0)\n'
    '                    _gs.setdefault("losses", 0)\n'
    '                    _gs["wins" if won else "losses"] += 1\n'
    '\n'
    '                # OB learning'
)

count = txt.count(search)
print(f'Match count: {count}')

if count == 1:
    new_txt = txt.replace(search, insert_block, 1)
    with open(fp, 'w', encoding='utf-8', newline='') as f:
        f.write(new_txt)
    print('Done.')
elif count == 0:
    # Try with CRLF
    search_crlf = search.replace('\n', '\r\n')
    count_crlf = txt.count(search_crlf)
    print(f'CRLF match count: {count_crlf}')
    if count_crlf == 1:
        insert_crlf = insert_block.replace('\n', '\r\n')
        new_txt = txt.replace(search_crlf, insert_crlf, 1)
        with open(fp, 'w', encoding='utf-8', newline='') as f:
            f.write(new_txt)
        print('Done with CRLF.')
    else:
        # Find and show context
        idx = txt.find('                ds["wins" if won else "losses"] += 1')
        print(f'Found ds["wins"...] at index: {idx}')
        if idx >= 0:
            snippet = txt[idx:idx+200]
            print(repr(snippet))
else:
    print(f'Multiple matches: {count}')

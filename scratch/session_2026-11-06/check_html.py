import html.parser, sys
class P(html.parser.HTMLParser):
    void = {"meta", "link", "br", "img", "input", "hr"}
    def __init__(s):
        super().__init__(); s.stack = []; s.err = []
    def handle_starttag(s, t, a):
        if t not in s.void: s.stack.append(t)
    def handle_endtag(s, t):
        if t in s.void: return
        if s.stack and s.stack[-1] == t: s.stack.pop()
        else: s.err.append((t, s.stack[-3:]))
p = P(); p.feed(open(sys.argv[1], encoding="utf-8").read())
print("unclosed:", p.stack[:5], "mismatches:", p.err[:5])
